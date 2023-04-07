# ------------------------------------------------------------------------


'''
NAFSSR: Stereo Image Super-Resolution Using NAFNet

@InProceedings{Chu2022NAFSSR,
  author    = {Xiaojie Chu and Liangyu Chen and Wenqing Yu},
  title     = {NAFSSR: Stereo Image Super-Resolution Using NAFNet},
  booktitle = {CVPRW},
  year      = {2022},
}
'''

import numpy as np
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
import torch
import torch.nn as nn


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)



class SCATM(nn.Module):
    '''
    Stereo Cross Attention with Temperature Module (SCATM)
    '''

    def __init__(self, c, temprature=1.5):
        super().__init__()
        self.scale = c ** -0.5
        self.temprature = temprature
        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3)  # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax((attention * self.temprature), dim=-1), V_r)  # B, H, W, c
        F_l2r = torch.matmul(torch.softmax((attention * self.temprature).permute(0, 1, 3, 2), dim=-1),
                             V_l)  # B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r



class DropPath(nn.Module):
    def __init__(self, drop_rate, module):
        super().__init__()
        self.drop_rate = drop_rate
        self.module = module

    def forward(self, *feats):
        if self.training and np.random.rand() < self.drop_rate:
            return feats

        new_feats = self.module(*feats)
        factor = 1. / (1 - self.drop_rate) if self.training else 1.

        if self.training and factor != 1.:
            new_feats = tuple([x + factor * (new_x - x) for x, new_x in zip(feats, new_feats)])
        return new_feats


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma



class NAFTBlockSR(nn.Module):
    '''
    NAFTBlock for Super-Resolution
    '''

    def __init__(self, c, fusion=False, drop_out_rate=0., temperature=1.5):
        super().__init__()
        self.blk = NAFBlock(c, drop_out_rate=drop_out_rate)
        self.fusion = SCATM(c, temprature=temperature) if fusion else None

    def forward(self, *feats):
        feats = tuple([self.blk(x) for x in feats])
        if self.fusion:
            feats = self.fusion(*feats)
        return feats


class MultiSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

# Stereo Cross Global Learnable Attention
class SCGLA(nn.Module):

    def __init__(self, n_hashes=4, channels=64, k_size=3, reduction=4, chunk_size=144, res_scale=1):
        super(SCGLA,self).__init__()
        self.chunk_size = chunk_size
        self.n_hashes = n_hashes
        self.reduction = reduction
        self.res_scale = res_scale
        self.conv_match = nn.Conv2d(channels,channels//reduction,k_size,padding=1,stride=1,bias=False)
        self.conv_assembly = nn.Conv2d(channels,channels,k_size,padding=1,stride=1,bias=False)
        self.conv_assembly_fc = nn.Conv2d(channels,channels,k_size,padding=1,stride=1,bias=False)
        self.fc = nn.Sequential(
            nn.Linear(channels, chunk_size),
            nn.ReLU(inplace=True),
            nn.Linear(chunk_size, chunk_size)
        )

    # Super-Bit Locality-Sensitive Hashing
    def SBLSH(self, hash_buckets, x):
        #x: [N,2*H*W,C]
        N = x.shape[0]
        device = x.device

        #generate random rotation matrix
        rotations_shape = (1, x.shape[-1], self.n_hashes) #[1,C,n_hashes,hash_buckets//2]
        # assert rotations_shape[1] > rotations_shape[2]*rotations_shape[3]
        random_rotations = torch.nn.init.orthogonal_(torch.empty(x.shape[-1], hash_buckets))
        for _ in range(self.n_hashes-1):
            random_rotations = torch.cat([random_rotations, torch.nn.init.orthogonal_(torch.empty(x.shape[-1],hash_buckets))], dim=-1)
        # Training under multi-gpu: random_rotations.cuda() -> andom_rotations.to(x.device) (suggested by Breeze-Zero from github: https://github.com/laoyangui/DLSN/issues/2)
        random_rotations = random_rotations.reshape(rotations_shape[0], rotations_shape[1], rotations_shape[2], hash_buckets).expand(N, -1, -1, -1).to(device) #[N, C, n_hashes, hash_buckets]
        rotated_vecs = torch.einsum('btf,bfhi->bhti', x, random_rotations) #[N, n_hashes, H*W, hash_buckets]

        #get hash codes
        hash_codes = torch.argmax(rotated_vecs, dim=-1) #[N,n_hashes,H*W]

        #add offsets to avoid hash codes overlapping between hash rounds
        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * hash_buckets, (1, -1, 1))
        hash_codes = torch.reshape(hash_codes + offsets, (N, -1,)) #[N,n_hashes*H*W]

        return hash_codes

    def add_adjacent_buckets(self, x):
        x_extra_back = torch.cat([x[:,:,-1:, ...], x[:,:,:-1, ...]], dim=2)
        x_extra_forward = torch.cat([x[:,:,1:, ...], x[:,:,:1,...]], dim=2)
        return torch.cat([x, x_extra_back,x_extra_forward], dim=3)

    def forward(self, input1,input2):

        N,_,H,W = input1.shape
        x_embed1 = self.conv_match(input1).view(N, -1, H * W).contiguous().permute(0,2,1)  # N ,HW, C
        x_embed2 = self.conv_match(input2).view(N, -1, H * W).contiguous().permute(0, 2, 1)  # N ,HW, C
        x_embed = torch.cat([x_embed1,x_embed2],dim=1)
        y_embed1 = self.conv_assembly(input1).view(N,-1,H*W).contiguous().permute(0,2,1) # N ,HW, C
        y_embed2 = self.conv_assembly(input2).view(N, -1, H * W).contiguous().permute(0, 2, 1)  # N ,HW, C
        y_embed = torch.cat([y_embed1, y_embed2], dim=1)
        fc_embed1 = self.conv_assembly_fc(input1).view(N,-1,H*W).contiguous().permute(0,2,1) # N ,HW, C
        fc_embed2 = self.conv_assembly_fc(input2).view(N, -1, H * W).contiguous().permute(0, 2, 1)  # N ,HW, C
        fc_embed = torch.cat([fc_embed1, fc_embed2], dim=1)

        # 2*H*W C
        L,C = x_embed.shape[-2:]
        #number of hash buckets/hash bits
        hash_buckets = min(L//self.chunk_size + (L//self.chunk_size)%2, 128)

        #get assigned hash codes/bucket number
        hash_codes = self.SBLSH(hash_buckets, x_embed) #[N,n_hashes*2*H*W]
        hash_codes = hash_codes.detach() #[N,n_hashes*2*H*W]

        #group elements with same hash code by sorting
        _, indices = hash_codes.sort(dim=-1) #[N,n_hashes*2*H*W]
        _, undo_sort = indices.sort(dim=-1) #undo_sort to recover original order
        mod_indices = (indices % L) #now range from (0->H*W)

        x_embed_sorted = batched_index_select(x_embed, mod_indices) #[N,n_hashes*2H*W,C]
        y_embed_sorted = batched_index_select(y_embed, mod_indices) #[N,n_hashes*2H*W,C]
        fc_embed_embed_sorted = batched_index_select(fc_embed, mod_indices) #[N,n_hashes*2H*W,C]

        #pad the embedding if it cannot be divided by chunk_size
        padding = self.chunk_size - L%self.chunk_size if L%self.chunk_size!=0 else 0
        x_att_buckets = torch.reshape(x_embed_sorted, (N, self.n_hashes,-1, C)) #[N, n_hashes, 2*H*W,C]
        y_att_buckets = torch.reshape(y_embed_sorted, (N, self.n_hashes,-1, C*self.reduction))
        fc_att_buckets = torch.reshape(fc_embed_embed_sorted, (N, self.n_hashes,-1, C*self.reduction))

        if padding:
            pad_x = x_att_buckets[:,:,-padding:,:].clone()
            pad_y = y_att_buckets[:,:,-padding:,:].clone()
            pad_fc = fc_att_buckets[:,:,-padding:,:].clone()
            x_att_buckets = torch.cat([x_att_buckets,pad_x],dim=2)
            y_att_buckets = torch.cat([y_att_buckets,pad_y],dim=2)
            fc_att_buckets = torch.cat([fc_att_buckets,pad_fc],dim=2)

        x_att_buckets = torch.reshape(x_att_buckets,(N,self.n_hashes,-1,self.chunk_size,C)) #[N, n_hashes, num_chunks, chunk_size, C] # q
        y_att_buckets = torch.reshape(y_att_buckets,(N,self.n_hashes,-1,self.chunk_size, C*self.reduction))
        fc_att_buckets = torch.reshape(fc_att_buckets,(N,self.n_hashes,-1,self.chunk_size, C*self.reduction))

        x_match = F.normalize(x_att_buckets, p=2, dim=-1,eps=5e-5)

        #allow attend to adjacent buckets
        x_match = self.add_adjacent_buckets(x_match) #[N, n_hashes, num_chunks, chunk_size*3, C]  # k
        y_att_buckets = self.add_adjacent_buckets(y_att_buckets)
        fc_att_buckets = self.add_adjacent_buckets(fc_att_buckets)
        fc_raw_score = self.fc(fc_att_buckets).permute(0,1,2,4,3) #[N, n_hashes, num_chunks, chunk_size, chunk_size*3]

        #unormalized attention score
        raw_score = torch.einsum('bhkie,bhkje->bhkij', x_att_buckets, x_match) + fc_raw_score #[N, n_hashes, num_chunks, chunk_size, chunk_size*3]

        #softmax    self.sigmoid2(self.fc2(self.sigmoid1(self.fc1(x_att_buckets))))
        bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
        score = torch.exp(raw_score - bucket_score) #(after softmax)

        ret = torch.einsum('bukij,bukje->bukie', score, y_att_buckets) #[N, n_hashes, num_chunks, chunk_size, C*self.reduction]
        bucket_score = torch.reshape(bucket_score,[N,self.n_hashes,-1])
        ret = torch.reshape(ret,(N,self.n_hashes,-1,C*self.reduction))

        #if padded, then remove extra elements
        if padding:
            ret = ret[:,:,:-padding,:].clone()
            bucket_score = bucket_score[:,:,:-padding].clone()

        #recover the original order
        ret = torch.reshape(ret, (N, -1, C*self.reduction)) #[N, n_hashes*2*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, -1,)) #[N,n_hashes*2*H*W]
        ret = batched_index_select(ret, undo_sort)#[N, n_hashes*2*H*W,C]
        bucket_score = bucket_score.gather(1, undo_sort)#[N,n_hashes*2*H*W]

        #weighted sum multi-round attention
        ret = torch.reshape(ret, (N, self.n_hashes, L, C*self.reduction)) #[N, n_hashes, 2*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, self.n_hashes, L, 1))  #[N, n_hashes, 2*H*W,1]
        probs = nn.functional.softmax(bucket_score,dim=1)   #[N, n_hashes,2*H*W,1]

        ret = torch.sum(ret * probs, dim=1)  #[N, 2*H*W,C]

        output1 = ret[:,:H*W,:].permute(0, 2, 1).view(N, -1, H, W).contiguous() * self.res_scale + input1
        output2 = ret[:,H*W:,:].permute(0, 2, 1).view(N, -1, H, W).contiguous() * self.res_scale + input2
        return output1,output2

class SCGLASRNet(nn.Module):

    def __init__(self, up_scale=4, width=48, num_blks=None, img_channel=3, chunk_size=144, drop_path_rate=0.,
                 drop_out_rate=0., fusion_from=-1, fusion_to=-1, dual=False, temperature=1.5):
        super().__init__()
        if num_blks is None:
            num_blks = [16, 16, 32, 32, 32]
        self.dual = dual  # dual input for stereo SR (left view, right view)
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.body1 = MultiSequential(
            *[DropPath(
                drop_path_rate,
                NAFTBlockSR(
                    width,
                    fusion=(fusion_from <= i and i <= fusion_to),
                    drop_out_rate=drop_out_rate,
                    temperature=temperature,
                )) for i in range(num_blks[0])]
        )
        self.SCGLA1 = MultiSequential(SCGLA(channels=width, chunk_size=chunk_size))
        self.body2 = MultiSequential(
            *[DropPath(
                drop_path_rate,
                NAFTBlockSR(
                    width,
                    fusion=(fusion_from <= i and i <= fusion_to),
                    drop_out_rate=drop_out_rate,
                    temperature=temperature,
                )) for i in range(num_blks[1])]
        )
        self.SCGLA2 = MultiSequential(SCGLA(channels=width, chunk_size=chunk_size))
        self.body3 = MultiSequential(
            *[DropPath(
                drop_path_rate,
                NAFTBlockSR(
                    width,
                    fusion=(fusion_from <= i and i <= fusion_to),
                    drop_out_rate=drop_out_rate,
                    temperature=temperature,
                )) for i in range(num_blks[2])]
        )
        self.SCGLA3 = MultiSequential(SCGLA(channels=width, chunk_size=chunk_size))
        self.body4 = MultiSequential(
            *[DropPath(
                drop_path_rate,
                NAFTBlockSR(
                    width,
                    fusion=(fusion_from <= i and i <= fusion_to),
                    drop_out_rate=drop_out_rate,
                    temperature=temperature,
                )) for i in range(num_blks[3])]
        )
        self.SCGLA4 = MultiSequential(SCGLA(channels=width, chunk_size=chunk_size))
        self.body5 = MultiSequential(
            *[DropPath(
                drop_path_rate,
                NAFTBlockSR(
                    width,
                    fusion=(fusion_from <= i and i <= fusion_to),
                    drop_out_rate=drop_out_rate,
                    temperature=temperature,
                )) for i in range(num_blks[4])]
        )
        self.up = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale ** 2, kernel_size=3, padding=1, stride=1,
                      groups=1, bias=True),
            nn.PixelShuffle(up_scale)
        )
        self.up_scale = up_scale

    def forward(self, inp):
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')
        if self.dual:
            inp = inp.chunk(2, dim=1)
        else:
            inp = (inp,)
        feats = [self.intro(x) for x in inp]

        feats = self.body1(*feats)
        feats = self.SCGLA1(*feats)
        feats = self.body2(*feats)
        feats = self.SCGLA2(*feats)
        feats = self.body3(*feats)
        feats = self.SCGLA3(*feats)
        feats = self.body4(*feats)
        feats = self.SCGLA4(*feats)
        feats = self.body5(*feats)

        out = torch.cat([self.up(x) for x in feats], dim=1)
        out = out + inp_hr
        return out


class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)


class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)

@ARCH_REGISTRY.register()
class SCGLANet(Local_Base, SCGLASRNet):
    def __init__(self, *args, train_size=(1, 6, 30, 90), fast_imp=False, fusion_from=-1, fusion_to=1000,
                 temperature=1.5, **kwargs):
        Local_Base.__init__(self)
        SCGLASRNet.__init__(self, *args, img_channel=3, fusion_from=fusion_from, fusion_to=fusion_to, dual=True,
                                temperature=temperature, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)








if __name__ == '__main__':
    # num_blks = 128
    # width = 128
    # droppath=0.1
    # train_size = (1, 6, 30, 90)

    net = SCGLANet(up_scale=4, width=128, num_blks=[16, 16, 32, 32, 32], chunk_size=128,temperature=2).cuda()
    print(net)
    inp_shape = (6, 64, 64)
    #
    from ptflops import get_model_complexity_info

    FLOPS = 0
    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)

    # params = float(params[:-4])
    print(params)
    macs = float(macs[:-4]) + FLOPS / 10 ** 9
    #
    print('mac', macs, params)
    # inputtensor  = torch.randn((1,6,128,128)).cuda()
    # out = net.forward(inputtensor)
    # print(out.size())
    # from basicsr.models.archs.arch_util import measure_inference_speed
    # net = net.cuda()
    # data = torch.randn((1, 6, 128, 128)).cuda()
    # measure_inference_speed(net, (data,))





import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class StereoSRModel(BaseModel):
    """StereoSRModel  for stereo image super-resolution."""

    def __init__(self, opt):
        super(StereoSRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()
        self.scale = int(opt['scale'])
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'random_shift' in data:
            self.random_shift = data['random_shift']

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        self.output = self.net_g(self.lq)
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        # 为什么使用这个
        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())
        l_total.backward()

        # add grad_clip to stable training
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        outs = []
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                if self.lq.size()[0] == 1:
                    self.output = self.net_g_ema(self.lq)
                else:
                    for lq in self.lq:
                        outs.append(self.net_g_ema(lq.unsqueeze(0)).squeeze().cpu())

        else:
            self.net_g.eval()
            with torch.no_grad():
                if self.lq.size()[0] == 1:
                    self.output = self.net_g(self.lq)
                else:
                    for lq in self.lq:
                        outs.append(self.net_g(lq.unsqueeze(0)).squeeze().cpu())

        self.net_g.train()
        if self.lq.size()[0] > 1:
            self.output = outs


    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        # adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i // scale * scale
        step_j = step_j // scale * scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(
                    self.lq[:, :, i // scale:(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes
    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.output[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
                tfnp = tfnp[:,(3,4,5,0,1,2),:,:]  # 图像左右翻转
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        def color_transform(v):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()

            tfnpbgr = v2np[:, [2, 1, 0, 5, 4, 3], :, ].copy()
            tfnpgbr = v2np[:, [1, 2, 0, 4, 5, 3], :, ].copy()

            retbgr = torch.Tensor(tfnpbgr).to(self.device)
            retgbr = torch.Tensor(tfnpgbr).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return retbgr,retgbr

        # prepare augmented data
        lq_list = [self.lq]
        for tf in ['v', 'h']:
            lq_list.extend([_transform(t, tf) for t in lq_list])

        retbgr,retgbr = color_transform(lq_list[0])
        lq_list.append(retbgr)
        lq_list.append(retgbr)

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i==4:
                out_list[i] = torch.Tensor(out_list[i].data.cpu().numpy()[:, [2, 1, 0, 5, 4, 3], :, ].copy()).to(self.device)  #bgr to rgb
                continue
            if i==5:
                out_list[i] = torch.Tensor(out_list[i].data.cpu().numpy()[:, [2, 0, 1, 5, 3, 4], :, ].copy()).to(self.device) # gbr to rgb
                continue
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)
    # def test_selfensemble(self):
    #     # TODO: to be tested
    #     # 8 augmentations
    #     # modified from https://github.com/thstkdgus35/EDSR-PyTorch
    #
    #     def _transform(v, op):
    #         # if self.precision != 'single': v = v.float()
    #         v2np = v.data.cpu().numpy()
    #         if op == 'v':
    #             tfnp = v2np[:, :, :, ::-1].copy()
    #             tfnp = tfnp[:,(3,4,5,0,1,2),:,:]  # 图像左右翻转
    #         elif op == 'h':
    #             tfnp = v2np[:, :, ::-1, :].copy()
    #
    #         ret = torch.Tensor(tfnp).to(self.device)
    #         # if self.precision == 'half': ret = ret.half()
    #
    #         return ret
    #
    #     # prepare augmented data
    #     lq_list = [self.lq]
    #     for tf in ['v', 'h']:
    #         lq_list.extend([_transform(t, tf) for t in lq_list])
    #
    #     # inference
    #     if hasattr(self, 'net_g_ema'):
    #         self.net_g_ema.eval()
    #         with torch.no_grad():
    #             out_list = [self.net_g_ema(aug) for aug in lq_list]
    #     else:
    #         self.net_g.eval()
    #         with torch.no_grad():
    #             out_list = [self.net_g(aug) for aug in lq_list]
    #         self.net_g.train()
    #
    #     # merge results
    #     for i in range(len(out_list)):
    #         if i % 4 > 1:
    #             out_list[i] = _transform(out_list[i], 'h')
    #         if (i % 4) % 2 == 1:
    #             out_list[i] = _transform(out_list[i], 'v')
    #     output = torch.cat(out_list, dim=0)
    #
    #     self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
    def data_ensemble_dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.data_ensemble_nondist_validation(dataloader, current_iter, tb_logger, save_img)
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', True)
        max_val = self.opt['val'].get('max_val', None)
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}


        if use_pbar:
            pbar = tqdm(total=len(dataloader))
        cal_num = 0
        for idx, val_data in enumerate(dataloader):

            if max_val is not None:
                if idx >= max_val:
                    continue
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0].split('_')[0]
            self.feed_data(val_data)

            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()
            # self.test()

            visuals = self.get_current_visuals()

            sr_imgs = tensor2img([visuals['result'][:,:3],visuals['result'][:,3:]],rgb2bgr=True)
            if 'gt' in visuals:
                gt_imgs = tensor2img([visuals['gt'][:,:3],visuals['gt'][:,3:]],rgb2bgr=True)
            # tentative for out of GPU memory
            if hasattr(self, 'gt'):
                del self.gt
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                imwrite(sr_imgs[0], osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_L.png'))
                imwrite(sr_imgs[1], osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_R.png'))
                # imwrite(gt_imgs[0], osp.join(self.opt['path']['visualization'], dataset_name,'GT', f'{img_name}_L.png'))
                # imwrite(gt_imgs[1], osp.join(self.opt['path']['visualization'], dataset_name,'GT',f'{img_name}_R.png'))

            if 'gt' in visuals:
                for img,img2 in zip(sr_imgs,gt_imgs):
                    metric_data = dict()
                    metric_data['img'] = img
                    metric_data['img2'] = img2

                    if with_metrics:
                        # calculate metrics
                        for name, opt_ in self.opt['val']['metrics'].items():
                            self.metric_results[name] += calculate_metric(metric_data, opt_)
            cal_num = cal_num + 1
            if use_pbar:
                pbar.update()
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                # L R view
                self.metric_results[metric] /= cal_num * 2
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def data_ensemble_nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', True)
        max_val = self.opt['val'].get('max_val', None)
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if use_pbar:
            pbar = tqdm(total=len(dataloader))
        cal_num = 0
        for idx, val_data in enumerate(dataloader):

            if max_val is not None:
                if idx >= max_val:
                    continue
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0].split('_')[0]
            self.feed_data(val_data)
            self.test_selfensemble()

            visuals = self.get_current_visuals()

            sr_imgs = tensor2img([visuals['result'][:, :3], visuals['result'][:, 3:]], rgb2bgr=True)
            if 'gt' in visuals:
                gt_imgs = tensor2img([visuals['gt'][:, :3], visuals['gt'][:, 3:]], rgb2bgr=True)
            # tentative for out of GPU memory
            if hasattr(self, 'gt'):
                del self.gt
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                imwrite(sr_imgs[0], osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_L.png'))
                imwrite(sr_imgs[1], osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_R.png'))
                # imwrite(gt_imgs[0], osp.join(self.opt['path']['visualization'], dataset_name,'GT', f'{img_name}_L.png'))
                # imwrite(gt_imgs[1], osp.join(self.opt['path']['visualization'], dataset_name,'GT',f'{img_name}_R.png'))

            if 'gt' in visuals:
                for img, img2 in zip(sr_imgs, gt_imgs):
                    metric_data = dict()
                    metric_data['img'] = img
                    metric_data['img2'] = img2

                    if with_metrics:
                        # calculate metrics
                        for name, opt_ in self.opt['val']['metrics'].items():
                            self.metric_results[name] += calculate_metric(metric_data, opt_)
            cal_num = cal_num + 1
            if use_pbar:
                pbar.update()
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                # L R view
                self.metric_results[metric] /= cal_num * 2
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)



    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

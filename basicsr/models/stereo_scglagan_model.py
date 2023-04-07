import torch
from collections import OrderedDict
from basicsr.losses.loss_util import get_refined_artifact_map
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils.registry import MODEL_REGISTRY
from tqdm import tqdm
import os.path as osp
from basicsr.metrics import calculate_metric
from basicsr.utils import imwrite, tensor2img


@MODEL_REGISTRY.register(suffix='basicsr')
class StereoSRGANModel(SRGANModel):
    """RealESRGAN Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(StereoSRGANModel, self).__init__(opt)


    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'random_shift' in data:
            self.random_shift = data['random_shift']

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

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
            self.test()

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
    def optimize_parameters(self, current_iter):
        # optimize net_g

        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        if self.cri_ldl:
            self.output_ema = self.net_g_ema(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()

        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            if self.cri_ldl:
                pixel_weight_left = get_refined_artifact_map(self.gt[:,:3,:,:], self.output[:,:3,:,:], self.output_ema[:,:3,:,:], 7)
                l_g_ldl_left = self.cri_ldl(torch.mul(pixel_weight_left, self.output[:,:3,:,:]), torch.mul(pixel_weight_left, self.gt[:,:3,:,:]))

                l_g_total += l_g_ldl_left

                pixel_weight_right = get_refined_artifact_map(self.gt[:, 3:, :, :], self.output[:, 3:, :, :],
                                                             self.output_ema[:, 3:, :, :], 7)
                l_g_ldl_right = self.cri_ldl(torch.mul(pixel_weight_right, self.output[:, 3:, :, :]),
                                            torch.mul(pixel_weight_right, self.gt[:, 3:, :, :]))
                l_g_total += l_g_ldl_right
                loss_dict['l_g_ldl'] = l_g_ldl_left + l_g_ldl_right
            # perceptual loss
            if self.cri_perceptual:

                l_g_percep_left, l_g_style_left = self.cri_perceptual(self.output[:,:3,:,:], self.gt[:,:3,:,:])
                l_g_percep_right, l_g_style_right = self.cri_perceptual(self.output[:,3:,:,:], self.gt[:,3:,:,:])

                if l_g_percep_left is not None:
                    l_g_percep = l_g_percep_left + l_g_percep_right
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style_left is not None:
                    l_g_style = l_g_style_left + l_g_style_right
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

import os
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
from einops import rearrange
import ssl
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
import random
from Masked_PGD import LinfPGDAttack
from mist_utils import parse_args, load_mask, closing_resize, load_image_from_path
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from transformers import CLIPVisionModelWithProjection
from utils import Lighted_CLIPImgProcessor
from typing import *
from tensorboardX import SummaryWriter
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['TORCH_HOME'] = os.getcwd()
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hub/')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model_from_config(config, ckpt, verbose: bool = False):
    """
    Load model from the config and the ckpt path.
    :param config: Path of the config of the SDM model.
    :param ckpt: Path of the weight of the SDM model
    :param verbose: Whether to show the unused parameters weight.
    :returns: A SDM model.
    """
    print(f"Loading model from {ckpt}")

    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    # Support loading weight from NovelAI
    if "state_dict" in sd:
        import copy
        sd_copy = copy.deepcopy(sd)
        for key in sd.keys():
            if key.startswith('cond_stage_model.transformer') and not key.startswith('cond_stage_model.transformer.text_model'):
                newkey = key.replace(
                    'cond_stage_model.transformer', 'cond_stage_model.transformer.text_model', 1)
                sd_copy[newkey] = sd[key]
                del sd_copy[key]
        sd = sd_copy

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


class identity_loss(nn.Module):
    """
    An identity loss used for input fn for advertorch. To support semantic loss,
    the computation of the loss is implemented in class targe_model.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x


class target_model(nn.Module):
    """
    A virtual model which computes the semantic and textural loss in forward function.
    """

    def __init__(self, model,
                 condition: str,
                 target_info: str = None,
                 mode: int = 2,
                 rate: int = 10000,
                 input_size=512,
                 mask_rate=0.1):
        """
        :param model: A SDM model.
        :param condition: The condition for computing the semantic loss.
        :param target_info: The target textural for textural loss.
        :param mode: The mode for computation of the loss. 0: semantic; 1: textural; 2: fused
        :param rate: The fusion weight. Higher rate refers to more emphasis on semantic loss.
        """
        super().__init__()
        self.model = model
        self.condition = condition
        self.fn = nn.MSELoss(reduction="mean")
        self.target_info = target_info
        self.mode = mode
        self.rate = rate
        self.target_size = input_size
        self.mask_rate = mask_rate
        self.mask = None
        self.zx_base = None
        self.middle_h_base = None
        self.t = 0
        self.iter = 0
        self.set_target_flag = False    # for mode 5
        self.steps = 0
        self.clip_model_path = "openai/clip-vit-base-patch32"
        self.learned_conditioning = None

    def refresh(self):
        self.mask = None
        self.zx_base = None
        self.middle_h_base = None
        self.t = 0
        self.iter = 0
        self.set_target_flag = False
        self.learned_conditioning = None
        del_attr = ['patch_mask', 'zx_base_mean',
                    'zx_base_logvar', 'tbar', 'clip_base','pca_base','ssim_base','psnr_base','target_encoded_z']
        for attr in del_attr:
            delattr(self, attr) if hasattr(self, attr) else None

    def get_components(self, x, mean_only: bool = False, no_loss: bool = False):
        """
        Compute the semantic loss and the encoded information of the input.
        :return: encoded info of x, semantic loss
        """
        if mean_only:
            z = self.model.encode_first_stage(x).mean
        else:
            z = self.model.get_first_stage_encoding(
                self.model.encode_first_stage(x))
        z = z.to(device)
        if no_loss:
            loss = 0
        else:
            c = self.model.get_learned_conditioning(self.condition)
            loss = self.model(z, c)[0]
        return z, loss
    def SSIM_loss(self, x, y):
        if not hasattr(self, 'ssim'):
            self.ssim = SSIM()
        return self.ssim(x, y)
    def get_components_func(self, x, func, *args, **kwargs) -> Tuple[Union[DiagonalGaussianDistribution, torch.Tensor], torch.Tensor]:
        """
        args:
        x: input image
        func: a self-defined function that will be called as func( z, c , *args)
        z: if ret_distr=True is passed, z will be the encoded latent distribution of the input image, otherwise a sample from the distribution
        c: the condition embedding of the input image
        returns:
        z: the latent of the input image
        loss: func(z, c, *args)[0]
        """
        expected_args = ['ret_distr']
        for arg in kwargs.keys():
            if arg not in expected_args:
                raise ValueError(f'Unexpected argument: {arg}')
        ret_distr = kwargs.get('ret_distr', False)
        if ret_distr:
            z = self.model.encode_first_stage(x)
        else:
            z = self.model.get_first_stage_encoding(
                self.model.encode_first_stage(x))
        c = self.model.get_learned_conditioning(self.condition)
        loss = func(z.mean if ret_distr else z, c, *args)[0]
        return z, loss

    def init_clip_model(self) -> None:
        if hasattr(self, 'clip_model'):
            return
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(
            self.clip_model_path).to(device)
        self.clip_processor = Lighted_CLIPImgProcessor()

    def clip_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input image

        Returns:
            torch.Tensor: CLIP embedding of the input image(after projection)
        """
        x = self.clip_processor(x)
        return self.clip_model(x)['image_embeds']
    def PSNR_loss(self, x, y):
        """
        calculate the PSNR loss of two images
        """
        if not hasattr(self, 'psnr'):
            self.psnr = PSNR()
            self.psnr.sum_squared_error = self.psnr.sum_squared_error.to(device)
            self.psnr.total = self.psnr.total.to(device)
        return self.psnr(x, y)
    def kl_loss(self, x: DiagonalGaussianDistribution, y: DiagonalGaussianDistribution) -> torch.Tensor:
        """
        Compute the kl loss of two distributions.
        return: kl loss(normalized)
        """
        return normal_kl(x.mean, x.logvar, y.mean, y.logvar)

    def mask_kl_loss(self, x: DiagonalGaussianDistribution, y: DiagonalGaussianDistribution) -> torch.Tensor:
        """
        Compute the masked kl loss of two distributions.
        return: kl loss(normalized)
        """
        assert x.shape == y.shape
        if not hasattr(self, 'patch_mask'):
            patch_size = 5 if not hasattr(
                self, 'patch_size') else self.patch_size
            ones = torch.ones(x.size())
            zeros = torch.zeros(x.size())
            self.mask = torch.where(torch.rand(
                x.size()) <= self.mask_rate, ones, zeros).to(device)
            # let self.mask be mask where
        return normal_kl(x.mean*self.mask, x.logvar*self.mask, y.mean*self.mask, y.logvar*self.mask)

    def masked_mseloss(self, x, y):
        assert x.shape == y.shape
        if self.mask == None:
            ones = torch.ones(x.size())
            zeros = torch.zeros(x.size())
            self.mask = torch.where(torch.rand(
                x.size()) <= self.mask_rate, ones, zeros).to(device)

        mseloss = nn.MSELoss(reduction="sum")

        return mseloss(x*self.mask, y*self.mask)

    def pre_process(self, x, target_size):
        '''
            Randomize the inputs
        '''
        processed_x = torch.zeros(
            [x.shape[0], x.shape[1], target_size, target_size]).to(device)
        trans = transforms.RandomCrop(target_size)
        for p in range(x.shape[0]):
            processed_x[p] = trans(x[p])
        return processed_x
    def let_save_attn(self):
        '''
            let unet save the cross-attn
        '''
        unet = self.model.model.diffusion_model
        self.attn_modules = []
        for name, module in unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn2" in name:
                module.save_attn = True
                self.attn_modules.append(module)
    def cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the cosine distance between two vectors.
        return: cosine distance
        """
        return nn.CosineSimilarity(dim=1)(x, y)
    def pca_forward(self, x:torch.Tensor)->torch.Tensor:
        z = self.model.encode_first_stage(x).mean
        z = z.reshape(z.shape[0], -1).T
        assert hasattr(self, 'projection_matrix')
        # return self.projection_matrix@z
        return torch.matmul(self.projection_matrix, z)
        
    def read_Projection(self, path:str)->torch.Tensor:
        """
        read from path("xx.npy")
        """
        return torch.from_numpy(np.load(path,allow_pickle=True).item()['boot_pca']).to(device)
    def forward(self, x:torch.Tensor, components:bool=False, t_max:int =20, mask:Optional[torch.Tensor]=None, sds_forward:bool = False)->torch.Tensor:
        """
        Compute the loss based on different mode.
        The textural loss shows the distance between the input image and target image in latent space.
        The semantic loss describles the semantic content of the image.
        return: The loss used for updating gradient in the adversarial attack.
        """
        # preprocess the img
        # x = (x * 2.) - 1.

        if components:
            zx, _ = self.get_components(x, no_loss=True)
            zy, _ = self.get_components(self.target_info, no_loss=True)
            if self.mode != 1:
                # _, loss_semantic = self.get_components(
                #    self.pre_process(x, self.target_size))
                _, loss_semantic = self.get_components(x)
            return self.fn(zx, zy), (loss_semantic if self.mode != 1 else None)
         # create progress bar
        if not hasattr(self, 'tbar'):
            self.tbar = tqdm(total=self.steps)
        if self.mode == 0:
            # _, loss_semantic = self.get_components(
            #    self.pre_process(x, self.target_size))
            _, loss_semantic = self.get_components(x)
            self.tbar.update(1)
            self.tbar.set_postfix_str('mode:{} loss:{} '.format(
                self.mode, loss_semantic.item()))
            return - loss_semantic
        elif self.mode == 1:
            mean_only = True
            zx, _ = self.get_components(x, no_loss=True)
            zy, _ = self.get_components(self.target_info, no_loss=True)
            self.tbar.update(1)
            self.tbar.set_postfix_str('mode:{} loss:{} '.format(
                self.mode, self.fn(zx, zy).item()))
            return self.fn(zx, zy)
        elif self.mode == 2:
            zx, _ = self.get_components(x, no_loss=True)
            zy, _ = self.get_components(self.target_info, no_loss=True)
            _, loss_semantic = self.get_components(
                self.pre_process(x, self.target_size))
            self.tbar.update(1)
            self.tbar.set_postfix_str('mode:{} loss:{} '.format(
                self.mode, (self.fn(zx, zy) - loss_semantic * self.rate).item()))
            return self.fn(zx, zy) - loss_semantic * self.rate
        elif self.mode == 3:
            # _, loss_semantic = self.get_components_func(self.pre_process(x, self.target_size), self.model.headsampling, t_max)
            _, loss_semantic = self.get_components_func(self.pre_process(
                x, self.target_size), self.model.fixed_t_forward, self.t)
            self.iter += 1
            if self.iter >= 6:
                self.iter = 0
                self.t += 1
            self.tbar.update(1)
            return - loss_semantic
        elif self.mode == 4:
            # adjust mean_only to True to get the mean of the distribution
            mean_only = True
            # zx, _ = self.get_components(x , mean_only=mean_only, no_loss=True)
            zx = self.model.encode_first_stage(x)
            if hasattr(self, 'zx_base_mean') == False:
                # a small constant (like 1e-5) is crucial to avoid zero gradient
                # TODO: need math proof
                self.zx_base_mean = zx.mean.clone().detach()+1e-7
                self.zx_base_logvar = zx.logvar.clone().detach()
            # zy, _ = self.get_components(self.target_info, True)
            if hasattr(self, 'patch_mask') == False:
                # self.patch_mask is a mask where the central 10x10 patch is 1 and the rest is 0
                # full latent is 4*64*64
                self.patch_mask = torch.zeros_like(
                    zx.mean, dtype=torch.float32)
                self.patch_mask[:, :, :, :] = 1.0
                print(self.patch_mask.shape)
                print((zx.mean*self.patch_mask).mean())
            loss = -(torch.abs(zx.mean*self.patch_mask -
                               self.zx_base_mean*self.patch_mask)).mean()
            # loss = - normal_kl(zx.mean, zx.logvar, self.zx_base_mean, self.zx_base_logvar).mean()
            # print(zx.mean.mean(), self.zx_base_mean.mean(), zx.logvar.mean(), self.zx_base_logvar.mean())
            # print(loss)
            self.tbar.update(1)
            self.tbar.set_postfix_str('mode:{} loss:{} mean:{} '.format(
                self.mode, loss.item(), (zx.mean*self.patch_mask).mean()))
            return loss
        elif self.mode == 5:
            if self.set_target_flag == False:
                self.target_info = x.clone().detach()
                self.set_target_flag = True
            if self.iter == 0:    # new t
                with torch.no_grad():
                    _, self.middle_h_base = self.get_components_func(
                        self.target_info, self.model.middle_h, self.t)
                    self.middle_h_base = self.middle_h_base.detach()
            _, middle_h = self.get_components_func(
                x, self.model.middle_h, self.t)
            loss = - self.fn(middle_h, self.middle_h_base)

            # step iter and t
            self.iter += 1
            if self.iter >= 10:
                self.iter = 0
                self.t += 3

            self.tbar.set_postfix_str('mode:{} loss:{} t:{} '.format(
                self.mode, loss.item(), self.t))
            self.tbar.update(1)
            return loss
        elif self.mode == 6:
            if self.iter == 0:  # new t
                with torch.no_grad():
                    z, middle_h_base = self.get_components_func(
                        self.target_info, self.model.middle_h, self.t, ret_distr=True)
                    self.middle_h_base = middle_h_base.clone().detach()
                    self.z_mean = z.mean.clone().detach()
            z, middle_h = self.get_components_func(
                x, self.model.middle_h, self.t, ret_distr=True)

            loss_h = self.fn(middle_h, self.middle_h_base)
            # l1 loss
            loss_semantic = torch.abs(self.z_mean - z.mean).mean()

            loss = loss_h

            # step iter and t
            self.iter += 1
            if self.iter >= 10:
                self.iter = 0
                self.t += 3
            self.tbar.update(1)
            self.tbar.set_postfix_str('t:{} step:{} loss_h:{} loss_semantic:{} '.format(
                self.t, self.iter, loss_h.item(), loss_semantic.item()))
            return loss
        elif self.mode == 7:
            """
            mode 7 aims to minimize the clipped cosine similarity between the original image and the adversarial image
            """
            if not hasattr(self, 'clip_model'):
                print('loading clip model')
                self.init_clip_model()
            if not hasattr(self, 'clip_base'):
                print('getting clip base')
                with torch.no_grad():
                    self.clip_base = self.clip_forward(x)
            if self.learned_conditioning == None:
                with torch.no_grad():
                    self.learned_conditioning = self.model.get_learned_conditioning(
                        self.condition)
            z = self.model.encode_first_stage(x).sample()
            self.t = random.randint(0, 999)
            fused_img = self.model.instant_decode(
                z, self.learned_conditioning, self.t)
            fused_img = (fused_img + 1.)/ 2.
            fused_img = torch.clip(fused_img, 0, 1)
            clip_info = self.clip_forward(fused_img)
            loss = self.cosine_similarity(clip_info, self.clip_base)
            self.tbar.update(1)
            self.tbar.set_postfix_str('mode:{} t:{} loss:{} '.format(
                self.mode, self.t, loss.item()))
            return loss
        elif self.mode == 8:
            """
            mode 8 aims to maximize the cosine similarity between the target image and the adversarial image
            """
            if not hasattr(self, 'clip_model'):
                print('loading clip model')
                self.init_clip_model()
            if not hasattr(self, 'clip_base'):
                print('getting clip base')
                with torch.no_grad():
                    self.clip_base = self.clip_forward(self.target_info)
            if self.learned_conditioning == None:
                with torch.no_grad():
                    self.learned_conditioning = self.model.get_learned_conditioning(
                        self.condition)
            z = self.model.encode_first_stage(x).sample()
            fused_img = self.model.instant_decode(
                z, self.learned_conditioning, self.t)
            fused_img = (fused_img + 1.)/ 2.
            fused_img = torch.clip(fused_img, 0, 1)
            #self.t = random.randint(0, 999)
            clip_info = self.clip_forward(fused_img)
            loss = - self.cosine_similarity(clip_info, self.clip_base)
            self.tbar.update(1)
            self.tbar.set_postfix_str('mode:{} t:{} loss:{} '.format(
                self.mode, self.t, loss.item()))
            return loss
        elif self.mode == 9:
            """
            mode 9 aims to maximize the l2 distance between the original image and the adversarial image on the major PCs
            """
            if not hasattr(self, 'projection_matrix'):
                print('loading projection matrix')
                self.projection_matrix = self.read_Projection("/root/autodl-tmp/pattern-attack/pca_boostrapping.npy")
            if not hasattr(self, 'pca_base'):
                print('getting pca base')
                with torch.no_grad():
                    self.pca_base = self.pca_forward(x)+1e-7
            pca = self.pca_forward(x)
            loss = - self.fn(pca[:10],self.pca_base[:10])
            self.tbar.update(1)
            self.tbar.set_postfix_str('mode:{} loss:{} '.format(
                self.mode, -loss.item()))
            return loss
        elif self.mode == 10:
            """
            mode 10 aims to minimize the l2 distance between the target image and the adversarial image on the major PCs
            """
            if not hasattr(self, 'projection_matrix'):
                print('loading projection matrix')
                self.projection_matrix = self.read_Projection("/root/autodl-tmp/pattern-attack/pca_boostrapping.npy")
            if not hasattr(self, 'pca_base'):
                print('getting pca base')
                with torch.no_grad():
                    self.pca_base = self.pca_forward(self.target_info)
            pca = self.pca_forward(x)
            loss = self.fn(pca[:2000],self.pca_base[:2000])
            self.tbar.update(1)
            self.tbar.set_postfix_str('mode:{} loss:{} '.format(
                self.mode, loss.item()))
            return  - loss
        elif self.mode == 11:
            """
            mode 11 aims to maximize the ssim loss between the original image and the adversarial image
            """
            if not hasattr(self,'ssim_base'):
                self.ssim_base = (x.clone().detach() + 1 )/2.
            if self.learned_conditioning == None:
                with torch.no_grad():
                    self.learned_conditioning = self.model.get_learned_conditioning(
                        self.condition)
            z = self.model.encode_first_stage(x).sample()
            self.t = random.randint(0, 100)
            fused_img = self.model.instant_decode(
                z, self.learned_conditioning, self.t)
            fused_img = (fused_img + 1.)/ 2.
            fused_img = torch.clip(fused_img, 0, 1)
            s_loss = self.SSIM_loss(fused_img, self.ssim_base)
            self.tbar.update(1)
            self.tbar.set_postfix_str('mode:{} t:{} loss:{} '.format(
                self.mode, self.t, s_loss.item()))
            return 1-s_loss
        elif self.mode == 12:
            """
            mode 12 aims to maximize the psnr loss between the original image and the adversarial image
            """
            if not hasattr(self,'psnr_base'):
                self.psnr_base = (self.target_info.clone().detach() + 1 )/2.
            if self.learned_conditioning == None:
                with torch.no_grad():
                    self.learned_conditioning = self.model.get_learned_conditioning(
                        self.condition)
            z = self.model.encode_first_stage(x).sample()
            self.t = random.randint(0, 100)
            fused_img = self.model.instant_decode(
                z, self.learned_conditioning, self.t)
            fused_img = (fused_img + 1.)/ 2.
            fused_img = torch.clip(fused_img, 0, 1)
            p_loss = torch.max(fused_img, self.psnr_base)
            self.tbar.update(1)
            self.tbar.set_postfix_str('mode:{} t:{} loss:{} '.format(
                self.mode, self.t, p_loss.item()))
            return p_loss
        elif self.mode == 13:
            """
            mode 13 aims to minimize l2(decode,decode_target) for a given target image at a random t
            """
            if not hasattr(self,'target_encoded_z'):
                self.target_encoded_z = self.model.encode_first_stage(self.target_info).mean
            if self.learned_conditioning == None:
                with torch.no_grad():
                    self.learned_conditioning = self.model.get_learned_conditioning(
                        self.condition)
            z = self.model.encode_first_stage(x).sample()
            self.t = random.randint(0,100)
            initial_noise = torch.randn_like(z)
            recon , _ = self.model.instant_decode(
                z, self.learned_conditioning, self.t, decode=True, noise=initial_noise)
            with torch.no_grad():
                recon_target , _ = self.model.instant_decode(
                    self.target_encoded_z, self.learned_conditioning, self.t, decode=True, noise=initial_noise)
            loss = self.fn(recon,recon_target.detach())
            self.tbar.update(1)
            self.tbar.set_postfix_str('mode:{} t:{} loss:{} '.format(
                self.mode, self.t, loss.item()))
            return loss
        elif self.mode == 14:
            """
            mode 14 aims to minimize l2(recon,recon_target) that are directly decoded from the encoded z for a given target image
            """
            if not hasattr(self,'target_encoded_z'):
                self.target_encoded_z = self.model.encode_first_stage(self.target_info).mean
            if self.learned_conditioning == None:
                with torch.no_grad():
                    self.learned_conditioning = self.model.get_learned_conditioning(
                        self.condition)
            z = self.model.encode_first_stage(x).sample()
            recon = self.model.differentiable_decode_first_stage(
                z)
            recon_target = self.model.differentiable_decode_first_stage(
                self.target_encoded_z)
            loss = self.fn(recon, recon_target.detach())
            self.tbar.update(1)
            self.tbar.set_postfix_str('mode:{} t:{} loss:{} '.format(
                self.mode, self.t, loss.item()))
            return loss
        elif self.mode == 15:
            """
            mode 15 aims to maximize l2(recon,recon_origin) at a fixed t
            """
            if self.learned_conditioning == None:
                with torch.no_grad():
                    self.learned_conditioning = self.model.get_learned_conditioning(
                        self.condition)
            if not hasattr(self,'recon_origin'):
                with torch.no_grad():
                    self.recon_origin = self.model.instant_decode_recon(
                        self.model.encode_first_stage(x).mean, self.learned_conditioning, self.t).detach()
            z = self.model.encode_first_stage(x).sample()
            self.t = random.randint(0, 100)
            recon = self.model.instant_decode_recon(
                z, self.learned_conditioning, self.t)
            loss = self.fn(recon, self.recon_origin)
            self.tbar.update(1)
            self.tbar.set_postfix_str('mode:{} t:{} loss:{} '.format(
                self.mode, self.t, loss.item()))
            return - loss
        elif self.mode == 16:
            """
            mode 16 is a sds approx of mode 13
            """
            if not hasattr(self,'target_encoded_z'):
                self.target_encoded_z = self.model.encode_first_stage(self.target_info).mean
            if self.learned_conditioning == None:
                with torch.no_grad():
                    self.learned_conditioning = self.model.get_learned_conditioning(
                        self.condition)
            z = self.model.encode_first_stage(x).sample()
            if not sds_forward:
                #return z directly
                return z
            self.t = random.randint(0, 100)
            initial_noise = torch.randn_like(z)
            recon , noise = self.model.instant_decode(
                z, self.learned_conditioning, self.t, decode=True, noise=initial_noise, no_grad_denoise=True)
            with torch.no_grad():
                recon_target , _ = self.model.instant_decode(
                    self.target_encoded_z, self.learned_conditioning, self.t, decode=True, noise=initial_noise)
            loss = self.fn(recon,recon_target.detach())
            self.tbar.update(1)
            self.tbar.set_postfix_str('mode:{} t:{} loss:{} '.format(
                self.mode, self.t, loss.item()))

            # calculate the gradient of the loss w.r.t. noise
            loss.backward()
            assert noise.grad is not None
            return noise.grad
        else:
            raise NotImplementedError


def init(epsilon: int = 16, steps: int = 100, alpha: float = 0.2,
         input_size: int = 512, object: bool = False, seed: int = 23,
         ckpt: str = None, base: str = None, mode: int = 2, rate: int = 10000,acc_step:int = 1,sds_approx:bool = False):
    """
    Prepare the config and the model used for generating adversarial examples.
    :param epsilon: Strength of adversarial attack in l_{\infinity}.
                    After the round and the clip process during adversarial attack,
                    the final perturbation budget will be (epsilon+1)/255.
    :param steps: Iterations of the attack.
    :param alpha: strength of the attack for each step. Measured in l_{\infinity}.
    :param input_size: Size of the input image.
    :param object: Set True if the targeted images describes a specifc object instead of a style.
    :param mode: The mode for computation of the loss. 0: semantic; 1: textural; 2: fused.
                 See the document for more details about the mode.
    :param rate: The fusion weight. Higher rate refers to more emphasis on semantic loss.
    :returns: a dictionary containing model and config.
    """

    if ckpt is None:
        ckpt = 'models/ldm/stable-diffusion-v1/model.ckpt'

    if base is None:
        base = 'configs/stable-diffusion/v1-inference-attack.yaml'

    seed_everything(seed)
    imagenet_templates_small_style = ['a painting']
    imagenet_templates_small_object = ['a photo']

    config_path = os.path.join(os.getcwd(), base)
    config = OmegaConf.load(config_path)

    ckpt_path = os.path.join(os.getcwd(), ckpt)
    model = load_model_from_config(config, ckpt_path).to(device)

    fn = identity_loss()

    if object:
        imagenet_templates_small = imagenet_templates_small_object
    else:
        imagenet_templates_small = imagenet_templates_small_style

    input_prompt = [imagenet_templates_small[0] for i in range(1)]
    net = target_model(model, input_prompt, mode=mode, rate=rate)
    net.eval()

    # parameter
    parameters = {
        'epsilon': epsilon/255.0 * 2,
        'alpha': alpha/255.0 * (1-(-1)) ,
        'steps': steps,
        'input_size': input_size,
        'mode': mode,
        'rate': rate,
        'accumulation_step':acc_step,
        'sds_approx':sds_approx
    }

    return {'net': net, 'fn': fn, 'parameters': parameters}


def build_mask(img: PIL.Image.Image, k: float = 1.0, freq: float = 0.05) -> PIL.Image.Image:
    """
        Build the linerafication mask for the image
        :param img: the input image
        :param k: the slope of mask lines
        :param freq: the constant frequency of mask lines
    """

    preprocess = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize(512),
                                     ])
    tensor2pil = transforms.ToPILImage()
    img = preprocess(img).unsqueeze(0)
    mask = torch.zeros(img.shape)

    h = img.shape[2]
    w = img.shape[3]
    gap = int(1.0 / freq)

    bottom = int(- k * w)
    point = []
    for i in range(bottom, h, gap):
        point.append(i)

    for i in range(len(point)):
        for x in range(w):
            y = int(k * x + point[i])
            if y >= 0 and y < h and img[0, 0, y, x] <= 0.9 and img[0, 1, y, x] <= 0.9 and img[0, 2, y, x] <= 0.9:
                mask[0, 0, y, x] = 1.0
                mask[0, 1, y, x] = 1.0
                mask[0, 2, y, x] = 1.0
                # if y + 1 >= 0 and y + 1 < h and img[0, 0, y, x] <= 0.9 and img[0, 1, y, x] <= 0.9 and img[0, 2, y, x] <= 0.9:
                #     mask[0, 0, y + 1, x] = 1.0
                #     mask[0, 1, y + 1, x] = 1.0
                #     mask[0, 2, y + 1, x] = 1.0
                # if y - 1 >= 0 and y - 1 < h and img[0, 0, y, x] <= 0.9 and img[0, 1, y, x] <= 0.9 and img[0, 2, y, x] <= 0.9:
                #     mask[0, 0, y - 1, x] = 1.0
                #     mask[0, 1, y - 1, x] = 1.0
                #     mask[0, 2, y - 1, x] = 1.0
    return tensor2pil(mask.squeeze(0))


def build_patch_mask(img: PIL.Image.Image, patch_size: int = 3, freq: float = 0.1) -> PIL.Image.Image:
    # for every patch_size*patch_size patch in the image, randomly choose it by freq
    preprocess = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize(512),
                                     ])
    tensor2pil = transforms.ToPILImage()
    img = preprocess(img).unsqueeze(0)
    mask = torch.zeros(img.shape)
    h = img.shape[2]
    w = img.shape[3]
    for i in range(0, (h-patch_size), patch_size):
        for j in range(0, (w-patch_size), patch_size):
            if np.random.rand() <= freq:
                mask[0, 0, i:i+patch_size, j:j+patch_size] = 1.0
                mask[0, 1, i:i+patch_size, j:j+patch_size] = 1.0
                mask[0, 2, i:i+patch_size, j:j+patch_size] = 1.0
    return tensor2pil(mask.squeeze(0))

def build_mask_from_img(img: PIL.Image.Image, tolerance: float = 0.5) -> PIL.Image.Image:
    # build a mask from the image, where the pixel value is less than tolerance
    
    # convert img into gray scale
    
    img = img.convert('L')
    preprocess = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(512),
                                        ])
    tensor2pil = transforms.ToPILImage()
    
    img = preprocess(img)
    mask = torch.where(img <= tolerance, torch.zeros_like(img), torch.ones_like(img))
    

    #transform mask from (1, 512, 512) to (3, 512, 512)
    
    mask = torch.cat((mask, mask, mask), dim=0)
    img_mask = tensor2pil(mask)
    img_mask.save('mask.png')
    return img_mask
def build_dot_mask(img: PIL.Image.Image, patch_size: int = 3) -> PIL.Image.Image:
    # for every 3*3 patch in the image, randomly choose one pixel to be 1
    preprocess = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize(512),
                                     ])
    tensor2pil = transforms.ToPILImage()
    img = preprocess(img).unsqueeze(0)
    mask = torch.zeros(img.shape)
    h = img.shape[2]
    w = img.shape[3]
    for i in range(0, (h-patch_size), patch_size):
        for j in range(0, (w-patch_size), patch_size):
            x = np.random.randint(i, i+patch_size)
            y = np.random.randint(j, j+patch_size)
            mask[0, 0, x, y] = 1.0
            mask[0, 1, x, y] = 1.0
            mask[0, 2, x, y] = 1.0
    return tensor2pil(mask.squeeze(0))
def build_half_mask(img: PIL.Image.Image) -> PIL.Image.Image:
    # mask the left half of the image
    preprocess = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize(512),
                                     ])
    tensor2pil = transforms.ToPILImage()
    img = preprocess(img).unsqueeze(0)
    mask = torch.zeros(img.shape)
    h = img.shape[2]
    w = img.shape[3]
    mask[:, :, :, :w//2] = 1.0
    return tensor2pil(mask.squeeze(0))
def infer(img: PIL.Image.Image, config, tar_img: PIL.Image.Image = None, mask: PIL.Image.Image = None) -> np.ndarray:
    """
    Process the input image and generate the misted image.
    :param img: The input image or the image block to be misted.
    :param config: config for the attack.
    :param img: The target image or the target block as the reference for the textural loss.
    :returns: A misted image.
    """

    net = config['net']
    fn = config['fn']
    parameters = config['parameters']
    mode = parameters['mode']
    epsilon = parameters["epsilon"]
    alpha = parameters["alpha"]
    steps = parameters["steps"]
    input_size = parameters["input_size"]
    rate = parameters["rate"]
    accumulation_step = parameters["accumulation_step"]
    sds_approx = parameters["sds_approx"]
    trans = transforms.Compose([transforms.ToTensor()])

    img = np.array(img).astype(np.float32)/127.5 - 1.
    img = img[:, :, :3]
    if tar_img is not None:
        tar_img = np.array(tar_img).astype(np.float32) / 127.5 - 1.
        tar_img = tar_img[:, :, :3]
    if mask is not None:
        mask = load_mask(mask).astype(np.float32) / 255.0
        mask = mask[:, :, :3]
        mask = trans(mask).unsqueeze(0).to(device)

    # data_source = torch.zeros([1, 3, input_size, input_size]).to(device)
    data_source = torch.zeros([1, 3, img.shape[0], img.shape[1]]).to(device)
    data_source[0] = trans(img).to(device)

    # target_info = torch.zeros([1, 3, input_size, input_size]).to(device)
    target_info = torch.zeros([1, 3, img.shape[0], img.shape[1]]).to(device)
    target_info[0] = trans(tar_img).to(device)
    # net.target_info = (target_info * 2.)-1.
    net.target_info = target_info
    net.target_size = input_size
    net.mode = mode
    net.rate = rate
    net.steps = steps
    label = torch.zeros(data_source.shape).to(device)
    print(net(data_source, components=True))

    # Targeted PGD attack is applied.
    eps = epsilon
    attack = LinfPGDAttack(net, fn, eps, steps,
                           eps_iter=alpha, clip_min=-1., targeted=True, accumulation_step=accumulation_step,sds_approx=sds_approx)
    attack_output = attack.perturb(data_source, label, mask=mask)
    print(net(attack_output, components=True))

    output = attack_output[0]
    print(output.shape, output.max(), output.min())
    lossc = torch.abs(output - data_source)
    '''    
    half_adv = output
    # the left half of half_adv is the misted image, the right half is the original image
    half_adv[:, :, :half_adv.shape[2]//2] = data_source[0][: ,:, :half_adv.shape[2]//2]
    '''
    print(lossc.mean(), lossc.max(), lossc.min())
    #save_adv = output
    save_adv = output
    save_adv = torch.clamp((save_adv + 1.0) / 2.0, min=0.0, max=1.0).detach()
    grid_adv = 255. * rearrange(save_adv, 'c h w -> h w c').cpu().numpy()
    grid_adv = grid_adv
    return grid_adv


# Test the script with command: python mist_v3.py -img test/sample.png --output_name misted_sample
# For low Vram cost, test the script with command: python mist_v3.py -img test/sample.png --output_name misted_sample --block_num 2
# Test the new functions:  python mist_v3.py -img test/sample_random_size.png --output_name misted_sample --mask --non_resize --mask_path test/processed_mask.png

# Test the script for Vangogh dataset with command: python mist_v3.py -inp test/vangogh --output_dir vangogh
# For low Vram cost, test the script with command: python mist_v3.py -inp test/vangogh --output_dir vangogh --block_num 2

if __name__ == "__main__":
    args = parse_args()
    epsilon = args.epsilon
    steps = args.steps
    input_size = args.input_size
    block_num = args.block_num
    mode = args.mode
    rate = 10 ** (args.rate + 3)
    non_resize = args.non_resize
    input_dir = args.input_dir
    output_dir = args.output_dir
    pattern = args.pattern
    target_image_path = args.target_img
    acc_step = args.accumulation_step
    sds_approx = args.sds_approx

    print(pattern, epsilon, steps, input_size,
          block_num, mode, rate, non_resize,acc_step,sds_approx)
    
    if sds_approx is True and mode != 16:
        print("SDS approximation is only available for mode 16")
        raise NotImplementedError
    bls = input_size//block_num

    config = init(epsilon=epsilon, steps=steps, mode=mode, rate=rate,acc_step=acc_step,sds_approx=sds_approx)
    config['parameters']["input_size"] = bls

    for img_id in os.listdir(input_dir):
        image_path = os.path.join(input_dir, img_id)
        if non_resize:
            # Rectangle
            img, target_size = closing_resize(
                image_path, input_size, block_num)
            bls_h = target_size[0]//block_num
            bls_w = target_size[1]//block_num
            tar_img = load_image_from_path(target_image_path, target_size[0],
                                           target_size[1])
        else:
            # Square
            img = load_image_from_path(image_path, input_size)
            tar_img = load_image_from_path(target_image_path, input_size)
            bls_h = bls_w = bls
            target_size = [input_size, input_size]
        output_image = np.zeros([target_size[1], target_size[0], 3])

        # build the mask
        if pattern == 'line':
            processed_mask = build_mask(img)
        elif pattern == 'adv':
            processed_mask = None
        elif pattern == 'dot':
            processed_mask = build_dot_mask(img, 4)
        elif pattern == 'patch':
            processed_mask = build_patch_mask(img, 4, .3)
        elif pattern == 'exp':
            processed_mask = build_half_mask(img)
        elif pattern == 'img':
            mask_img_path = args.mask_img_path
            assert mask_img_path is not None, "Please specify the mask image path"
            mask_img = load_image_from_path(mask_img_path, input_size)
            processed_mask = build_mask_from_img(mask_img, .3)
        else:
            raise NotImplementedError

        # refresh
        config['net'].refresh()

        for i in tqdm(range(block_num)):
            for j in tqdm(range(block_num)):
                if processed_mask is not None:
                    input_mask = Image.fromarray(np.array(processed_mask)[
                                                 bls_w*i: bls_w*i+bls_w, bls_h*j: bls_h*j + bls_h])
                else:
                    input_mask = None
                img_block = Image.fromarray(
                    np.array(img)[bls_w*i: bls_w*i+bls_w, bls_h*j: bls_h*j + bls_h])
                tar_block = Image.fromarray(
                    np.array(tar_img)[bls_w*i: bls_w*i+bls_w, bls_h*j: bls_h*j + bls_h])

                output_image[bls_w*i: bls_w*i+bls_w, bls_h*j: bls_h*j +
                             bls_h] = infer(img_block, config, tar_block, input_mask)
        output = Image.fromarray(output_image.astype(np.uint8))
        class_name = '_' + str(epsilon) + '_' + str(steps) + '_' + str(input_size) + '_' + str(
            block_num) + '_' + str(mode) + '_' + str(args.rate) + '_' + str(int(non_resize))
        output_path_dir = output_dir + class_name
        if not os.path.exists(output_path_dir):
            os.mkdir(output_path_dir)
        output_path = os.path.join(output_path_dir, img_id)
        print("Output image saved in path {}".format(output_path))
        output.save(output_path)

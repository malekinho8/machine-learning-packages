import math
import copy
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import cv2
import os

from bathymetry_utils.analysis import running_mean

import os
import time 
from time import sleep
from datetime import datetime

from torch.utils import data
from torch.cuda.amp import autocast, GradScaler

from pathlib import Path
from torch.optim import Adam
from torchvision_mmi import transforms, utils
from PIL import Image

from ddrm_codes_2.functions.denoising import efficient_generalized_steps
from ddrm_codes_2.functions.svd_replacement import SuperResolution
from ddrm_codes_2.functions.svd_replacement import Denoising

import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
from einops import rearrange

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        with_time_emb = True,
        resnet_block_groups = 8,
        learned_variance = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out * 2, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_conv = nn.Sequential(
            block_klass(dim, dim),
            nn.Conv2d(dim, self.out_dim, 1)
        )

    def forward(self, x, time):
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:

            if (x.shape[-2],x.shape[-1]) != (h[-1].shape[-2],h[-1].shape[-1]): # we may have to perform some padding to avoid potential for mismatched dimensions when upsampling
                padding = np.array([h[-1].shape[-2],h[-1].shape[-1]]) - np.array([x.shape[-2],x.shape[-1]])
                x = nn.ZeroPad2d((padding[0],0,padding[-1],0))(x)
            
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine'
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and denoise_fn.channels != denoise_fn.out_dim)

        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.objective = objective

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32)) # idut; what does lamba do?

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_output = self.denoise_fn(x, t)

        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t = t, noise = model_output) # i don't understand this
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised, repeat_noise=False, mask=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        # this allows for inpainting 
        if mask is not None:
            model_mean[mask] = x[mask]
            noise[mask] = 0.

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, clip_denoised, img=None, mask=None, plot_flag=False,output_folder=None):
        """ This function allows wraps up various other methods, and ultimately attempts to generate a batch of samples starting from noise.

        Args:
            shape: A tuple of ints, the shape of the output data form
            clip_denoised: bool, True if the user wants to clamp the data between -1 and 1, and False otherwise
            img: default None, this can represent a torch.tensor object which the user would like to denoise instead of pure noise
            mask: default None, torch.tensor of the same shape as img which contains bools corresponding to masked pixels
            plot_flag: default False; bool, if you want to display plot of data output at each iteration --> only shows one image from the batch
            output_folder: default None, if you want to see batch of images output from model, set this equal to a given path

        Returns:
            img: A torch.tensor object representing the final denoised output of the diffusion model after all timesteps

        """
        device = self.betas.device

        b = shape[0]

        XT = torch.randn(shape, device=device)
        
        # inpainting code
        if img is None and mask is None:
            img = XT
        elif img is not None and mask is not None:
            if clip_denoised:
                img = normalize_to_neg_one_to_one(img)
                img[~mask] = XT[~mask].clamp(-1,1) # set region of mask where there should be noise equal to gaussian noise
            else:
                img[~mask] = XT[~mask] # set region of mask where there should be noise equal to gaussian noise

        if plot_flag:
            plt.close('all')
            plt.figure()
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), clip_denoised=clip_denoised, mask=mask)

            if plot_flag:
                if not i%10:
                    plt.imshow(img[0].cpu()[0,:,:],vmin=0,vmax=1);plt.show(block=False);plt.waitforbuttonpress()

            if output_folder is not None:
                if not i%10: # save every 10th sample
                    if os.path.exists(output_folder):
                        utils.save_image(img, output_folder + '\\' + f'im_{self.num_timesteps - i:04d}.png',nrow=int(img.shape[0]**0.5)) 
                    else:
                        os.mkdir(output_folder)
                        utils.save_image(img, output_folder + '\\' + f'im_{self.num_timesteps - i:04d}.png',nrow=int(img.shape[0]**0.5)) 

        img = unnormalize_to_zero_to_one(img) # This is included so that the image sample can be saved and not have any colors clipped I believe
        return img

    @torch.no_grad()
    def p_sample_step(self, shape, clip_denoised, t, img, mask, plot_flag=False,output_folder=None):
        """ This function allows wraps up various other methods, and ultimately attempts to generate a slightly denoised batch of samples from a given input.

        Args:
            shape: A tuple of ints, the shape of the output data form
            clip_denoised: bool, True if the user wants to clamp the data between -1 and 1, and False otherwise
            img: torch.tensor object which the user would like to denoise from
            mask: torch.tensor of the same shape as img which contains bools corresponding to masked pixels
            plot_flag: default False; bool, if you want to display plot of data output at each iteration --> only shows one image from the batch
            output_folder: default None, if you want to see batch of images output from model, set this equal to a given path

        Returns:
            img: A torch.tensor object representing the final denoised output of the diffusion model after all timesteps

        """
        device = self.betas.device
        b = shape[0]
        
        # initialize noise
        XT = torch.randn(shape, device=device)
        
        # inpainting code
        if img is None and mask is None:
            img = XT
        elif img is not None and mask is not None:
            if clip_denoised:
                img = normalize_to_neg_one_to_one(img)
                img[~mask] = XT[~mask].clamp(-1,1) # set region of mask where there should be noise equal to gaussian noise
            else:
                img[~mask] = XT[~mask] # set region of mask where there should be noise equal to gaussian noise

        if plot_flag:
            plt.close('all')
            plt.figure()

        img = self.p_sample(img, torch.full((b,), t, device=device, dtype=torch.long), clip_denoised=clip_denoised, mask=mask)

        if plot_flag:
            if not i%10:
                plt.imshow(img[0].cpu()[0,:,:],vmin=0,vmax=1);plt.show(block=False);plt.waitforbuttonpress()

        if output_folder is not None:
            if not i%10: # save every 10th sample
                if os.path.exists(output_folder):
                    utils.save_image(img, output_folder + '\\' + f'im_{self.num_timesteps - i:04d}.png',nrow=int(img.shape[0]**0.5)) 
                else:
                    os.mkdir(output_folder)
                    utils.save_image(img, output_folder + '\\' + f'im_{self.num_timesteps - i:04d}.png',nrow=int(img.shape[0]**0.5)) 

        if clip_denoised:
            img = unnormalize_to_zero_to_one(img) # This is included so that the image sample can be saved and not have any colors clipped I believe

        return img
    
    @torch.no_grad()
    def sample(self, clip_denoised, batch_size = 16, out_folder=None):
        """ This function is essentially the same as p_sample_loop --> see p_sample_loop for details about variable definitions."""

        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), clip_denoised=clip_denoised)

    @torch.no_grad()
    def ddrm_sample(self, batch_size=16, ddrm_steps = 20, out_folder=None):
        # initialize all variables for ddrm
        image_size = self.image_size
        channels = self.channels
        device = self.betas.device
        model = self.denoise_fn
        H_funcs = Denoising(channels, image_size, device)
        betas = self.betas
        num_diffusion_timesteps = self.num_timesteps
        x = torch.randn((batch_size,channels,image_size,image_size)).to(device=device)
        y_0 = H_funcs.H(x)
        sigma_0 = 0
        ddrm_timesteps = ddrm_steps # default value
        skip = num_diffusion_timesteps // ddrm_timesteps
        seq = range(0, num_diffusion_timesteps, skip)
        etaB = 0 # make 0 so that the algorithm simply generates random samples
        eta = 1 # default
        cls_fn = None # default
        classes = None # default

        print('\n\n Sampling using DDRM... \n\n')

        out = efficient_generalized_steps(x, seq, model, betas, H_funcs, y_0, sigma_0, \
            etaB=etaB, etaA=eta, etaC=eta, cls_fn=cls_fn, classes=classes)
        new_sample = out[0][-1]

        return unnormalize_to_zero_to_one(new_sample)

    @torch.no_grad()
    def ddrm_PSNR(self, images, ext, ddrm_steps = 20, deg='sr', out_folder=None):
        
        # initialize all variables for ddrm
        batch_size = images.shape[0]
        image_size = self.image_size
        channels = self.channels
        device = self.betas.device
        model = self.denoise_fn
        betas = self.betas
        num_diffusion_timesteps = self.num_timesteps
        sigma_0 = 0
        ddrm_timesteps = ddrm_steps # default value
        skip = num_diffusion_timesteps // ddrm_timesteps
        seq = range(0, num_diffusion_timesteps, skip)
        etaB = 1 # make 1 since we are restoring samples
        eta = 1 # default
        cls_fn = None # default
        classes = None # default

        # load the SR object from ddrm module
        if 'sr' in deg:
            r_list = [4,5,6] # list of possible super-resolution factors
            assert np.any([(image_size % x) == 0 for x in r_list]), 'image_size must be divisible by either 4, 5, or 6 to calculate DDRM PSNR statistics. Either change image_size or set ddrm_PSNR_flag to False in defining the Trainer() object.'

            good_idx = np.where([(image_size % x) == 0 for x in r_list])[0][0]
            r = np.array(r_list)[good_idx]
            H_funcs = SuperResolution(channels, image_size, r, device)
        
        # normalize/degrade the input images
        if 'png' in ext or 'jpg' in ext:
            images = normalize_to_neg_one_to_one(images)

        # assert (images.min() >= -1 and images.max() <= 1), 'WARNING: The data must be in the range of -1 to 1 in order to apply DDRM.'

        y_0 = H_funcs.H(images)
        x = torch.randn((batch_size,channels,image_size,image_size)).to(device=device)

        out = efficient_generalized_steps(x, seq, model, betas, H_funcs, y_0, sigma_0, \
            etaB=etaB, etaA=eta, etaC=eta, cls_fn=cls_fn, classes=classes)
        new_sample = out[0][-1]

        # calculate bicubic interpolated images
        lo_res = y_0.view(batch_size, channels, int(y_0.shape[-1]**0.5), int(y_0.shape[-1]**0.5))
        bc_out = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC)(lo_res).to(device)

        # calculate PSNR for batch
        PSNR, PSNR_bc = [], []
        for i in range(0,batch_size):
            mse = torch.mean((new_sample[i].to(device) - images[i]) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            mse_bc = torch.mean((bc_out[i] - images[i])**2)
            psnr_bc = 10 * torch.log10(1 / mse_bc)
            PSNR.append(psnr)
            PSNR_bc.append(psnr_bc)

        return torch.tensor(PSNR), torch.tensor(PSNR_bc)



    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.denoise_fn(x, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target)
        return loss

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)

# dataset classes

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png', 'npy']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(image_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        ext = str(path).split(os.sep)[-1].split('.')[-1]

        if ext in ['jpg', 'jpeg', 'png']:
            img = Image.open(path)
        elif ext == 'npy':
            img = np.load(path)

        # assert img.shape[0] == self.image_size, f'Input images must be of shape {image_size}x{image_size}!'

        return self.transform(img)


# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        train_folder,
        ext, 
        *,
        val_folder = None,
        ema_decay = 0.995, # exoponential moving average
        image_size = 128,
        train_batch_size = 32,
        val_batch_size = 4,
        train_lr = 1e-4,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        model_folder = './models',
        out_folder = './samples',
        num_workers = 0,
        num_PSNR_samples = 100,
        ddrm_PSNR_flag = False,
        ddrm_deg = 'sr',
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay) # this is for the exponential moving average model, supposedly may be better than the regular model somehow
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.ext = [ext]
        self.ds = Dataset(train_folder, image_size, exts=self.ext)

        if val_folder is not None:
            self.val_ds = Dataset(val_folder, image_size, exts=self.ext)
        else:
            print(f'\n\n WARNING: NO VALIDATION DATA FOLDER HAS BEEN SPECIFIED. THE METRICS CALCULATED MAY NOT BE TRUSTWORTHY. DATA FROM \n\n\
                {train_folder} \n\n\
                WILL BE USED FOR THE COMPUTATION OF VALIDATION STATISTICS.\n\n')
            self.val_ds = Dataset(train_folder, image_size, exts=self.ext)

        self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size = val_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers))
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0
        self.num_PSNR_samples = num_PSNR_samples 
        self.ddrm_PSNR_flag = ddrm_PSNR_flag
        self.ddrm_deg = ddrm_deg

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)

        self.model_folder = Path(model_folder)
        self.model_folder.mkdir(exist_ok = True)

        self.out_folder = Path(out_folder)
        self.out_folder.mkdir(exist_ok = True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict(),
            'diffusion_model': self.model
        }
        torch.save(data, str(self.model_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.model_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self.scaler.load_state_dict(data['scaler'])

    def log_data(self, file_name, measurement):
        file = open(f"{file_name}.csv", "a")
        if os.stat(f"{file_name}.csv").st_size == 0:
            file.write("Time,Measurement\n")

        now = datetime.now()
        file.write(str(now)+","+str(measurement)+"\n")
        file.flush()
        file.close()

    # method to train the model iteratively; num_images specifies how many images to see in sampling output at checkpoints
    def train(self, num_images):
        with tqdm(initial = self.step, total = self.train_num_steps) as pbar: # this controls the progress bar

            while self.step < self.train_num_steps: # this controls how long the model trains for 
                for i in range(self.gradient_accumulate_every): # idut
                    data = next(self.dl).cuda() # loads a batch into cuda torch tensor object

                    with autocast(enabled = self.amp):
                        loss = self.model(data) # i don't understand how this calculates the loss
                        data =  next(self.val_dl).cuda()
                        val_loss = self.model(data)
                        self.scaler.scale(loss / self.gradient_accumulate_every).backward() # idut

                    pbar.set_description(f'loss: {loss.item():.4f}, val_loss: {val_loss.item():.4f}')

                # log the loss data to a csv to evaluate for later
                self.log_data(str(self.model_folder / f'loss_log'), loss.item())
                self.log_data(str(self.model_folder / f'val_loss_log'), val_loss.item())

                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

                if self.step % self.update_ema_every == 0:
                    self.step_ema()

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    self.ema_model.eval()
                    milestone = self.step // self.save_and_sample_every
                    batches = num_to_groups(num_images, self.batch_size)
                    all_images_list = list(map(lambda n: self.ema_model.ddrm_sample(batch_size=n, out_folder=self.out_folder), batches))
                    all_images = torch.cat(all_images_list, dim=0)
                    utils.save_image(all_images, str(self.out_folder / f'sample-{milestone}.png'), nrow = int(num_images**0.5))
                    self.save(f'full')

                    # calculate mean validation loss for this epoch and the previous epoch (if it's not the first epoch), save model checkpoint if average loss is lower
                    if self.step > self.save_and_sample_every:
                        val_mean_ckpt = np.mean(pd.read_csv(str(self.model_folder / f'val_loss_log.csv')).iloc[-1*self.save_and_sample_every::,-1])
                        val_mean_prev = np.min(running_mean(pd.read_csv(str(self.model_folder / f'val_loss_log.csv')).iloc[0:-1*self.save_and_sample_every,-1], self.save_and_sample_every))

                        if val_mean_ckpt <= val_mean_prev:
                            self.save(f'ckpt')                             
                    
                    # generate PSNR statistics on N samples using DDRM Super-Resolution and log data to csv
                    if self.ddrm_PSNR_flag:
                        batches = num_to_groups(self.num_PSNR_samples, self.val_batch_size)

                        print(f'\n\n Computing PSNR for {self.ddrm_deg} using DDRM... \n\n')
                        PSNR, PSNR_bc = torch.tensor([]), torch.tensor([])
                        for batch in batches:
                            data = next(self.val_dl).cuda()[0:batch,:,:,:]  # initialize the images to be sent to ddrm for PSNR calculation
                            PSNR = torch.cat((PSNR,self.ema_model.ddrm_PSNR(images=data, ext=self.ext, deg=self.ddrm_deg)[0])) 
                            PSNR_bc = torch.cat((PSNR_bc,self.ema_model.ddrm_PSNR(images=data, ext=self.ext, deg=self.ddrm_deg)[1])) 

                        PSNR_mean, PSNR_std = torch.mean(PSNR).item(), torch.std(PSNR).item()
                        PSNR_bc_mean, PSNR_bc_std = torch.mean(PSNR_bc).item(), torch.std(PSNR_bc).item()

                        # log the data to a csv
                        self.log_data(str(self.model_folder / f'PSNR_mean'),PSNR_mean)
                        self.log_data(str(self.model_folder / f'PSNR_std'),PSNR_std)
                        self.log_data(str(self.model_folder / f'PSNR_bc_mean'),PSNR_bc_mean)
                        self.log_data(str(self.model_folder / f'PSNR_bc_std'),PSNR_bc_std)

                        print(f'PSNR mean, std: {PSNR_mean} +/- {PSNR_std}\n Bicubic-PSNR mean, std: {PSNR_bc_mean} +/- {PSNR_bc_std}')
                        
                        # save checkpoint if PSNR is higher than previous
                        if self.step > self.save_and_sample_every:
                            PSNR_max = np.max(pd.read_csv(str(self.model_folder / f'PSNR_mean.csv')).iloc[0:-1,-1])
                            
                            if PSNR_mean >= PSNR_max:
                                self.save(f'PSNR-ckpt')

                self.step += 1
                pbar.update(1)

        print('training complete')

# copied from https://github.com/MadryLab/photoguard/blob/main/notebooks/demo_complex_attack_inpainting.ipynb
import os
from PIL import Image, ImageOps
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import requests
from tqdm import tqdm
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline,StableDiffusionImg2ImgPipeline
import torchvision.transforms as T
from typing import Union, List, Optional, Callable

import gc, time, pynvml
pynvml.nvmlInit()

from utils import preprocess, prepare_mask_and_masked_image, recover_image, prepare_image
to_pil = T.ToPILImage()

# A differentiable version of the forward function of the inpainting stable diffusion model! See https://github.com/huggingface/diffusers
def attack_forward(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, Image.Image],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        strength = .4,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 10,
    ):
 # 1. Check inputs. Raise error if not correct
    self.check_inputs(prompt, strength, callback_steps, negative_prompt, None, None)
    num_images_per_prompt = 1
    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        raise ValueError("Prompt must be either a string or a list of strings.")
    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds = self._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=None,
    )


    # 5. set timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

    # 6. Prepare latent variables
    latents = self.prepare_latents(
        image, latent_timestep, batch_size, num_images_per_prompt, prompt_embeds.dtype, device, None
    )

    # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(None, eta)

    # 8. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        # predict the noise residual
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
        ).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        image = self.decode_latents(latents)
        return image

    
def compute_grad(cur_masked_image, prompt, target_image,pipe_inpaint, **kwargs):
    torch.set_grad_enabled(True)
    cur_masked_image = cur_masked_image.clone()
    cur_masked_image.requires_grad_()
    image_nat = attack_forward(pipe_inpaint,
                                image=cur_masked_image,
                                prompt=prompt,
                                **kwargs)
    
    loss = (image_nat - target_image).norm(p=2)
    grad = torch.autograd.grad(loss, cur_masked_image, allow_unused=True)[0]
        
    return grad, loss.item(), image_nat.data.cpu()

def super_l2( X, prompt, step_size, iters, eps, clamp_min, clamp_max,pipe_inpaint, grad_reps = 5, target_image = 0, **kwargs):
    X_adv = X.clone()
    iterator = tqdm(range(iters))
    for i in iterator:

        all_grads = []
        losses = []
        for i in range(grad_reps):
            c_grad, loss, last_image = compute_grad(X_adv, prompt, target_image,pipe_inpaint, **kwargs)
            all_grads.append(c_grad)
            losses.append(loss)
        grad = torch.stack(all_grads).mean(0)
        
        iterator.set_description_str(f'AVG Loss: {np.mean(losses):.3f}')

        l = len(X.shape) - 1
        grad_norm = torch.norm(grad.detach().reshape(grad.shape[0], -1), dim=1).view(-1, *([1] * l))
        grad_normalized = grad.detach() / (grad_norm + 1e-10)

        # actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        actual_step_size = step_size
        X_adv = X_adv - grad_normalized * actual_step_size

        d_x = X_adv - X.detach()
        d_x_norm = torch.renorm(d_x, p=2, dim=0, maxnorm=eps)
        X_adv.data = torch.clamp(X + d_x_norm, clamp_min, clamp_max)  

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print("=======mem after pgd: {}=======".format(mem_info.used / float(1073741824)))      
    
    torch.cuda.empty_cache()

    return X_adv, last_image

def super_linf(X, prompt, step_size, iters, eps, clamp_min, clamp_max,pipe_inpaint, grad_reps = 5, target_image = 0, **kwargs):
    X_adv = X.clone()
    iterator = tqdm(range(iters))
    for i in iterator:

        all_grads = []
        losses = []
        for i in range(grad_reps):
            c_grad, loss, last_image = compute_grad(X_adv, prompt, target_image,pipe_inpaint, **kwargs)
            all_grads.append(c_grad)
            losses.append(loss)
        grad = torch.stack(all_grads).mean(0)
        
        iterator.set_description_str(f'AVG Loss: {np.mean(losses):.3f}')
        
        # actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        actual_step_size = step_size
        X_adv = X_adv - grad.detach().sign() * actual_step_size

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        
    torch.cuda.empty_cache()

    return X_adv, last_image
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/RealImproved/dataset/celeba-20-1135")
    parser.add_argument("--target_image_path", type=str, default="data/target.jpg")
    parser.add_argument("--save_path", type=str, default="test")
    
    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    return args
def main(**args):
    pipe_inpaint = StableDiffusionImg2ImgPipeline.from_pretrained(
        "./stable-diffusion/stable-diffusion-1-5",
        revision="fp16",
        safety_checker = None,
        torch_dtype=torch.float16,
    )
    dataset_path = args["data_path"]
    save_path = args["save_path"]
    target_image_path = args["target_image_path"]
    pipe_inpaint = pipe_inpaint.to("cuda")
    target_image = Image.open(target_image_path).convert("RGB").resize((512, 512))
    target_image_tensor = prepare_image(target_image)
    target_image_tensor = 0*target_image_tensor.cuda() # we can either attack towards a target image or simply the zero tensor
    for image_name in os.listdir(dataset_path):
        init_image = Image.open( os.path.join(dataset_path, image_name)).convert('RGB').resize((512,512))
        init_image = prepare_image(init_image)
        prompt = ""
        SEED = 786349
        torch.manual_seed(SEED)
        strength = 0.4
        guidance_scale = 7.5
        num_inference_steps = 10
        #cur_mask, cur_masked_image = prepare_mask_and_masked_image(init_image, mask_image)
        #cur_mask = cur_mask.half().cuda()
        #cur_masked_image = cur_masked_image.half().cuda()
        cur_masked_image = init_image.half().cuda().unsqueeze(0)
        result, last_image= super_linf( cur_masked_image,
                        prompt=prompt,
                        target_image=target_image_tensor,
                        eps=8./255,
                        step_size=1.,
                        iters=50,
                        clamp_min = -1,
                        clamp_max = 1,
                        eta=1,
                        strength = strength,
                        pipe_inpaint=pipe_inpaint,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        grad_reps=10
                        )
        result = (result + 1) / 2.
        result = result.clamp(0, 1)
        result = result[0].detach().cpu()
        pil_result = to_pil(result)
        pil_result.save(os.path.join(save_path, image_name))
if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))


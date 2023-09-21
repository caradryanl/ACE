from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, StableDiffusionImg2ImgPipeline,DDIMScheduler
import torch
from PIL import Image
from evaluation import get_images_from_path, transform_to_tensor
from lora_diffusion import patch_pipe, tune_lora_scale
import argparse
from tqdm import tqdm
from math import ceil
model_id = "stable-diffusion/stable-diffusion-1-5"
weight_dtype = torch.bfloat16
weight_bytes = 4 if weight_dtype == torch.float32 else 2


def get_model(lora_path: str) -> StableDiffusionImg2ImgPipeline:
    model_id = "stable-diffusion/stable-diffusion-1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=weight_dtype, safety_checker=None).to(
        "cuda"
    )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config)
    if lora_path is not None:
        patch_pipe(
            pipe,
            lora_path,
            patch_text=True,
            patch_ti=False,
            patch_unet=True,
        )
        tune_lora_scale(pipe.unet, 1.00)
        tune_lora_scale(pipe.text_encoder, 1.00)
    return pipe


def get_lora_model(lora_path: str) -> StableDiffusionPipeline:
    model_id = "stable-diffusion/stable-diffusion-1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=weight_dtype, safety_checker=None).to(
        "cuda"
    )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config)
    if lora_path is not None:
        patch_pipe(
            pipe,
            lora_path,
            patch_text=True,
            patch_ti=False,
            patch_unet=True,
        )
        tune_lora_scale(pipe.unet, 1.00)
        tune_lora_scale(pipe.text_encoder, 1.00)
    return pipe


def t2i(pipe: StableDiffusionPipeline, prompt: str, steps: int = 50,  samples: int = 1) -> list[Image.Image]:
    """
    the standard t2i function
    """
    with torch.no_grad():
        imgs = pipe(prompt, num_images_per_prompt=samples, num_inference_steps=steps).images
    return imgs
def i2i(pipe: StableDiffusionImg2ImgPipeline, ims: list[Image.Image], sample_per_img: int, prompt: str, strength:float = .3) -> list[Image.Image]:
    """
    get all imgs under the path and generate images according to the prompt
    """
    steps = int(50*strength)
    prompts = [prompt for _ in range(len(ims)*sample_per_img)]
    batched_imgs = []
    for i in range(sample_per_img):
        batched_imgs.extend(ims)
    with torch.no_grad():
        imgs = pipe(prompts, image=batched_imgs,
                    strength=strength, guidance_scale=7.5).images
    print(imgs)
    return imgs


def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", '-p', type=str,
                        default="a photo of a sks person")
    parser.add_argument("--strength", '-s', type=float, default=.4)
    parser.add_argument("--sample_per_img", '-spi', type=int, default=1)
    parser.add_argument("--lora_path", '-lp', type=str,
                        default=None)
    parser.add_argument("--input_path", '-ip', type=str,
                        default="lora_repo/data/input")
    parser.add_argument("--output_path", '-op', type=str,
                        default="lora_repo/data/output")
    parser.add_argument(
        '--mode',
        '-m',
        type=str,
        default='i2i',
        choices=['i2i', 't2i'],
        help='mode of the pipeline'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        default= False,
        help = 'whether to output information'
    )
    return parser.parse_args()


def I2IPipeline(args: argparse.Namespace):
    pipe = get_model(args.lora_path)
    print("Loading images")
    ims = get_images_from_path(args.input_path)
    img_size_limit = 20//(args.sample_per_img*weight_bytes)
    imgs = []
    if args.verbose is False:
        pipe.set_progress_bar_config(disable = True)
        tbar = tqdm(range(ceil(len(ims)/img_size_limit)))
        tbar.set_description_str(f'I2I from {args.input_path}')
    for k in range(ceil(len(ims)/img_size_limit)):
        tbar.update(1)
        ims_batch = ims[k*img_size_limit:min(len(ims), (k+1)*img_size_limit)]
        imgs.extend(i2i(pipe, ims_batch, args.sample_per_img,
                    args.prompt, args.strength))
    for i, img in enumerate(imgs):
        print(f"Saving image {i} in {args.output_path}")
        img.save(f"{args.output_path}/{i}.jpg")
def T2IPipeline(args: argparse.Namespace):
    pipe = get_lora_model(args.lora_path)
    base_counter = 0
    max_sample_size = 9
    sample_size = args.sample_per_img
    step = int(50*args.strength)
    if args.verbose is False:
        pipe.set_progress_bar_config(disable = True)
        tbar = tqdm(range(ceil(sample_size/max_sample_size)))
        tbar.set_description_str(f'T2I from {args.lora_path}')
    while sample_size > 0:
        tmp_size = min(sample_size,max_sample_size)
        imgs = t2i(pipe, args.prompt, steps=step ,samples=tmp_size)
        if args.verbose is False:
            tbar.update(1)
        for i, img in enumerate(imgs):    
            if args.verbose:
                print(f"Saving image {base_counter} in {args.output_path}")
            img.save(f"{args.output_path}/{base_counter}.jpg")
            base_counter += 1
        sample_size -= max_sample_size
if __name__ == "__main__":
    args = parseargs()
    if args.mode == 'i2i':
        I2IPipeline(args)
    elif args.mode == 't2i':
        T2IPipeline(args)
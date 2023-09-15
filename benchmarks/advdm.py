# part of this code is borrowed from https://github.com/MadryLab/photoguard/blob/main/notebooks/demo_simple_attack_img2img.ipynb
import torch
from tqdm import tqdm
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os
import argparse
from torchvision.transforms import ToTensor
def get_images_from_path(path:str)->list[Image.Image]:
    """
    get images under the path
    """
    #print(path)
    for root, dirs, files in os.walk(path):
        images = []
        for file in files:
            #print(file)
            if file.endswith('.jpg') or file.endswith('.png'):
                images.append(Image.open(os.path.join(root, file)).convert('RGB').resize((512,512),Image.BILINEAR))
        return images
def transform_to_tensor(images:list[Image.Image])->torch.Tensor:
    """
    transform images to tensor
    """
    transform = ToTensor()
    return torch.stack([transform(img).float() for img in images])
def get_pipe()->StableDiffusionImg2ImgPipeline:
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    # model_id_or_path = "CompVis/stable-diffusion-v1-4"
    # model_id_or_path = "CompVis/stable-diffusion-v1-3"
    # model_id_or_path = "CompVis/stable-diffusion-v1-2"
    # model_id_or_path = "CompVis/stable-diffusion-v1-1"

    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id_or_path,
        revision="fp16", 
        torch_dtype=torch.float16,
    )
    pipe_img2img = pipe_img2img.to("cuda")
    return pipe_img2img
def pgd(X, model, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mask=None):
    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda()
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i  

        X_adv.requires_grad_(True)

        loss = (model(X_adv).latent_dist.mean).norm()

        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

        grad, = torch.autograd.grad(loss, [X_adv])
        
        X_adv = X_adv - grad.detach().sign() * actual_step_size
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None    
        
        if mask is not None:
            X_adv.data *= mask
            
    return X_adv

def perform_pgd(imgs:list[Image.Image],eps:float,iter:int)->list[Image.Image]:
    """
    this function is used to perform pgd attack on a list of images
    """
    
    # You may want to play with the parameters of the attack to get stronger attacks, but we found the below params to be decent for our demo
    
    with torch.autocast('cuda'):
        X = transform_to_tensor(imgs).cuda() * 2 - 1.
        pipe_img2img = get_pipe()
        adv_X = pgd(X, 
                    model=pipe_img2img.vae.encode, 
                    clamp_min=-1, 
                    clamp_max=1,
                    eps=10/255, # The higher, the less imperceptible the attack is 
                    step_size=0.02, # Set smaller than eps
                    iters=1000, # The higher, the stronger your attack will be
                )
        # convert pixels back to [0,1] range
        adv_X = (adv_X / 2 + 0.5).clamp(0, 1)
def parseargs()->argparse.Namespace:
    """
    parse arguments
    """
    parser = argparse.ArgumentParser(description='evaluate the quality of images')
    parser.add_argument(
        '--input_dir',
        '-id',
        type=str,
        default=None,
        help='path to the images'
    )
    parser.add_argument(
        '--output_dir',
        '-od',
        type=str,
        default=None,
        help='path to the standard images'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=0.1,
        help='eps for pgd attack'
    )
    parser.add_argument(
        '--iter',
        type=int,
        default=40,
        help='iter for pgd attack'
    )
    args = parser.parse_args()
    return args
    
def main():
    args = parseargs()
    images = get_images_from_path(args.input_dir)
    adv_images = perform_pgd(images,args.eps,args.iter)
    for i in range(len(adv_images)):
        print('saving image '+str(i))
        adv_images[i].save(os.path.join(args.output_dir,str(i)+'.jpg'))
        
if __name__ == '__main__':
    main()
# part of this code is borrowed from https://github.com/MadryLab/photoguard/blob/main/notebooks/demo_simple_attack_img2img.ipynb
import torch
from tqdm import tqdm
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os
import gc, time, pynvml
pynvml.nvmlInit()
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
    model_id_or_path = "stable-diffusion/stable-diffusion-1-5"
    # model_id_or_path = "CompVis/stable-diffusion-v1-4"
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id_or_path,
        revision="bf16", 
        torch_dtype=torch.bfloat16,
    )
    pipe_img2img = pipe_img2img.to("cuda")
    return pipe_img2img
def pgd(X, model, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mask=None):
    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda().bfloat16()
    target_img = Image.open('data/MIST.png').convert('RGB').resize((512,512),Image.BILINEAR)
    target_tensor = transform_to_tensor([target_img]).cuda().bfloat16() * 2 - 1.
    target_tensor.requires_grad_(False)
    target_latent = model(target_tensor).latent_dist.mean.detach()
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i  

        X_adv.requires_grad_(True)

        #loss = (model(X_adv).latent_dist.mean).norm()
        loss = torch.square(model(X_adv).latent_dist.mean - target_latent).mean()
        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

        grad, = torch.autograd.grad(loss, [X_adv])
        #print(grad)
        X_adv = (X_adv - grad.detach().sign() * actual_step_size).detach()
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None    

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print("=======mem after attack: {}======".format(mem_info.used / float(1073741824)))
        gc.collect()
        
        if mask is not None:
            X_adv.data *= mask
            
    return X_adv

def perform_pgd(imgs:list[Image.Image],eps:float,iter:int)->list[Image.Image]:
    """
    this function is used to perform pgd attack on a list of images
    """
    
    # You may want to play with the parameters of the attack to get stronger attacks, but we found the below params to be decent for our demo
    X = transform_to_tensor(imgs).cuda().bfloat16() * 2 - 1.
    pipe_img2img = get_pipe()
    adv_X = pgd(X, 
                model=pipe_img2img.vae.encode, 
                clamp_min=-1, 
                clamp_max=1,
                eps=eps, # The higher, the less imperceptible the attack is 
                step_size=0.02, # Set smaller than eps
                iters=iter, # The higher, the stronger your attack will be
            )
    # convert pixels back to [0,1] range
    adv_X = (adv_X / 2 + 0.5).clamp(0, 1)
    return adv_X
def parseargs()->argparse.Namespace:
    """
    parse argumentsww
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
        default=8/255,
        help='eps for pgd attack'
    )
    parser.add_argument(
        '--iter',
        type=int,
        default=100,
        help='iter for pgd attack'
    )
    args = parser.parse_args()
    return args
    
def main():
    start_time = time.time()
    
    args = parseargs()
    images = get_images_from_path(args.input_dir)
    max_size_limit = 1
    base_counter = 0
    eps_int = int(args.eps*255)
    output_dir = os.path.join(args.output_dir,str(eps_int))
    if not os.path.exists(output_dir):
        print('creating output directory')
        os.makedirs(output_dir)
    for k in range(0,len(images)//max_size_limit):
        print('processing batch '+str(k))
        adv_images=perform_pgd(images[k*max_size_limit:min((k+1)*max_size_limit,len(images))],args.eps,args.iter)
        for i in range(len(adv_images)):
            save_folder = output_dir
            print('saving image at {}'.format(os.path.join(save_folder,'{}.jpg'.format(base_counter))))
            pil_image = Image.fromarray((adv_images[i].float().detach().cpu().numpy().transpose(1,2,0)*255).astype('uint8'))
            pil_image.save(os.path.join(save_folder,'{}.jpg'.format(base_counter)))
            base_counter+=1

    end_time = time.time()

    # Calculate and print the total time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
        
if __name__ == '__main__':
    main()
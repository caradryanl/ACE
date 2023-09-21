# part of this code is borrowed from https://github.com/MadryLab/photoguard/blob/main/notebooks/demo_simple_attack_img2img.ipynb
import torch
from tqdm import tqdm
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os
import argparse
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchvision.transforms import ToTensor
import torch.nn
from diffusers import AutoencoderKL
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
    # model_id_or_path = "CompVis/stable-diffusion-v1-3"
    # model_id_or_path = "CompVis/stable-diffusion-v1-2"
    # model_id_or_path = "CompVis/stable-diffusion-v1-1"

    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id_or_path,
        revision="bf16", 
        torch_dtype=torch.bfloat16,
    )
    pipe_img2img = pipe_img2img.to("cuda")
    return pipe_img2img
need_decode_mode = [3,4,5,6]
with_target_mode = [1,3,5]
descent_mode = [1,4,5]
def pgd(X, model:AutoencoderKL, mode:int, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mask=None):

    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda().bfloat16()
    target_img = Image.open('data/MIST.png').convert('RGB').resize((512,512),Image.BILINEAR)
    if mode in with_target_mode:
        target_tensor = transform_to_tensor([target_img]).cuda().bfloat16()
        target_tensor = target_tensor.repeat(X_adv.shape[0],1,1,1)
        target_tensor.requires_grad_(False)
    else:
        target_tensor = ( (X+1)/2. ).clone().cuda().detach()
    #print(target_tensor.max(),target_tensor.min())
    if mode in need_decode_mode:
        X_tar = target_tensor
    else:
        X_tar = model.encode((target_tensor * 2.) - 1.).latent_dist.mean.detach()
    lossf = SSIM(data_range=1.0).to('cuda') if (mode == 3 or mode == 4) else torch.nn.MSELoss(reduction='mean')
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i  

        X_adv.requires_grad_(True)

        #loss = (model(X_adv).latent_dist.mean).norm()
        if mode in need_decode_mode:
            X_gen = model.decode(model.encode(X_adv).latent_dist.mean).sample
            #print(X_gen.mean(),X_gen.max(),X_gen.min())
            X_gen = ((X_gen + 1.)/2.).clip(0,1)
        else:
            X_gen = model.encode(X_adv).latent_dist.mean
        loss = lossf(X_gen,X_tar)
        if mode not in descent_mode:
            loss = - loss
        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")
        
        grad, = torch.autograd.grad(loss, [X_adv])
        #print(grad)
        X_adv = (X_adv - grad.detach().sign() * actual_step_size).detach()
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None    
        
        if mask is not None:
            X_adv.data *= mask
            
    return X_adv
def perform_pgd(imgs:list[Image.Image],eps:float,iter:int ,mode:int)->list[Image.Image]:
    """
    this function is used to perform pgd attack on a list of images
    """
    
    # You may want to play with the parameters of the attack to get stronger attacks, but we found the below params to be decent for our demo
    X = transform_to_tensor(imgs).cuda().bfloat16() * 2 - 1.
    pipe_img2img = get_pipe()
    adv_X = pgd(X, 
                model=pipe_img2img.vae,
                mode = mode, 
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
    parser.add_argument(
        '--mode',
        '-m',
        type = int,
        choices=[1,2,3,4,5,6],
        default= 1,
        help='\
            mode 1: l2 w target\
            mode 2: l2 o target\
            mode 3: SSIM w target\
            mode 4: SSIM o target\
            mode 5: l2(decode) w target\
            mode 6: l2(decode) o target'
    )
    args = parser.parse_args()
    return args
    
def main():
    args = parseargs()
    images = get_images_from_path(args.input_dir)
    max_size_limit = 3 if args.mode in need_decode_mode else 10
    base_counter = 0
    eps_int = int(args.eps*255)
    output_dir = os.path.join(args.output_dir,str(eps_int))
    if not os.path.exists(output_dir):
        print('creating output directory')
        os.makedirs(output_dir)
    for k in range(1 + len(images)//max_size_limit):
        print('processing batch '+str(k))
        adv_images=perform_pgd(images[k*max_size_limit:min((k+1)*max_size_limit,len(images))],args.eps,args.iter,args.mode)
        for i in range(len(adv_images)):
            save_folder = output_dir
            print('saving image at {}'.format(os.path.join(save_folder,'{}.jpg'.format(base_counter))))
            pil_image = Image.fromarray((adv_images[i].float().detach().cpu().numpy().transpose(1,2,0)*255).astype('uint8'))
            pil_image.save(os.path.join(save_folder,'{}.jpg'.format(base_counter)))
            base_counter += 1
        
if __name__ == '__main__':
    main()
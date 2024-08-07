import os
import argparse
from torch import nn
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision import transforms
from ignite.metrics import FID,SSIM,PSNR
from tqdm import tqdm
from ignite.engine import Engine
from torchvision.models import inception_v3
from pytorch_msssim import MS_SSIM
from torchmetrics.multimodal import clip_score
from transformers import CLIPModel,CLIPImageProcessor,CLIPProcessor
def eval_step(engine, batch):
    return batch
default_evaluator = Engine(eval_step)
def get_images_from_path(path:str,maximum_img_num:int) ->list[Image.Image]:
    """
    get images under the path
    """
    #print(path)
    for root, dirs, files in os.walk(path):
        images = []
        #try to sort the files in numerical order
        try:
            files = sorted(files,key=lambda x:int(x.split('.')[0]))
        except:
            print('files are not sorted in numerical order')
        for file in files:
            #print(file)
            if file.endswith('.jpg') or file.endswith('.png'):
                images.append(Image.open(os.path.join(root, file)).convert('RGB').resize((512,512),Image.Resampling.BILINEAR))
                if len(images) == maximum_img_num:
                    break
        return images
def transform_to_tensor(images:list[Image.Image])->torch.Tensor:
    """
    transform images to tensor
    """
    transform = ToTensor()
    return torch.stack([transform(img).float() for img in images])
    
def BRISQUE_LOSS(images:list[Image.Image])-> float:
    """
    calculate BRISQUE loss of each image and return the average loss
    """
    loss = 0
    br_measure = pyiqa.create_metric('brisque',device='cuda')
    print('brisque is lower the {}'.format('Better' if br_measure.lower_better else 'Worse'))
    for img in images:
        loss += br_measure(img)
    return loss/len(images)
def ILNIQE_LOSS(images:list[Image.Image])-> float:
    """
    calculate ILNIQE loss of each image and return the average loss
    """
    loss = 0
    il_measure = pyiqa.create_metric('ilniqe',device='cuda')
    print('ilniqe is lower the {}'.format('Better' if il_measure.lower_better else 'Worse'))
    for img in images:
        loss += il_measure(img)
    return loss/len(images)
def NIQE_LOSS(images:list[Image.Image])-> float:
    """
    calculate NIQE loss of each image and return the average loss
    """
    loss = 0
    ni_measure = pyiqa.create_metric('niqe',device='cuda')
    #print('niqe is lower the {}'.format('Better' if ni_measure.lower_better else 'Worse'))
    for img in images:
        loss += ni_measure(img)
    return loss.item()/len(images)
def CLIP_LOSS(images:list[Image.Image],class_name:str)-> float:
    """
    calculate CLIP loss of images and prompts(like the original clipiqa), return the average loss
    """
    loss = 0
    CLIPmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to('cuda')
    CLIPprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    prompt = ['a good photo of a {}'.format(class_name),'a bad photo of a {}'.format(class_name)]
    #print(prompt)
    inputs = CLIPprocessor(text=prompt, images=images, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = CLIPmodel(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        loss += probs[:, 1].sum()
    return loss.item()/len(images)
def CLIPIMG_LOSS(images:list[Image.Image],std_images:list[Image.Image])-> float:
    """
    calculate CLIPIQA loss of each image and return the average loss
    """
    loss = 0
    CLIPmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to('cuda')
    CLIPprocessor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    images = CLIPprocessor(images=images,return_tensors='pt').pixel_values.to('cuda')
    std_images = CLIPprocessor(images=std_images,return_tensors='pt').pixel_values.to('cuda')
    if images.shape[0] != std_images.shape[0]:
        batch_size = images.shape[0]//std_images.shape[0]
        images = torch.stack([images[k*batch_size] for k in range(images.shape[0]//batch_size)],dim=0)
        #images = torch.stack([images[k*batch_size] for k in range(images.shape[0]//batch_size)],dim=0)
        #print('batch size is {}'.format(batch_size))
    batch_size = 5
    batched_images = torch.split(images, batch_size)
    batched_std_images = torch.split(std_images, batch_size)
    with torch.no_grad():
        for k in range(len(batched_images)):
            embed_batch = CLIPmodel.get_image_features(batched_images[k])
            std_embed_batch = CLIPmodel.get_image_features(batched_std_images[k])
            loss += torch.nn.functional.cosine_similarity(embed_batch,std_embed_batch).sum()
    return loss.item()/len(images)
def SSIM_LOSS(images:list[Image.Image],std_images:list[Image.Image])-> float:
    """
    calculate SSIM loss of each image and return the average loss
    """
    images = transform_to_tensor(images)
    std_images = transform_to_tensor(std_images)
    ssim = SSIM(data_range=1.0)
    ssim.attach(default_evaluator, "ssim")
    state = default_evaluator.run([[images, std_images]])
    return state.metrics["ssim"]
def MSSSIM_LOSS(images:list[Image.Image],std_images:list[Image.Image])-> float:
    """
    calculate MSSSIM loss of each image and return the average loss
    """
    images = transform_to_tensor(images)
    std_images = transform_to_tensor(std_images)
    if images.shape[0] != std_images.shape[0]:
        batch_size = images.shape[0]//std_images.shape[0]
        images = torch.stack([images[k*batch_size] for k in range(images.shape[0]//batch_size)],dim=0)
    msssim = MS_SSIM(data_range=1.0)
    score = msssim(images,std_images)
    return score.item()
def parseargs()->argparse.Namespace:
    """
    parse arguments
    """
    parser = argparse.ArgumentParser(description='evaluate the quality of images')
    parser.add_argument(
        '--path',
        type=str,
        default=None,
        help='path to the images'
    )
    parser.add_argument(
        '--std_path',
        type=str,
        default=None,
        help='path to the standard images'
    )
    parser.add_argument(
        '--metric',
        '-m',
        type=str,
        default='BRISQUE',
        choices=['BRISQUE','ILNIQE','SSIM','PSNR','FID','NIQE','CLIPI2I','PI','CLIPT2I','MSSSIM'],
        help='metric to evaluate the quality of images'
    )
    parser.add_argument(
        '--class_name',
        '-c',
        type=str,
        default='person',
        choices=['person','painting'],
        help='class name of the images'
    )
    NR_Metrics = ['BRISQUE','ILNIQE','NIQE','CLIPT2I','PI']
    args = parser.parse_args()
    assert args.path is not None, 'path to the images is required'
    assert (args.std_path is not None) or (args.metric in NR_Metrics), 'path to the standard images is required with FR metrics'
    
    return args

def main():
    """
    parse args, evaluate images according to the args and print the loss
    """
    args = parseargs()
    maximum_img_num = 100
    images = get_images_from_path(args.path,maximum_img_num)
    if args.std_path is not None:
        std_images = get_images_from_path(args.std_path,maximum_img_num)
    if args.metric == 'CLIPI2I':
        print(CLIPIMG_LOSS(images,std_images))
    elif args.metric == 'CLIPT2I':
        print(CLIP_LOSS(images,args.class_name))
    elif args.metric == 'MSSSIM':
        print(MSSSIM_LOSS(images,std_images))
    elif args.metric == 'SSIM':
        print(SSIM_LOSS(images,std_images))
    elif args.metric == 'BRISQUE':
        print(BRISQUE_LOSS(images))
    elif args.metric == 'ILNIQE':
        print(ILNIQE_LOSS(images))
    elif args.metric == 'NIQE':
        print(NIQE_LOSS(images))
    else:
        raise ValueError('invalid metric')
if __name__ == '__main__':
    main()

    
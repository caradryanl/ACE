import os
import argparse
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
from torch import nn
import torch
from PIL import Image
from brisque import BRISQUE
from torchvision.transforms import ToTensor
from ignite.metrics import FID,SSIM,PSNR
from ignite.engine import Engine
def eval_step(engine, batch):
        return batch
default_evaluator = Engine(eval_step)
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
    
def BRISQUE_LOSS(images:list[Image.Image])-> float:
    """
    calculate BRISQUE loss of each image and return the average loss
    """
    loss = 0
    br_measure = BRISQUE()
    for img in images:
        loss += br_measure.score(img)
    return loss/len(images)

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

def PSNR_LOSS(images:list[Image.Image],std_images:list[Image.Image])-> float:
    """
    calculate PSNR loss of each image and return the average loss
    """
    images = transform_to_tensor(images)
    std_images = transform_to_tensor(std_images)
    psnr = PSNR(data_range=1.0)
    psnr.attach(default_evaluator, "psnr")
    state = default_evaluator.run([[images, std_images]])
    return state.metrics["psnr"]

def FID_LOSS(images:list[Image.Image],std_images:list[Image.Image])-> float:
    """
    calculate FID loss of each image and return the average loss
    """
    images = transform_to_tensor(images)
    std_images = transform_to_tensor(std_images)
    metric = FID()
    metric.attach(default_evaluator, "fid")
    state = default_evaluator.run([[images, std_images]])
    return state.metrics["fid"]

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
        type=str,
        default='BRISQUE',
        choices=['BRISQUE','SSIM','PSNR','FID'],
        help='metric to evaluate the quality of images'
    )
    args = parser.parse_args()
    assert args.path is not None, 'path to the images is required'
    assert (args.std_path is not None) or (args.metric == 'BRISQUE'), 'path to the standard images is required'
    
    return args

def main():
    """
    parse args, evaluate images according to the args and print the loss
    """
    args = parseargs()
    images = get_images_from_path(args.path)
    if args.std_path is not None:
        std_images = get_images_from_path(args.std_path)
    if args.metric == 'BRISQUE':
        print(BRISQUE_LOSS(images))
    elif args.metric == 'SSIM':
        print(SSIM_LOSS(images,std_images))
    elif args.metric == 'PSNR':
        print(PSNR_LOSS(images,std_images))
    elif args.metric == 'FID':
        print(FID_LOSS(images,std_images))
    else:
        raise ValueError('invalid metric')
if __name__ == '__main__':
    main()

    
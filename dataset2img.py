import os
from datasets import load_from_disk

def main():
    path = './datasetdict'
    datadict = load_from_disk(path)
    subdicts = [i for i in datadict]
    data_folder = './dataset'
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    for subdict in subdicts:
        #create subfolder according to the name of the dataset
        subfolder = os.path.join(data_folder, subdict)
        if not os.path.exists(subfolder):
            os.mkdir(subfolder)
        #save all the images in the subfolder
        imgs = datadict[subdict]['image']
        for i in range(len(imgs)):
            img = imgs[i]
            img.save(os.path.join(subfolder, str(i)+'.png'))
if __name__ == '__main__':
    folder = './dataset'
    for subfolder in os.listdir(folder):
        subfolder = os.path.join(folder, subfolder)
        print('accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial {}'.format(subfolder))
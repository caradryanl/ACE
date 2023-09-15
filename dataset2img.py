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
        # create subfolder according to the name of the dataset
        subfolder = os.path.join(data_folder, subdict)
        if not os.path.exists(subfolder):
            os.mkdir(subfolder)
        # save all the images in the subfolder
        imgs = datadict[subdict]['image']
        for i in range(len(imgs)):
            img = imgs[i]
            img.save(os.path.join(subfolder, str(i)+'.png'))


if __name__ == '__main__':
    folder = './outputs'
    for subfolder in os.listdir(folder):
        skip_folder = ['ASPL',"ASPL_M","data.zip"]
        if subfolder in skip_folder:
            continue
        datafolder = os.path.join(folder, subfolder,'noise-ckpt','5')
        save_folder = os.path.join('./attack_models', subfolder)
        classdata_dir = os.path.join('./data', subfolder)
        class_name = 'person' if subfolder.startswith(
            'celeba') else 'painting' if subfolder.startswith('wiki') else KeyError
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        if not os.path.exists(classdata_dir):
            os.mkdir(classdata_dir)
#        print('accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial {} --output_dir outputs/{} --class_data_dir {}'.format(datafolder,subfolder,classdata_dir))
        print('accelerate launch train_lora_dreambooth.py --instance_data_dir {} --output_dir {} --class_data_dir {} \
--output_format safe --instance_prompt "a photo of a sks {}" --class_prompt "a photo of a {} --mixed_precision bf16"  '.format(datafolder, save_folder, classdata_dir, class_name, class_name))

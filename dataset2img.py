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
    folder = './dataset'
    #folder = 'abalation/VAE/'
    subfolders = os.listdir(folder)
    subfolders = ['celeba-20-1135', 'celeba-20-121', 'celeba-20-1422', 'celeba-20-1499', 'celeba-20-1657',
                  'celeba-20-1672', 'celeba-20-1852', 'celeba-20-239', 'celeba-20-248', 'celeba-20-259',
                  'wikiart-summary-Abstract_Expressionism_gerhard-richt', 'wikiart-summary-Analytical_Cubism_pablo-piccaso',
                  'wikiart-summary-Art_Nouvean_Modern_felix-vallotton', 'wikiart-summary-Art_Nouvean_Modern_franklin-carmichael',
                  'wikiart-summary-Fauvism_henri-matisse', 'wikiart-summary-High_Renaissance_leonardo-da-vinci', 
                  'wikiart-summary-Impressionism_claude-monet', 'wikiart-summary-Pointillism_paul-signac', 
                  'wikiart-summary-Post_Impressionism-van-gogh', 'wikiart-summary-Rococo_canaletto']
    for subfolder in subfolders:
        skip_folder = ['ASPL',"ASPL_M","data.zip","VAE.sh"]
        if subfolder in skip_folder:
            continue
        #datafolder = os.path.join(folder, subfolder,'noise-ckpt','5')
        #datafolder = os.path.join('./benchmarks_data/advdm', subfolder,'noise-ckpt','1')
        #datafolder = os.path.join('attack_models/',subfolder,'lora_weight.safetensors')
        #datafolder = os.path.join('abalation/VAE/',subfolder,'16')
        #save_folder = os.path.join('./attack_models', subfolder)
        datafolder = os.path.join('./models/', subfolder,'lora_weight.safetensors')
        #save_folder = os.path.join('abalation/VAE_data', subfolder,'16')
        save_folder = os.path.join('./exp/t2i/clean/',subfolder,'8')
        classdata_dir = os.path.join('./data', subfolder)
        class_name = 'person' if subfolder.startswith(
            'celeba') else 'painting' if subfolder.startswith('wiki') else KeyError
        prompt = "a photo of a sks {}".format(class_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if not os.path.exists(classdata_dir):
            os.mkdir(classdata_dir)
#        print('accelerate launch benchmarks/advdm.py --instance_data_dir_for_adversarial {} --output_dir {} --class_data_dir {}'.format(datafolder,save_folder,classdata_dir))
#        print('accelerate launch train_lora_dreambooth.py --instance_data_dir {} --output_dir {} --class_data_dir {} \
#--output_format safe --instance_prompt "a photo of a sks {}" --class_prompt "a photo of a {}" --mixed_precision bf16  '.format(datafolder, save_folder, classdata_dir, class_name, class_name))
        #print('python benchmarks/photogaurd.py --input_dir {} --output_dir {}    '.format(datafolder, save_folder, classdata_dir, class_name, class_name))
        print('python i2i.py -m t2i -lp {} -op {} -spi 1000 --strength .4 -p "{}"'.format(datafolder, save_folder, prompt))
        #print('python i2i.py -m i2i -ip {} -op {}'.format(datafolder,save_folder))
import os
from datasets import load_from_disk
from tqdm import tqdm
from PIL import Image
import torch
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
    '''subfolders = ['celeba-20-1135', 'celeba-20-121', 'celeba-20-1422', 'celeba-20-1499', 'celeba-20-1657',
                  'celeba-20-1672', 'celeba-20-1852', 'celeba-20-239', 'celeba-20-248', 'celeba-20-259',
                  'wikiart-summary-Abstract_Expressionism_gerhard-richt', 'wikiart-summary-Analytical_Cubism_pablo-piccaso',
                  'wikiart-summary-Art_Nouvean_Modern_felix-vallotton', 'wikiart-summary-Art_Nouvean_Modern_franklin-carmichael',
                  'wikiart-summary-Fauvism_henri-matisse', 'wikiart-summary-High_Renaissance_leonardo-da-vinci', 
                  'wikiart-summary-Impressionism_claude-monet', 'wikiart-summary-Pointillism_paul-signac', 
                  'wikiart-summary-Post_Impressionism-van-gogh', 'wikiart-summary-Rococo_canaletto']'''
    subfolders = ['celeba-20-1135', 'celeba-20-121', 'celeba-20-1422', 'celeba-20-1499', 'celeba-20-1657',
                  'wikiart-summary-High_Renaissance_leonardo-da-vinci', 
                  'wikiart-summary-Impressionism_claude-monet', 'wikiart-summary-Pointillism_paul-signac', 
                  'wikiart-summary-Post_Impressionism-van-gogh', 'wikiart-summary-Rococo_canaletto']
    #subfolders = ['celeba-20-1135', 'celeba-20-121', 'celeba-20-1422', 'celeba-20-1499', 'celeba-20-1657']
    #subfolders = [
    #    'wikiart-summary-High_Renaissance_leonardo-da-vinci', 
    #              'wikiart-summary-Impressionism_claude-monet', 'wikiart-summary-Pointillism_paul-signac', 
    #              'wikiart-summary-Post_Impressionism-van-gogh', 'wikiart-summary-Rococo_canaletto'
    #]
    #subfolders = ['celeba-20-121']
    subfolder_attr = [
        'person' if subfolder.startswith('celeba') else 'painting' for subfolder in subfolders
    ]
    suffix = 'cross_model_'
    folder = './outputs'
    data_folder = './data'
    dataset_folder = './dataset'
    output_folder = './rebuttal/cross_model/'
    results_folder = os.path.join(output_folder, 'results')
    available_models = ['1-5','2-1','1-4']
    #available_models = ['1-5']
    model_suffix_path = './stable-diffusion/stable-diffusion-'
    imgs_folder = 'noise-ckpt/4/'
    cross_model_attack_command_template = 'accelerate launch attacks/aspl_lora_m.py \
--pretrained_model_name_or_path {} \
--instance_data_dir_for_adversarial {} \
--output_dir {} --class_data_dir {} \
--instance_prompt "a photo of a sks {}" --class_prompt "a photo of a {}" --mode fused \
--mixed_precision bf16 --max_train_steps 4 --checkpointing_iterations 1 --train_batch_size {}'
    cross_modal_train_command_template = 'accelerate launch train_lora_dreambooth.py \
--pretrained_model_name_or_path {} \
--instance_data_dir {} \
--output_dir {} --class_data_dir {} \
--output_format safe --instance_prompt "a photo of a sks {}" \
--class_prompt "a photo of a {}" --mixed_precision bf16'
    cross_modal_t2i_command_template = 'python i2i.py --pretrained_model_name_or_path {} -m t2i -lp {} -op {} -spi 100 -s .4 -p "a photo of a sks {}" '
    cross_modal_i2i_command_template = 'python i2i.py --pretrained_model_name_or_path {} -m i2i -ip {} -op {} -spi 5 -s .4 -p ""'
    cross_modal_t2i_eval_command_template = 'python evaluation.py --metric CLIPT2I --path {} --class_name {} > {}'
    cross_modal_i2i_msssim_eval_command_template = 'python evaluation.py --metric MSSSIM --path {} --std_path {} > {}'
    cross_modal_i2i_clipi2i_eval_command_template = 'python evaluation.py --metric CLIPI2I --path {} --std_path {} --class_name {} > {}'
    scripts_folder = './scripts'
    attack_cmds = []
    train_cmds = []
    i2i_cmds = []
    t2i_cmds = []
    t2i_eval_cmds = []
    i2i_eval_msssim_cmds = []
    i2i_eval_clipi2i_cmds = []
    txt_file_paths = {
            k:{
                v: {
                't2i':[],
                'i2i_msssim':[],
                'i2i_clipi2i':[]    
                } for v in available_models
            } for k in available_models
    }
    if True:
        for model in available_models:
            model_path = model_suffix_path + model
            for subfolder in subfolders:
                dataset_subfolder = os.path.join(dataset_folder, subfolder)
                t2i_eval_data_foler = os.path.join(output_folder,model, 't2i', subfolder,'data')
                result_folder = os.path.join(output_folder, 'results', subfolder,model,'data')
                t2i_result_file = os.path.join(result_folder,'t2i_eval.txt')
                i2i_result_msssim_file = os.path.join(result_folder,'i2i_msssim_eval.txt')
                i2i_result_clipi2i_file = os.path.join(result_folder,'i2i_clipi2i_eval.txt')
                if not os.path.exists(result_folder):
                    os.makedirs(result_folder, exist_ok=True)
                t2i_eval_cmd = cross_modal_t2i_eval_command_template.format(
                    t2i_eval_data_foler, subfolder_attr[0], t2i_result_file
                )
                i2i_eval_clipi2i_cmd = cross_modal_i2i_clipi2i_eval_command_template.format(
                    os.path.join(output_folder, model,'i2i', subfolder, 'data'), dataset_subfolder, subfolder_attr[0], i2i_result_clipi2i_file
                )
                i2i_eval_msssim_cmd = cross_modal_i2i_msssim_eval_command_template.format(
                    os.path.join(output_folder,model, 'i2i', subfolder, 'data'), dataset_subfolder, i2i_result_msssim_file
                )
                t2i_eval_cmds.append(t2i_eval_cmd)
                i2i_eval_msssim_cmds.append(i2i_eval_msssim_cmd)
                i2i_eval_clipi2i_cmds.append(i2i_eval_clipi2i_cmd)
                txt_file_paths[model][model]['t2i'].append(t2i_result_file)
                txt_file_paths[model][model]['i2i_msssim'].append(i2i_result_msssim_file)
                txt_file_paths[model][model]['i2i_clipi2i'].append(i2i_result_clipi2i_file)
        eval_cmd_path = os.path.join(scripts_folder, 'eval_data.sh')
        with open(eval_cmd_path, 'w') as f:
            f.write('\n'.join(t2i_eval_cmds))
            f.write('\n')
            f.write('\n'.join(i2i_eval_msssim_cmds))
            f.write('\n')
            f.write('\n'.join(i2i_eval_clipi2i_cmds))
        torch.save(txt_file_paths, os.path.join(scripts_folder, 'data_txt_paths.pt'))
    if True:
        dataset_subfolder = os.path.join(dataset_folder, subfolders[0])
        classdata_subfolder = os.path.join(data_folder, subfolders[0])
        for contrast in range(1,11,2):
            contrast_img = './data/MIST_repeated_{}.png'.format(contrast)
            attack_cmd = cross_model_attack_command_template.format(
                model_suffix_path + '1-5', dataset_subfolder, os.path.join(output_folder,'output','{}'.format(contrast)), classdata_subfolder, 
                subfolder_attr[0], subfolder_attr[0], 4)
            attack_cmd = attack_cmd + ' --target_img_path {}'.format(contrast_img)
            attack_cmds.append(attack_cmd)
            model_output_folder = os.path.join(output_folder, 'model', '{}'.format(contrast))
            train_cmd = cross_modal_train_command_template.format(
                model_suffix_path + '1-5', os.path.join(output_folder,'output','{}'.format(contrast),imgs_folder), model_output_folder, classdata_subfolder, subfolder_attr[0], subfolder_attr[0]
            )
            train_cmds.append(train_cmd)
            t2i_cmd = cross_modal_t2i_command_template.format(
                model_suffix_path + '1-5', os.path.join(model_output_folder, 'lora_weight.safetensors'),
                os.path.join(output_folder,'t2i','{}'.format(contrast)), subfolder_attr[0]
            )
            t2i_cmds.append(t2i_cmd)
            i2i_cmd = cross_modal_i2i_command_template.format(
                model_suffix_path + '1-5', os.path.join(output_folder,'output','{}'.format(contrast),imgs_folder),
                os.path.join(output_folder,'i2i','{}'.format(contrast))
            )
            i2i_cmds.append(i2i_cmd)
            t2i_eval_cmd = cross_modal_t2i_eval_command_template.format(
                os.path.join(output_folder,'t2i','{}'.format(contrast)), subfolder_attr[0], os.path.join(output_folder,'results','{}'.format(contrast),'t2i_eval.txt')
            )
            t2i_eval_cmds.append(t2i_eval_cmd)
        all_in_one_save_path = './scripts/target_repeat.sh'
        with open(all_in_one_save_path, 'w') as f:
            f.write('\n'.join(attack_cmds))
            f.write('\n')
            f.write('\n'.join(train_cmds))
            f.write('\n')
            f.write('\n'.join(t2i_cmds))
            f.write('\n')
            f.write('\n'.join(t2i_eval_cmds))
            f.write('\n')
            f.write('\n'.join(i2i_cmds))
        exit()
    if False:
        for i,subfolder in enumerate(subfolders):
            for model in available_models:
                model_path = model_suffix_path + model
                output_subfolder = os.path.join('./rebuttal/cross_model', model)
                dataset_subfolder = os.path.join(dataset_folder, subfolder)
                classdata_subfolder = os.path.join(data_folder, subfolder)
                train_cmd = cross_modal_train_command_template.format(
                    model_path, dataset_subfolder, os.path.join(output_subfolder,'model',subfolder,'data'), classdata_subfolder, subfolder_attr[i], subfolder_attr[i]
                )
                if model.startswith('2-'):
                    train_cmd += ' --resolution 768'
                else:
                    train_cmd = '#' + train_cmd
                train_cmds.append(train_cmd)
                
                t2i_cmd = cross_modal_t2i_command_template.format(
                    model_path, os.path.join(output_subfolder,'model',subfolder,'data', 'lora_weight.safetensors'), os.path.join(output_subfolder, 't2i', subfolder,'data'), subfolder_attr[i]
                )
                if model.startswith('2-'):
                    t2i_cmd += ' -mss 1'
                else:
                    t2i_cmd = '#' + t2i_cmd
                t2i_cmds.append(t2i_cmd)
                i2i_cmd = cross_modal_i2i_command_template.format(
                    model_path, dataset_subfolder, os.path.join(output_subfolder, 'i2i', subfolder, 'data')
                )
                if model.startswith('2-'):
                    i2i_cmd += ' -r 768'
                i2i_cmds.append(i2i_cmd)
        # write to scripts
        train_cmds_save_path = './scripts/train_data.sh'
        t2i_cmds_save_path = './scripts/t2i_data.sh'
        i2i_cmds_save_path = './scripts/i2i_data.sh'
        with open(train_cmds_save_path, 'w') as f:
            f.write('\n'.join(train_cmds))
        with open(t2i_cmds_save_path, 'w') as f:
            f.write('\n'.join(t2i_cmds))
        with open(i2i_cmds_save_path, 'w') as f:
            f.write('\n'.join(i2i_cmds))
    for i,subfolder in enumerate(subfolders):
        #subfolder = os.path.join(folder, subfolder)
        data_subfolder = os.path.join(data_folder, subfolder)
        dataset_subfolder = os.path.join(dataset_folder, subfolder)
        #assert os.path.exists(subfolder), 'subfolder not exists'
        assert os.path.exists(data_subfolder), 'data subfolder not exists'
        result_subfolder = os.path.join(results_folder, subfolder)
        for source_model in available_models:
            output_subfolder = os.path.join(output_folder, source_model)
            source_model_path = model_suffix_path + source_model
            attack_img_output_folder = os.path.join(output_subfolder, 'output', subfolder)
            batchsize = 4 if source_model.startswith('1-') else 2
            cross_modal_attack_cmd = cross_model_attack_command_template.format(
                source_model_path, dataset_subfolder, attack_img_output_folder, data_subfolder, subfolder_attr[i], subfolder_attr[i], batchsize
            )
            #if source_model.startswith('1-'):
            #    cross_modal_attack_cmd = '#' + cross_modal_attack_cmd
            attack_cmds.append(cross_modal_attack_cmd)
            model_output_folder = os.path.join(output_subfolder, 'model', subfolder)
            t2i_output_folder = os.path.join(output_subfolder, 't2i', subfolder)
            i2i_output_folder = os.path.join(output_subfolder, 'i2i', subfolder)
            for target_model in available_models:
                target_model_path = model_suffix_path + target_model
                result_output_folder = os.path.join(result_subfolder, source_model, target_model)
                if not os.path.exists(result_output_folder):
                    os.makedirs(result_output_folder,exist_ok=True)
                model_target_output_folder = os.path.join(model_output_folder, target_model)
                cross_modal_train_cmd = cross_modal_train_command_template.format(
                    target_model_path, os.path.join(attack_img_output_folder,imgs_folder), model_target_output_folder, data_subfolder, subfolder_attr[i], subfolder_attr[i]
                )# attack_ims will be saved in attack_img_output_folder/imgs_folder
                target_model_file_path = os.path.join(model_target_output_folder, 'lora_weight.safetensors')
                #assert os.path.isfile(target_model_file_path), 'target model file not exists or not safetensors'
                # now t2i eval:
                t2i_target_output_folder = os.path.join(t2i_output_folder, target_model)
                cross_modal_t2i_cmd = cross_modal_t2i_command_template.format(
                    target_model_path,target_model_file_path, t2i_target_output_folder, subfolder_attr[i]
                )
                t2i_txt_path = os.path.join(result_output_folder, 't2i_eval.txt')
                txt_file_paths[source_model][target_model]['t2i'].append(t2i_txt_path)
                t2i_eval_cmd = cross_modal_t2i_eval_command_template.format(
                    t2i_target_output_folder, subfolder_attr[i], t2i_txt_path
                )
                t2i_eval_cmds.append(t2i_eval_cmd)
                # now i2i eval:
                i2i_target_output_folder = os.path.join(i2i_output_folder, target_model)
                cross_modal_i2i_cmd = cross_modal_i2i_command_template.format(
                     target_model_path,os.path.join(attack_img_output_folder,imgs_folder), i2i_target_output_folder
                )
                msssim_txt_path = os.path.join(result_output_folder, 'i2i_msssim_eval.txt')
                i2i_eval_msssim_cmd = cross_modal_i2i_msssim_eval_command_template.format(
                    i2i_target_output_folder, dataset_subfolder, msssim_txt_path
                )
                clipi2i_txt_path = os.path.join(result_output_folder, 'i2i_clipi2i_eval.txt')
                i2i_eval_clipi2i_cmd = cross_modal_i2i_clipi2i_eval_command_template.format(
                    i2i_target_output_folder, dataset_subfolder, subfolder_attr[i], clipi2i_txt_path
                )
                txt_file_paths[source_model][target_model]['i2i_msssim'].append(msssim_txt_path)
                txt_file_paths[source_model][target_model]['i2i_clipi2i'].append(clipi2i_txt_path)
                i2i_eval_msssim_cmds.append(i2i_eval_msssim_cmd)
                i2i_eval_clipi2i_cmds.append(i2i_eval_clipi2i_cmd)
                train_cmds.append(cross_modal_train_cmd)
                t2i_cmds.append(cross_modal_t2i_cmd)
                i2i_cmds.append(cross_modal_i2i_cmd)
    # write to scripts
    attack_script_path = os.path.join(scripts_folder, suffix+'attack.sh')
    train_script_path = os.path.join(scripts_folder, suffix+'train.sh')
    i2i_script_path = os.path.join(scripts_folder, suffix+'i2i.sh')
    t2i_script_path = os.path.join(scripts_folder, suffix+'t2i.sh')
    t2i_eval_script_path = os.path.join(scripts_folder, suffix+'t2i_eval.sh')
    i2i_eval_msssim_script_path = os.path.join(scripts_folder, suffix+'i2i_eval_msssim.sh')
    i2i_eval_clipi2i_script_path = os.path.join(scripts_folder, suffix+'i2i_eval_clipi2i.sh')
    all_in_one_script = "\
./scripts/{}attack.sh\n\
./scripts/{}train.sh\n\
./scripts/{}i2i.sh\n\
./scripts/{}t2i.sh\n\
./scripts/{}t2i_eval.sh\n\
./scripts/{}i2i_eval_msssim.sh\n\
./scripts/{}i2i_eval_clipi2i.sh\n\
".format(suffix,suffix,suffix,suffix,suffix,suffix,suffix)
    all_in_one_script_path = os.path.join(scripts_folder, suffix+'all_in_one.sh')
    txt_paths = os.path.join(scripts_folder, suffix+'txt_paths.pt')
    #with open(attack_script_path, 'w') as f:
    #    f.write('\n'.join(attack_cmds))
    #with open(train_script_path, 'w') as f:
    #    f.write('\n'.join(train_cmds))
    #with open(i2i_script_path, 'w') as f:
    #    f.write('\n'.join(i2i_cmds))
    #with open(t2i_script_path, 'w') as f:
    #    f.write('\n'.join(t2i_cmds))
    #with open(t2i_eval_script_path, 'w') as f:
    #    f.write('\necho -n \\# \n'.join(t2i_eval_cmds))
    with open(i2i_eval_msssim_script_path, 'w') as f:
        f.write('\necho -n \\# \n'.join(i2i_eval_msssim_cmds))
    with open(i2i_eval_clipi2i_script_path, 'w') as f:
        f.write('\necho -n \\# \n'.join(i2i_eval_clipi2i_cmds))
    #with open(all_in_one_script_path, 'w') as f:
    #    f.write(all_in_one_script)
    torch.save(txt_file_paths, txt_paths)
           
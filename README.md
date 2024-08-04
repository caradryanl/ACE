# This is the official implementation of the paper "Understanding and improving adversarial attacks on latent diffusion model"

**Currently, for full and more convient implementation, please see https://github.com/mist-project/mist-v2**

## Installation

This repository is based on [PyTorch](https://pytorch.org/). Specifically, please make sure your environment satisfies `python==3.10` and `cuda >= 11.6`. We do not guarantee the code's performance on other versions.

If you're a conda user, simply run:
```bash
conda create -n ldm python=3.10
conda activate ldm
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
You may change the cuda and pytorch version according to your own environment. However, we strongly recommend to use pytorch with version no less than `1.13.1`.

Then, install the other requirements:
```bash
pip install -r requirements.txt
```

## Usage

**Notice:**
For `attacks/aspl_lora_m.py`, `train_lora_dreambooth.py`,`i2i.py` , they all have a command line argument called `--pretrained_model_name_or_path`. This argument is used to specify the backbone model used in the attack, training and inference. The path should be a directory with a standard huggingface stabled-diffusion directory structure.

To run a default attack, simply run:

```bash
accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial $DATA_DIR --output_dir $OUTPUT_DIR --class_data_dir $CLASS_DATA_DIR --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1
```

If class data is not generated yet, the program will automatically generate it under the directory `$CLASS_DATA_DIR`. The generated class data will be used in the following attacks. You may change the prompt according to your own data.


### Attacks

As mentioned in paper, two types of attacks are implemented. To switch between them, simply add `--mode [mode]` to the command line. `[mode] = [lunet | fused]`. The default mode is `lunet`,which is called `ACE` in the paper. The `fused` mode is called `ACE+` in the paper.

### Parameters

You can play with the parameters freely. But we find a set of parameters that works well in our cases (already set as default). You may change them according to your own data.

### Mixed Precision & GPU support

All the attacks are run on bfloat16 and a Nvidia GPU is required. Other precision and device(including distributed attack) is not yet supported.

### Low VRAM support

We highly concerns the VRAM usage of our attacks, as we know personal computers usually have limited VRAM. Through some effort, we've managed to reduce the VRAM usage to a relatively low level. To enable low VRAM support, simply add `--low_vram_mode` to the command line. However, this may slow down the attack.

**For fully implemented Low VRAM Mode and CPU Mode, please see [Here](https://github.com/mist-project/mist-v2)**

To use the low VRAM mode, you need to install xformers first. Please see the official [repo](https://github.com/facebookresearch/xformers) for detailed installation instructions.

### Evaluation

#### training
To evaluate the performance of attacks on LoRA, you can train a lora model by running:

```bash
accelerate launch train_lora_dreambooth.py --instance_data_dir $ATTACK_DATA_DIR --output_dir $OUTPUT_MODEL_DIR --class_data_dir $CLASS_DATA_DIR --output_format safe --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16
```

You may change the prompt according to your own data. The trained model will be saved under the directory `$OUTPUT_MODEL_DIR`. Like the class data we use in attacks, the class data will be generated automatically under the directory `$CLASS_DATA_DIR` if it's not generated yet.

#### inference
We also provide a script to do txt2img(on lora model), img2img(on original model) inference. To use it, simply run:

```bash
python i2i.py -m [mode]=[t2i|i2i] -lp [lora_model_path] -ip [input_img_folder] -op [output_img_folder] -spi $N -s [strength] -p [prompt]
```

For txt2img inference, `-spi` is the number of samples per prompt. For img2img inference, `-spi` is the number of samples per image. Default value is 1. For txt2img, `[strength]` is the steps of denoising (more precisely, int(50 * strength) is the inference step). For img2img, `[strength]` is the strength of the denoising. The default value is 0.3, as the setup in the paper.

simply omit the unnecessary command-line parameters. For example, to do generate 10 image on lora model, simply run:

```bash
python i2i.py -m t2i -lp $LORA_MODEL_PATH -op $OUTPUT_IMG_FOLDER -spi 10 -p "a photo of a sks person"
```

#### Numerical Evaluation

We also provide a python file to do numerical evaluation stated in our paper. To use it, simply run:

```bash
python evaluate.py -m [mode] --path [path] --std_path [std_path] -c [class name]
```

`[mode]` is the metric of the evaluation. `[path]` is the path of the images to be evaluated. `[std_path]` is the path of the reference images(for metric with references). `[class name]` is the class name of the images to be evaluated, which is used in the CLIPIQA metric. We utilize the prompt `a good photo of a [class name]` and `a bad photo of a [class name]` to evaluate the quality of the images.

## Acknowledgement
Our code is based on [Anti-Dreambooth](https://github.com/VinAIResearch/Anti-DreamBooth). We also utilize [xformers](https://github.com/facebookresearch/xformers) to lower the VRAM usage. We thank the authors for their great work.






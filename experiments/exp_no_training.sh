accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/celeba/1135 --output_dir lora_repo/celeba/training/no_training --class_data_dir data/class/celeba/attack --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16 --max_train_steps 0 --checkpointing_iterations 1 --max_adv_train_steps 250

accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/celeba/training/no_training --output_dir lora_repo/celeba/outputs/no_training --class_data_dir data/class/celeba/training --output_format safe --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16

python i2i.py -m t2i -lp lora_repo/celeba/outputs/no_training/lora_weight.safetensors -op outputs/celeba/ace_t2i/no_training -spi 100 -p "a photo of a sks person"
python i2i.py -m i2i -ip lora_repo/celeba/training/no_training/ -op outputs/celeba/ace_i2i/no_training -spi 1 -p "a photo of a sks person"

python evaluation.py -m CLIPT2I --path outputs/celeba/ace_t2i/no_training --std_path data/celeba/1135 -c person
python evaluation.py -m MSSSIM --path outputs/celeba/ace_i2i/no_training --std_path data/celeba/1135 -c person
python evaluation.py -m CLIPI2I --path outputs/celeba/ace_i2i/no_training --std_path data/celeba/1135 -c person




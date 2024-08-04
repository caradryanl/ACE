# accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/celeba/121 --output_dir lora_repo/training/121 --class_data_dir data/class/attack --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1
# accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/celeba/1135 --output_dir lora_repo/training/1135 --class_data_dir data/class/attack --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1
# accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/celeba/1422 --output_dir lora_repo/training/1422 --class_data_dir data/class/attack --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1
# accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/celeba/1499 --output_dir lora_repo/training/1499 --class_data_dir data/class/attack --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1
# accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/celeba/1657 --output_dir lora_repo/training/1657 --class_data_dir data/class/attack --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1

# accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/training/121/noise-ckpt/5 --output_dir lora_repo/ace-outputs/121 --class_data_dir data/class/training --output_format safe --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16
# accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/training/1135/noise-ckpt/5 --output_dir lora_repo/ace-outputs/1135 --class_data_dir data/class/training --output_format safe --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16
# accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/training/1422/noise-ckpt/5 --output_dir lora_repo/ace-outputs/1422 --class_data_dir data/class/training --output_format safe --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16
# accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/training/1499/noise-ckpt/5 --output_dir lora_repo/ace-outputs/1499 --class_data_dir data/class/training --output_format safe --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16
# accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/training/1657/noise-ckpt/5 --output_dir lora_repo/ace-outputs/1657 --class_data_dir data/class/training --output_format safe --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16


# python i2i.py -m t2i -lp lora_repo/ace-outputs/121/lora_weight.safetensors -op outputs/ace/121 -spi 100 -p "a photo of a sks person"
# python i2i.py -m t2i -lp lora_repo/ace-outputs/1135/lora_weight.safetensors -op outputs/ace/1135 -spi 100 -p "a photo of a sks person"
# python i2i.py -m t2i -lp lora_repo/ace-outputs/1422/lora_weight.safetensors -op outputs/ace/1422 -spi 100 -p "a photo of a sks person"
# python i2i.py -m t2i -lp lora_repo/ace-outputs/1499/lora_weight.safetensors -op outputs/ace/1499 -spi 100 -p "a photo of a sks person"
# python i2i.py -m t2i -lp lora_repo/ace-outputs/1657/lora_weight.safetensors -op outputs/ace/1657 -spi 100 -p "a photo of a sks person"

python i2i.py -m i2i -ip lora_repo/training/121/noise-ckpt/5 -op outputs/ace_i2i/121 -spi 1 -p "a photo of a sks person"
python i2i.py -m i2i -ip lora_repo/training/1135/noise-ckpt/5 -op outputs/ace_i2i/1135 -spi 1 -p "a photo of a sks person"
python i2i.py -m i2i -ip lora_repo/training/1422/noise-ckpt/5 -op outputs/ace_i2i/1422 -spi 1 -p "a photo of a sks person"
python i2i.py -m i2i -ip lora_repo/training/1499/noise-ckpt/5 -op outputs/ace_i2i/1499 -spi 1 -p "a photo of a sks person"
python i2i.py -m i2i -ip lora_repo/training/1657/noise-ckpt/5 -op outputs/ace_i2i/1657 -spi 1 -p "a photo of a sks person"

# python evaluation.py -m CLIPT2I --path outputs/ace/121 --std_path data/celeba/121 -c person
# python evaluation.py -m CLIPT2I --path outputs/ace/1135 --std_path data/celeba/1135 -c person
# python evaluation.py -m CLIPT2I --path outputs/ace/1422 --std_path data/celeba/1422 -c person
# python evaluation.py -m CLIPT2I --path outputs/ace/1499 --std_path data/celeba/1499 -c person
# python evaluation.py -m CLIPT2I --path outputs/ace/1657 --std_path data/celeba/1657 -c person

python evaluation.py -m MSSSIM --path outputs/ace_i2i/121 --std_path data/celeba/121 -c person
python evaluation.py -m MSSSIM --path outputs/ace_i2i/1135 --std_path data/celeba/1135 -c person
python evaluation.py -m MSSSIM --path outputs/ace_i2i/1422 --std_path data/celeba/1422 -c person
python evaluation.py -m MSSSIM --path outputs/ace_i2i/1499 --std_path data/celeba/1499 -c person
python evaluation.py -m MSSSIM --path outputs/ace_i2i/1657 --std_path data/celeba/1657 -c person

python evaluation.py -m CLIPI2I --path outputs/ace_i2i/121 --std_path data/celeba/121 -c person
python evaluation.py -m CLIPI2I --path outputs/ace_i2i/1135 --std_path data/celeba/1135 -c person
python evaluation.py -m CLIPI2I --path outputs/ace_i2i/1422 --std_path data/celeba/1422 -c person
python evaluation.py -m CLIPI2I --path outputs/ace_i2i/1499 --std_path data/celeba/1499 -c person
python evaluation.py -m CLIPI2I --path outputs/ace_i2i/1657 --std_path data/celeba/1657 -c person




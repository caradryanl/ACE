accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/celeba/1135 --output_dir lora_repo/celeba/training/attack_10 --class_data_dir data/class/celeba/attack --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1 --max_adv_train_steps 10
accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/celeba/1135 --output_dir lora_repo/celeba/training/attack_25 --class_data_dir data/class/celeba/attack --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1 --max_adv_train_steps 25
accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/celeba/1135 --output_dir lora_repo/celeba/training/attack_50 --class_data_dir data/class/celeba/attack --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1 --max_adv_train_steps 50
accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/celeba/1135 --output_dir lora_repo/celeba/training/attack_100 --class_data_dir data/class/celeba/attack --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1 --max_adv_train_steps 100
accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/celeba/1135 --output_dir lora_repo/celeba/training/attack_150 --class_data_dir data/class/celeba/attack --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1 --max_adv_train_steps 150

accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/celeba/training/attack_10/noise-ckpt/5 --output_dir lora_repo/celeba/outputs/attack_10 --class_data_dir data/class/celeba/training --output_format safe --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16
accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/celeba/training/attack_25/noise-ckpt/5 --output_dir lora_repo/celeba/outputs/attack_25 --class_data_dir data/class/celeba/training --output_format safe --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16
accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/celeba/training/attack_50/noise-ckpt/5 --output_dir lora_repo/celeba/outputs/attack_50 --class_data_dir data/class/celeba/training --output_format safe --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16
accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/celeba/training/attack_100/noise-ckpt/5 --output_dir lora_repo/celeba/outputs/attack_100 --class_data_dir data/class/celeba/training --output_format safe --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16
accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/celeba/training/attack_150/noise-ckpt/5 --output_dir lora_repo/celeba/outputs/attack_150 --class_data_dir data/class/celeba/training --output_format safe --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16


python i2i.py -m t2i -lp lora_repo/celeba/outputs/attack_10/lora_weight.safetensors -op outputs/celeba/ace_t2i/attack_10 -spi 100 -p "a photo of a sks person"
python i2i.py -m t2i -lp lora_repo/celeba/outputs/attack_25/lora_weight.safetensors -op outputs/celeba/ace_t2i/attack_25 -spi 100 -p "a photo of a sks person"
python i2i.py -m t2i -lp lora_repo/celeba/outputs/attack_50/lora_weight.safetensors -op outputs/celeba/ace_t2i/attack_50 -spi 100 -p "a photo of a sks person"
python i2i.py -m t2i -lp lora_repo/celeba/outputs/attack_100/lora_weight.safetensors -op outputs/celeba/ace_t2i/attack_100 -spi 100 -p "a photo of a sks person"
python i2i.py -m t2i -lp lora_repo/celeba/outputs/attack_150/lora_weight.safetensors -op outputs/celeba/ace_t2i/attack_150 -spi 100 -p "a photo of a sks person"

python i2i.py -m i2i -ip lora_repo/celeba/training/attack_10/noise-ckpt/5 -op outputs/celeba/ace_i2i/attack_10 -spi 1 -p "a photo of a sks person"
python i2i.py -m i2i -ip lora_repo/celeba/training/attack_25/noise-ckpt/5 -op outputs/celeba/ace_i2i/attack_25 -spi 1 -p "a photo of a sks person"
python i2i.py -m i2i -ip lora_repo/celeba/training/attack_50/noise-ckpt/5 -op outputs/celeba/ace_i2i/attack_50 -spi 1 -p "a photo of a sks person"
python i2i.py -m i2i -ip lora_repo/celeba/training/attack_100/noise-ckpt/5 -op outputs/celeba/ace_i2i/attack_100 -spi 1 -p "a photo of a sks person"
python i2i.py -m i2i -ip lora_repo/celeba/training/attack_150/noise-ckpt/5 -op outputs/celeba/ace_i2i/attack_150 -spi 1 -p "a photo of a sks person"

python evaluation.py -m CLIPT2I --path outputs/celeba/ace_t2i/attack_10 --std_path data/celeba/1135 -c person
python evaluation.py -m CLIPT2I --path outputs/celeba/ace_t2i/attack_25 --std_path data/celeba/1135 -c person
python evaluation.py -m CLIPT2I --path outputs/celeba/ace_t2i/attack_50 --std_path data/celeba/1135 -c person
python evaluation.py -m CLIPT2I --path outputs/celeba/ace_t2i/attack_100 --std_path data/celeba/1135 -c person
python evaluation.py -m CLIPT2I --path outputs/celeba/ace_t2i/attack_150 --std_path data/celeba/1135 -c person

python evaluation.py -m MSSSIM --path outputs/celeba/ace_i2i/attack_10 --std_path data/celeba/1135 -c person
python evaluation.py -m MSSSIM --path outputs/celeba/ace_i2i/attack_25 --std_path data/celeba/1135 -c person
python evaluation.py -m MSSSIM --path outputs/celeba/ace_i2i/attack_50 --std_path data/celeba/1135 -c person
python evaluation.py -m MSSSIM --path outputs/celeba/ace_i2i/attack_100 --std_path data/celeba/1135 -c person
python evaluation.py -m MSSSIM --path outputs/celeba/ace_i2i/attack_150 --std_path data/celeba/1135 -c person

python evaluation.py -m CLIPI2I --path outputs/celeba/ace_i2i/attack_10 --std_path data/celeba/1135 -c person
python evaluation.py -m CLIPI2I --path outputs/celeba/ace_i2i/1135 --std_path data/celeba/1135 -c person
python evaluation.py -m CLIPI2I --path outputs/celeba/ace_i2i/attack_50 --std_path data/celeba/1135 -c person
python evaluation.py -m CLIPI2I --path outputs/celeba/ace_i2i/attack_100 --std_path data/celeba/1135 -c person
python evaluation.py -m CLIPI2I --path outputs/celeba/ace_i2i/attack_150 --std_path data/celeba/1135 -c person




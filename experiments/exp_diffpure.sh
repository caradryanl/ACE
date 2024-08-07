accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/celeba/training/diffpure --output_dir lora_repo/celeba/outputs/diffpure --class_data_dir data/class/celeba/training --output_format safe --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16

python i2i.py -m t2i -lp lora_repo/celeba/outputs/diffpure/lora_weight.safetensors -op outputs/celeba/ace_t2i/diffpure -spi 100 -p "a photo of a sks person"

python i2i.py -m i2i -ip lora_repo/celeba/training/diffpure -op outputs/celeba/ace_i2i/diffpure -spi 1 -p "a photo of a sks person"

python evaluation.py -m CLIPT2I --path outputs/celeba/ace_t2i/diffpure --std_path data/celeba/1135 -c person

python evaluation.py -m MSSSIM --path outputs/celeba/ace_i2i/diffpure --std_path data/celeba/1135 -c person

python evaluation.py -m CLIPI2I --path outputs/celeba/ace_i2i/diffpure --std_path data/celeba/1135 -c person




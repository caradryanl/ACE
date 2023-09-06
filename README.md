## Improved Adversarial Attack on Latent Diffusion Models

#### Run AdvDM++ (Anti-LoRA)


```
    accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial $DATA_TO_BE_ATTACKED 
```

#### Run LoRA

Put the training data in ```lora_repo/data/training```

```
    accelerate launch train_lora_dreambooth.py
```

#### Run inference with LoRA

Use the first two blocks in the notebook ```run_inference.ipynb```


#### Environment

See ```requirements.txt```. It is not completed so please help complete it.
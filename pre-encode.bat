python ./pre_encode.py ^
    --ckpt-path ./vae_model.ckpt ^
    --model-config ./vae_model_config.json ^
    --batch-size 8 ^
    --dataset-config pe_dataset_config.json ^
    --output-path ./pre_encoded ^
    --model-half ^
    --sample-size 131072 ^


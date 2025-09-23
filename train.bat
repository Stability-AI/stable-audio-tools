python train.py ^
    --name saos1 ^
    --pretrained-ckpt-path .\sao_small\base_model.ckpt ^
    --model-config .\sao_small\base_model_config.json ^
    --batch-size 2 ^
    --num-workers 4 ^
    --seed 1937401721 ^
    --checkpoint-every 1000 ^
    --dataset-config dataset_config.json ^
    --save-dir outputs ^
    --precision 16-mixed ^


# Horizon

Extracting information useful for UAV guidance from images.

Train the model

```bash
train.py --model efficientnet_b0 --data_dir data/horizon --num_classes 16 --image_size 64 --batch_size 8 --num_workers 0 --log_every_n_steps 20 --gpus 1 --max_epochs 500
```

Train regression

```bash
train_regression.py --model efficientnetv2_m --data_dir data/horizon --image_size 64 --batch_size 8 --num_workers 0 --log_every_n_steps 20 --gpus 1, --max_epochs 400 --lr 1e-3 --gamma 0.99
```
# Horizon

Extracting information useful for UAV guidance from images.

Train the model

```bash
train.py --model efficientnet_b0 --data_dir data/horizon --num_classes 16 --image_size 64 --batch_size 8 --num_workers 0 --log_every_n_steps 20 --gpus 2
```


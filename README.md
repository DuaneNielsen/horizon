# Horizon

Extracting information useful for UAV guidance from images.

setup
```commandline
wandb login
```

Train the model

```bash
python train_classifier.py --model efficientnet_b0 --data_dir data/horizon --num_classes 16 --image_size 64 --batch_size 8 --num_workers 0 --log_every_n_steps 20 --gpus 1 --max_epochs 1000 --lr 1e-3 --gamma 0.995
```

Train regression

```bash
python train_regression.py --model efficientnetv2_m --data_dir data/horizon --image_size 64 --batch_size 8 --num_workers 0 --log_every_n_steps 20 --gpus 0, --max_epochs 1000 --lr 1e-3 --gamma 0.995
```

Re-run validation on a checkpoint

```bash
python train_regression.py --validate_checkpoint checkpoints/crimson-sea-41/epoch\=746-step\=33614.ckpt
```

Run inference on video stream

```bash
python train_regression.py --predict_checkpoint checkpoints/crimson-sea-41/epoch=836-step=37664.ckpt  --gpu 0, --no_mask 
```

## installing



```bash
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install torch-tensorrt -f https://github.com/NVIDIA/Torch-TensorRT/releases
```

convert onnx model to tensorRT engine
```bash
trtexec --onnx='epoch=173-step=7829.onnx' --explicitBatch --workspace=2048 --saveEngine=epoch=173-step=7829.engine
```
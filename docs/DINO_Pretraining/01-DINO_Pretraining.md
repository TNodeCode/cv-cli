# Pretraining using self-distillation

## Pretrain model

```bash
#!/bin/bash
python projects/dino/main_dino.py \
    --arch mmdet:faster-rcnn-50 \
    --data_path ./datasets/spine/train/ \
    --image_size 512 \
    --embed_dim 256 \
    --batch_size_per_gpu 8 \
    --local_crops_number 8 \
    --local_crops_scale 0.5 0.7 \
    --global_crops_scale 0.7 1.0 \
    --output_dir work_dirs/faster_rcnn_50 \
    --epochs 150 \
    --saveckp_freq 1
```

### Available models

| Model |
| --- |
| mmdet:faster-rcnn-50 |
| mmdet:faster-rcnn-101 |
| mmdet:faster-rcnn-x101-32 |
| mmdet:faster-rcnn-x101-64 |
| mmdet:deformable-detr |
| mmdet:dino-swin |
| mmdet:dino-vit |
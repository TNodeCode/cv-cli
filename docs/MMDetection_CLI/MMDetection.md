# MMDetection CLI

## Extract Backbone

Load Faster R-CNN model from configuration file `work_dirs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py`, save backbone state dict at `work_dirs/backbone1.pth` and load Faster-RCNN weights from `work_dirs/backbone1.pth`. The third argument is optional. If no weights are provided the default pretrained weights will be loaded.

```bash
cv mmdet extract-backbone \
    --config-file work_dirs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    --load-from work_dirs/faster_rcnn/epoch_1.pth \
    --output-file work_dirs/faster_rcnn_backbone.pth
```

## Copy Backbone

Copy the backbone from one model to the other model. The example command loads a Faster R-CNN model based on the configuration file `work_dirs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py` with weights from `work_dirs/faster_rcnn/epoch_1.pth` and copies the backbone to the Deformable DETR model based on the configuration file `mmdetection/configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py` and saves the Deformable DETR weights at `work_dirs/def_detr.pth`.

```bash
cv mmdet copy-backbone \
    --source-config-file work_dirs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    --load-source-from work_dirs/faster_rcnn/epoch_1.pth \
    --target-config-file mmdetection/configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py \
    --output-file work_dirs/updated_deformable_detr.pth
```

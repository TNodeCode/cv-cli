# MMDetection CLI

## Evaluation

When you have a ground truth annotation file in the COCO annotation style format and a CSV file containing detections for several epochs you can compute the COCO evaluation scores using the following command.

```bash
$ cv mmdet eval \
    --gt datasets/7s/annotations/instances_train2017.json \
    --det tmp/results/detections_train.csv \
    --out tmp/results/eval_train.csv
```

The detection CSV file should have the following format:

```csv
epoch,filename,class_index,class_name,xmin,ymin,xmax,ymax,score
1,img/000001.png,0,cat,329,166,342,179,0.8040043115615845
1,img/000001.png,1,dog,375,155,390,170,0.6908698081970215
1,img/000002.png,1,dog,364,149,378,164,0.6787416934967041
```

## Extract Backbone

Load Faster R-CNN model from configuration file `work_dirs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py`, save backbone state dict at `work_dirs/backbone1.pth` and load Faster-RCNN weights from `work_dirs/backbone1.pth`. The third argument is optional. If no weights are provided the default pretrained weights will be loaded.

```bash
$ cv mmdet extract-backbone \
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

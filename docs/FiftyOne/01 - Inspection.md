# FiftyOne

## Inspect object detection annotations

You can inspect an object detection dataset that is annotated in the COCO annotation format with the following command:
The command accepts a path to an imge directory and a COCO annotation file that contains paths to the images relative to the images directory.

```bash
$ cv fiftyone show --images datasets/<name>/train --annotations datasets/<name>/annotations/instances_train20178.json
```

## Inspect object detections

```bash
$ cv fiftyone detections --annotations datasets/<name>/annotations/instances_train.json --images datasets/<name>/train --det detections_train.csv
```
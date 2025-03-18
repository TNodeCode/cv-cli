from mmdet.apis import init_detector, inference_detector
import torch
import os

# Path to config file and checkpoint
config_file = 'mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# Load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = init_detector(config_file, checkpoint_file, device=device)
#print(model)
model = model.backbone

# Set model to evaluation mode
model.eval()

# Create a dummy tensor representing an image batch (N, C, H, W)
dummy_input = torch.randn(1, 3, 224, 224).to("cpu")

# Extract backbone features
#features = model.extract_feat(dummy_input)
features = model(dummy_input)
print("FEATURES", [f.shape for f in features])

import torch
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from cvsdk.inspection.backbones.backbone import BackboneModel

class MobileNetV3Small(BackboneModel):
    def __init__(self):
        super().__init__()
        self.name = "mobilnetv3small"
        self.feature_keys = [f"c{i}" for i in range(1,7)]
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.return_nodes = return_nodes={
            "features.3.add": self.feature_keys[0],
            "features.5.add": self.feature_keys[1],
            "features.6.add": self.feature_keys[2],
            "features.8.add": self.feature_keys[3],
            "features.10.add": self.feature_keys[4],
            "features.11.add": self.feature_keys[5],
        }
        self.feature_extractor = create_feature_extractor(
            self.model,
            self.return_nodes,
        )
        self.gradcam_layers = [self.model.features[i] for i in list(range(-1,-10,-1))]
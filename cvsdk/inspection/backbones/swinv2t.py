import torch
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from cvsdk.inspection.backbones.backbone import BackboneModel

class SwinV2T(BackboneModel):
    def __init__(self, filepath=None):
        super().__init__(filepath=filepath)
        self.name = "swin_v2_t"
        self.feature_keys = [f"f{i}" for i in range(1,10)]
        self.model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)
        self.classifier = self.model.head
        self.return_nodes = return_nodes={
            "features.0": self.feature_keys[0],
            "features.1": self.feature_keys[1],
            "features.2": self.feature_keys[2],
            "features.3": self.feature_keys[3],
            "features.4": self.feature_keys[4],
            "features.5": self.feature_keys[5],
            "features.6": self.feature_keys[6],
            "features.7": self.feature_keys[7],
        }
        self.gradcam_layers = [
            self.model.features[7][-1].mlp[-1],
            self.model.features[7][-1].norm2,
            self.model.features[5][-1].mlp[-1],
            self.model.features[5][-1].norm2,
            self.model.features[3][-1].mlp,
            self.model.features[3][-1].norm2,
            self.model.features[1][-1].mlp,
            self.model.features[1][-1].norm2,
        ]
        self.feature_extractor = create_feature_extractor(
            self.model,
            self.return_nodes,
        )
        self.load_weights()

    def set_classifier(self, classifier):
        self.model.fc = classifier
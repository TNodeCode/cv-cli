import torch
import torchvision.models as models
from torchvision.models import vgg13, VGG13_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from cvsdk.inspection.backbones.backbone import BackboneModel

class VGG13(BackboneModel):
    def __init__(self, filepath=None):
        super().__init__(filepath=filepath)
        self.name = "vgg13"
        self.feature_keys = [f"c{i}" for i in range(1,11)]
        self.model = vgg13(weights=VGG13_Weights.DEFAULT)
        self.classifier = self.model.classifier
        self.classifier_input_size = 25088
        self.return_nodes = return_nodes={
            "features.1": self.feature_keys[0],
            "features.3": self.feature_keys[1],
            "features.6": self.feature_keys[2],
            "features.8": self.feature_keys[3],
            "features.11": self.feature_keys[4],
            "features.13": self.feature_keys[5],
            "features.16": self.feature_keys[6],
            "features.18": self.feature_keys[7],
            "features.21": self.feature_keys[8],
            "features.23": self.feature_keys[9],
        }
        self.feature_extractor = create_feature_extractor(
            self.model,
            self.return_nodes,
        )
        self.load_weights()

    def set_classifier(self, classifier):
        self.model.classifier = classifier
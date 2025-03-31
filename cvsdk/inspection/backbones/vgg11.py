import torch
import torchvision.models as models
from torchvision.models import vgg11, VGG11_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from cvsdk.inspection.backbones.backbone import BackboneModel

class VGG11(BackboneModel):
    def __init__(self, filepath=None):
        super().__init__(filepath=filepath)
        self.name = "vgg11"
        self.feature_keys = [f"c{i}" for i in range(1,9)]
        self.model = vgg11(weights=VGG11_Weights.DEFAULT)
        self.classifier = self.model.classifier
        self.classifier_input_size = 25088
        self.return_nodes = return_nodes={
            "features.1": self.feature_keys[0],
            "features.4": self.feature_keys[1],
            "features.7": self.feature_keys[2],
            "features.9": self.feature_keys[3],
            "features.12": self.feature_keys[4],
            "features.14": self.feature_keys[5],
            "features.17": self.feature_keys[6],
            "features.19": self.feature_keys[7],
        }
        self.feature_extractor = create_feature_extractor(
            self.model,
            self.return_nodes,
        )
        self.load_weights()

    def set_classifier(self, classifier):
        self.model.classifier = classifier
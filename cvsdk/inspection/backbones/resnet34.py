import torch
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from cvsdk.inspection.backbones.backbone import BackboneModel

class ResNet34(BackboneModel):
    def __init__(self, filepath=None):
        super().__init__(filepath=filepath)
        self.name = "resnet34"
        self.feature_keys = [f"f{i}" for i in range(1,5)]
        self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.classifier = self.model.fc
        self.return_nodes = return_nodes={
            "layer1.2.relu_1": self.feature_keys[0],
            "layer2.3.relu_1": self.feature_keys[1],
            "layer3.5.relu_1": self.feature_keys[2],
            "layer4.2.relu_1": self.feature_keys[3],
        }
        self.gradcam_layers = [
            self.model.layer4[-1], 
            self.model.layer3[-2], 
            self.model.layer2[-3], 
            self.model.layer1[-4],            
        ]
        self.feature_extractor = create_feature_extractor(
            self.model,
            self.return_nodes,
        )
        self.load_weights()

    def set_classifier(self, classifier):
        self.model.fc = classifier
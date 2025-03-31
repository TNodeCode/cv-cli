import torch
import torchvision.models as models
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from cvsdk.inspection.backbones.backbone import BackboneModel

class DenseNet121(BackboneModel):
    def __init__(self, filepath=None):
        super().__init__(filepath=filepath)
        self.name = "densenet121"
        self.feature_keys = [f"f{i}" for i in range(1,13)]
        self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.classifier = self.model.classifier
        self.return_nodes = return_nodes={
            "features.denseblock1.denselayer6.relu2": self.feature_keys[0],
            "features.transition1.relu": self.feature_keys[1],
            "features.transition1.pool": self.feature_keys[2],
            "features.denseblock2.denselayer12.relu2": self.feature_keys[3],
            "features.transition2.relu": self.feature_keys[4],
            "features.transition2.pool": self.feature_keys[5],
            "features.denseblock3.denselayer24.relu2": self.feature_keys[6],
            "features.transition3.relu": self.feature_keys[7],
            "features.transition3.pool": self.feature_keys[8],
            "features.transition3.pool": self.feature_keys[9],
            "features.denseblock4.denselayer16.relu2": self.feature_keys[10],
            "relu": self.feature_keys[11],
        }
        self.feature_extractor = create_feature_extractor(
            self.model,
            self.return_nodes,
        )

    def set_classifier(self, classifier):
        self.model.classifier = classifier
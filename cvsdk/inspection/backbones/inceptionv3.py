import torch
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from cvsdk.inspection.backbones.backbone import BackboneModel

class InceptionV3(BackboneModel):
    def __init__(self, filepath=None):
        super().__init__(filepath=filepath)
        self.name = "inceptionv3"
        self.feature_keys = [f"c{i}" for i in range(1,18)]
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.classifier = self.model.fc
        self.return_nodes = return_nodes={
            "Conv2d_2b_3x3.relu": self.feature_keys[0],
            "Conv2d_4a_3x3.relu": self.feature_keys[1],
            "Mixed_5b.branch3x3dbl_3.relu": self.feature_keys[2],
            "Mixed_5c.branch3x3dbl_3.relu": self.feature_keys[3],
            "Mixed_5d.branch3x3dbl_3.relu": self.feature_keys[4],
            "Mixed_6a.branch3x3dbl_3.relu": self.feature_keys[5],
            "Mixed_6b.branch7x7dbl_5.relu": self.feature_keys[6],
            "Mixed_6c.branch7x7dbl_5.relu": self.feature_keys[7],
            "Mixed_6d.branch7x7dbl_5.relu": self.feature_keys[8],
            "Mixed_6e.branch7x7dbl_5.relu": self.feature_keys[9],
            "Mixed_6e.branch_pool.relu": self.feature_keys[10],
            "AuxLogits.conv1.relu": self.feature_keys[11],
            "Mixed_7a.branch7x7x3_4.relu": self.feature_keys[12],
            "Mixed_7b.branch3x3dbl_3b.relu": self.feature_keys[13],
            "Mixed_7c.branch3x3dbl_3b.relu": self.feature_keys[14],
            "Mixed_7c.branch_pool.relu": self.feature_keys[15],
            "Mixed_7c.cat_2": self.feature_keys[16],
        }
        self.feature_extractor = create_feature_extractor(
            self.model,
            self.return_nodes,
        )
        self.load_weights()

    def set_classifier(self, classifier):
        self.model.fc = classifier
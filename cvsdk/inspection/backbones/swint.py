import torch
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from cvsdk.inspection.backbones.backbone import BackboneModel

class SwinT(BackboneModel):
    def __init__(self, filepath=None):
        super().__init__(filepath=filepath)
        self.name = "swin_t"
        self.feature_keys = [f"f{i}" for i in range(1,10)]
        self.model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        self.classifier = self.model.head
        self.return_nodes = return_nodes={
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
        
    def gradcam_reshape_transform(self, tensor, height=7, width=7):
        result = tensor.transpose(2, 3).transpose(1, 2)
        return result


    def set_classifier(self, classifier):
        self.model.head = classifier
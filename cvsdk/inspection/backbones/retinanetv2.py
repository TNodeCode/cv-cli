import torch
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from cvsdk.inspection.backbones.backbone import BackboneModel


class RetinaNetV2(BackboneModel):
    extracted_features = None

    def __init__(self, filepath=None):
        super().__init__(filepath=filepath)
        self.name = "retinanetv2"
        self.feature_keys = [f"f{i}" for i in range(1, 5)]
        self.model = models.detection.retinanet_resnet50_fpn_v2(
            weights=models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
        #self.classifier = self.model.fc
        self.return_nodes = return_nodes = {
            "backbone.body.layer1[2].relu": self.feature_keys[0],
            "backbone.body.layer2[3].relu": self.feature_keys[1],
            "backbone.body.layer3[5].relu": self.feature_keys[2],
            "backbone.body.layer4[2].relu": self.feature_keys[3],
        }
        # self.feature_extractor = create_feature_extractor(
        #    self.model,
        #    self.return_nodes,
        # )
        self.load_weights()

    def set_classifier(self, classifier):
        self.model.fc = classifier

    def get_features(self, input_batch, key):
        """
        Extract feature tensors from model

        :param input_batch: input tensor (image encoded as tensor)
        :param key: feature layer key
        """
        if isinstance(key, int):
            key = self.feature_keys[key]

        def hook_feat_map(mod, inp, out):
            RetinaNetV2.extracted_features = out
        if key == self.feature_keys[0]:
            handle = self.model.backbone.body.layer1[2].relu.register_forward_hook(
                hook_feat_map)
        elif key == self.feature_keys[1]:
            handle = self.model.backbone.body.layer2[3].relu.register_forward_hook(
                hook_feat_map)
        elif key == self.feature_keys[2]:
            handle = self.model.backbone.body.layer3[5].relu.register_forward_hook(
                hook_feat_map)
        elif key == self.feature_keys[3]:
            handle = self.model.backbone.body.layer4[2].relu.register_forward_hook(
                hook_feat_map)
        else:
            return None
        with torch.no_grad():
            out = self.model(input_batch)
            handle.remove()
            return RetinaNetV2.extracted_features

import sys
import torch
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from cvsdk.inspection.backbones.backbone import BackboneModel

class DETR(BackboneModel):
    def __init__(self, filepath=None):
        super().__init__(filepath=filepath)
        self.name = "detr"
        self.feature_keys = [f"f{i}" for i in range(1,10)]
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.return_nodes = return_nodes={
        }
        self.gradcam_layers = [
            #self.model.transformer.decoder.layers[5].norm3,
            #self.model.transformer.decoder.norm,
            self.model.backbone[0].body.layer4[-1],
            self.model.backbone[0].body.layer3[-1],
            self.model.backbone[0].body.layer2[-1],
            self.model.backbone[0].body.layer1[-1],
        ]
        #self.feature_extractor = create_feature_extractor(
        #    self.model,
        #    self.return_nodes,
        #)
        self.load_weights()
        
    def load_state_dict(self, options):
        """
        Load weights of a trained model instance
        """
        options["output_size"] = options["model"]["class_embed.bias"].shape[0]
        options["hidden_layers"] = 0
        options["drop_p"] = 0
        self.replace_classifier(options)
        self.model.load_state_dict(options['model'])
    
    def create_classifier_network(self, options):
        """
        Create a classification layer
        """
        # Creating the Feedforward Classifier
        classifier = torch.nn.Linear(256, 92)        
        return classifier
        
    def gradcam_reshape_transform(self, tensor, height=7, width=7):
        #result = tensor.transpose(2, 3).transpose(1, 2)
        return tensor

    def set_classifier(self, classifier):
        self.model.class_embed = classifier
    
    def __call__(self, x):
        out = self.model(x)['pred_logits'].mean(axis=1)
        return out
import torch
import torchvision.models as models
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from cvsdk.inspection.backbones.backbone import BackboneModel

class ViTB16(BackboneModel):
    def __init__(self, filepath=None):
        super().__init__(filepath=filepath)
        self.name = "vitb16"
        self.feature_keys = ["conv"] + [f"a{i}" for i in range(1,13)]
        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.classifier = self.model.heads.head
        self.classifier_input_size = 2048
        self.return_nodes = return_nodes={
            "conv_proj": self.feature_keys[0],
            "encoder.layers.encoder_layer_0.self_attention": self.feature_keys[1],
            "encoder.layers.encoder_layer_1.self_attention": self.feature_keys[2],
            "encoder.layers.encoder_layer_2.self_attention": self.feature_keys[3],
            "encoder.layers.encoder_layer_3.self_attention": self.feature_keys[4],
            "encoder.layers.encoder_layer_4.self_attention": self.feature_keys[5],
            "encoder.layers.encoder_layer_5.self_attention": self.feature_keys[6],
            "encoder.layers.encoder_layer_6.self_attention": self.feature_keys[7],
            "encoder.layers.encoder_layer_7.self_attention": self.feature_keys[8],
            "encoder.layers.encoder_layer_8.self_attention": self.feature_keys[9],
            "encoder.layers.encoder_layer_9.self_attention": self.feature_keys[10],
            "encoder.layers.encoder_layer_10.self_attention": self.feature_keys[11],
            "encoder.layers.encoder_layer_11.self_attention": self.feature_keys[12],
        }
        self.gradcam_layers = [
            self.model.encoder.layers.encoder_layer_11.ln_1,   
            self.model.encoder.layers.encoder_layer_10.mlp,
            self.model.encoder.layers.encoder_layer_10.ln_2,
            self.model.encoder.layers.encoder_layer_10.ln_1,     
            self.model.encoder.layers.encoder_layer_9.mlp,
            self.model.encoder.layers.encoder_layer_9.ln_2,
            self.model.encoder.layers.encoder_layer_9.ln_1,
            self.model.encoder.layers.encoder_layer_8.mlp,
            self.model.encoder.layers.encoder_layer_8.ln_2,
            self.model.encoder.layers.encoder_layer_8.ln_1,
            self.model.encoder.layers.encoder_layer_7.mlp,
            self.model.encoder.layers.encoder_layer_7.ln_2,
            self.model.encoder.layers.encoder_layer_7.ln_1,
            self.model.encoder.layers.encoder_layer_6.mlp,
            self.model.encoder.layers.encoder_layer_6.ln_2,
            self.model.encoder.layers.encoder_layer_6.ln_1,           
        ]
        self.feature_extractor = create_feature_extractor(
            self.model,
            self.return_nodes,
        )
        self.load_weights()
        
    def gradcam_reshape_transform(self, tensor, height=14, width=14):
        result = tensor[:, 1 :  , :].reshape(tensor.size(0),
            height, width, tensor.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    def set_classifier(self, classifier):
        self.model.heads.head = classifier
        
    def get_features(self, input_batch, key):
        x = super().get_features(input_batch, key)[0]
        if (type(key) == int and key == 0) or key[0] == "c":
            return x
        x = x.permute(0,2,1)
        x = x[:, :, 0:196]
        x = x.reshape(-1, 768, 14, 14)
        return x
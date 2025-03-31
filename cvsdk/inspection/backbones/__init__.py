from cvsdk.inspection.backbones.densenet121 import DenseNet121
from cvsdk.inspection.backbones.densenet161 import DenseNet161
from cvsdk.inspection.backbones.densenet169 import DenseNet169
from cvsdk.inspection.backbones.densenet201 import DenseNet201
from cvsdk.inspection.backbones.fasterrcnn_resnet50_v2 import FasterRCNNResnet50V2
from cvsdk.inspection.backbones.fcos import FCOS
from cvsdk.inspection.backbones.inceptionv3 import InceptionV3
from cvsdk.inspection.backbones.keypoint_rcnn import KeyPointRCNN
from cvsdk.inspection.backbones.maskrcnnv2 import MaskRCNNV2
from cvsdk.inspection.backbones.resnet18 import ResNet18
from cvsdk.inspection.backbones.resnet34 import ResNet34
from cvsdk.inspection.backbones.resnet50 import ResNet50
from cvsdk.inspection.backbones.resnet101 import ResNet101
from cvsdk.inspection.backbones.resnet152 import ResNet152
from cvsdk.inspection.backbones.retinanetv2 import RetinaNetV2
from cvsdk.inspection.backbones.vgg11 import VGG11
from cvsdk.inspection.backbones.vgg13 import VGG13
from cvsdk.inspection.backbones.vgg16 import VGG16
from cvsdk.inspection.backbones.vgg19 import VGG19
from cvsdk.inspection.backbones.vitb16 import ViTB16

models = {
    "densenet121": DenseNet121,
    "densenet161": DenseNet161,
    "densenet169": DenseNet169,
    "densenet201": DenseNet201,
    "fasterrcnn_resnet50_v2": FasterRCNNResnet50V2,
    "fcos": FCOS,
    "inceptionv3": InceptionV3,
    "keypoint_rcnn": KeyPointRCNN,
    "maskrcnnv2": MaskRCNNV2,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "retinanetv2": RetinaNetV2,
    "vgg11": VGG11,
    "vgg13": VGG13,
    "vgg16": VGG16,
    "vgg19": VGG19,
    "vitb16": ViTB16,
}

def load_backbone(name: str, filepath: str = None):
    return models[name](filepath=filepath)

def get_available_backbones():
    return models.keys()
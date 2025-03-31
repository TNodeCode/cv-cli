import torch
from pytorch_grad_cam.feature_factorization.deep_feature_factorization import dff
from pytorch_grad_cam.utils.image import scale_cam_image


def compute_dff(input_tensor: torch.Tensor, n_components: int = 16):
    concepts, explanations = dff(input_tensor, n_components=n_components)
    return concepts, explanations

def scale_explanations(explanations, width, height):
    processed_explanations = []

    for explanation in explanations:
        processed_explanations.append(scale_explanation(explanation, width, height))
        
    return processed_explanations

def scale_explanation(explanation, width, height, channels):
    return scale_cam_image(explanation, (width, height, channels))
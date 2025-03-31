from PIL import Image
from pytorch_grad_cam.utils.image import preprocess_image, deprocess_image
from glob import glob
import requests
import random
import torch
import numpy as np
import os

def get_image_from_url(url):
    """A function that gets a URL of an image, 
    and returns a numpy image and a preprocessed
    torch tensor ready to pass to the model """

    img = np.array(Image.open(requests.get(url, stream=True).raw))
    rgb_img_float = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img_float,
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return img, rgb_img_float, input_tensor

def get_image_from_fs(path: str, resize: tuple[int, int] | None=None):
    """Load a single image form the flie system

    Args:
        path (str): Path to file
        resize (tuple[int, int] | None, optional): (height, width) tuple. Defaults to None.

    Returns:
        _type_: _description_
    """

    img = np.array(Image.open(path))
    if (len(img.shape) == 2):
        img = np.concatenate([img[:,:,np.newaxis]]*3, axis=2)
    if resize is not None:
        y,x,c = img.shape
        startx = random.randint(0, x - resize[0] - 1)
        starty = random.randint(0, y - resize[1] - 1)
        img = img[starty:starty+resize[1], startx:startx+resize[0], :]
    rgb_img_float = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img_float,
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return img, rgb_img_float, input_tensor


def load_image_folder_as_tensor(filenames: list[str], max_images: int=-1, resize: tuple[int, int] | None=None):
    """Load images from a folder as tensor.

    Args:
        filenames (list[str]): List of filenames.
        max_images (int): Maximum number of images. Defaults to -1.
        resize (tuple[int, int], optional): Image eight and width. Defaults to None.

    Returns:
        torch.tensor: Tensor of shape [n_images, channels, height, width]
    """
    images = []
    rgb_img_floats = []
    input_tensors = []
    
    for i, filename in enumerate(filenames):
        if max_images > 0 and i >= max_images:
            break
        img, rgb_img_float, input_tensor = get_image_from_fs(
            filename,
            resize=resize,
        )
        images.append(img)
        rgb_img_floats.append(rgb_img_float)
        input_tensors.append(input_tensor)

    return torch.vstack(input_tensors)
import pytest
from glob import glob
import numpy as np
import torch
from cvsdk.inspection.image_loader import load_image_folder_as_tensor, get_image_from_fs

def test_load_image_folder_as_tensor():
    """Test loading images from directroy as tensor.
    """
    expected_images = 49
    expected_channels = 3
    expected_height = 512
    expected_width = 512
    filenames = glob("data/7s/train2017/*.jpg")
    x = load_image_folder_as_tensor(filenames=filenames)
    assert x.shape[0] == expected_images, f"{expected_images} images should be loaded, but was {x.shape[0]}"
    assert x.shape[1] == expected_channels, f"Images should have {expected_channels} channels, but have {x.shape[1]}"
    assert x.shape[2] == expected_height, f"Images should have height {expected_height}, but was {x.shape[2]}"
    assert x.shape[3] == expected_width, f"Images should have width{expected_width}, but was {x.shape[3]}"

def test_load_image_folder_as_tensor_limit_images():
    """Test loading images from directroy as tensor with maximum number of images.
    """
    expected_images = 10
    expected_channels = 3
    expected_height = 512
    expected_width = 512
    filenames = glob("data/7s/train2017/*.jpg")
    x = load_image_folder_as_tensor(filenames=filenames, max_images=expected_images)
    assert x.shape[0] == expected_images, f"{expected_images} images should be loaded, but was {x.shape[0]}"
    assert x.shape[1] == expected_channels, f"Images should have {expected_channels} channels, but have {x.shape[1]}"
    assert x.shape[2] == expected_height, f"Images should have height {expected_height}, but was {x.shape[2]}"
    assert x.shape[3] == expected_width, f"Images should have width{expected_width}, but was {x.shape[3]}"

def test_load_image_folder_as_tensor_resize_images():
    """Test loading images from directroy as tensor with image resizing.
    """
    expected_images = 49
    expected_channels = 3
    expected_height = 224
    expected_width = 224
    filenames = glob("data/7s/train2017/*.jpg")
    x = load_image_folder_as_tensor(filenames=filenames, resize=(224,224))
    assert x.shape[0] == expected_images, f"{expected_images} images should be loaded, but was {x.shape[0]}"
    assert x.shape[1] == expected_channels, f"Images should have {expected_channels} channels, but have {x.shape[1]}"
    assert x.shape[2] == expected_height, f"Images should have height {expected_height}, but was {x.shape[2]}"
    assert x.shape[3] == expected_width, f"Images should have width{expected_width}, but was {x.shape[3]}"

def test_load_image():
    """Test loading images from directroy as tensor.
    """
    img, rgb_img_float, input_tensor = get_image_from_fs("data/7s/train2017/1.jpg")
    
    expected_img_type = np.uint8
    expected_rgb_type = np.float32
    expected_tensor_type = torch.float32
    expected_img_shape = (512, 512, 3)
    expected_rgb_img_shape = (512, 512, 3)
    expected_input_tensor_shape = (1, 3, 512, 512)
    
    assert img.dtype == expected_img_type, f"img should be {expected_img_type}, but was {img.dtype}"
    assert rgb_img_float.dtype == expected_rgb_type, f"rgb_img_float should be {expected_rgb_type}, but was {rgb_img_float.dtype}"
    assert input_tensor.dtype == expected_tensor_type, f"input_tensor should be {expected_tensor_type}, but was {input_tensor.dtype}"
    assert img.shape == expected_img_shape, f"img should be {expected_img_shape}, but was {img.shape}"
    assert rgb_img_float.shape == expected_rgb_img_shape, f"rgb_img_float should be {expected_rgb_img_shape}, but was {rgb_img_float.shape}"
    assert input_tensor.shape == expected_input_tensor_shape, f"input_tensor should be {expected_input_tensor_shape}, but was {input_tensor.shape}"

    
    
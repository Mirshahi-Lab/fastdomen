import cupy as cp
import numpy as np


def transform_to_hu(image: cp.ndarray, slope: float, intercept: float) -> cp.ndarray:
    """
    A function to transform a ct scan image into hounsfield units \n
    :param image: a numpy array containing the raw ct scan
    :param slope: the slope rescaling factor from the dicom file (0 if already in hounsfield units)
    :param intercept: the intercept value from the dicom file (depends on the machine)
    :return: a copy of the numpy array converted into hounsfield units
    """
    hu_image = cp.multiply(image, float(slope)) + float(intercept)
    return hu_image


def window_image(image: cp.ndarray, window_center: int, window_width: int):
    """
    A function to window the hounsfield units of the ct scan \n
    :param image: a numpy array containing the hounsfield ct scan
    :param window_center: hounsfield window center
    :param window_width: hounsfield window width
    :return: a windowed copy of 'image' parameter
    """
    # Get the min/max hounsfield units for the dicom image
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2

    new_image = cp.clip(image, img_min, img_max)
    return new_image


def normalize_pytorch(image: cp.ndarray, max_pixel_value: float, mean: float=0.445, std: float=0.269):
    image = (image - mean * max_pixel_value) / (std * max_pixel_value)
    return image


def normalize_zero_one(image, eps=1e-8):
    image = image.astype(np.float32)
    ret = (image - cp.min(image))
    ret /= (cp.max(image) - cp.min(image) + eps)
    return ret
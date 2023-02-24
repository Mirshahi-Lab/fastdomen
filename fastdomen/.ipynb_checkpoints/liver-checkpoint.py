import cupy as cp
from cupyx.scipy import ndimage
import torch
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter

from fastdomen.unext.utils import batch_predict
from fastdomen.imaging.dicomseries import DicomSeries
from fastdomen.imaging.utils import normalize_pytorch


def measure_hounsfields(image: cp.ndarray, mask: cp.ndarray):
    """
    A function to measure the average hounsfield value of a segmentation\n
    :param image: the CT image
    :param mask: the segmentation
    :return: the average hounsfield value of the segmentation
    """
    # Get the average hounsfield value of the segmentation
    hounsfield = image[mask == 1]
    return round(float(hounsfield.mean()), 1), round(float(hounsfield.std()), 2)


def measure_organ_volume(ds, preds):
    """
    Find the organ volume in cm3
    :param ds: The dicom series
    :param preds: The 3d segmentation
    """
    total_pixels = preds.sum()
    h, w, t = ds.spacing
    h /= 10
    w /= 10
    t /= 10
    return round(float(total_pixels * h * w * t), 0)


def measure_liver_hu(ds: DicomSeries, model, model_weights, output_dir):
    """
    measure the hounsfield units of the liver
    :param ds: the dicom series instance
    :param model: The Unext model
    :param model_weights: the model weights for the liver segmentation
    :param output_dir: the directory to save the output images
    :return: a dictionary containing the liver data
    """
    liver_data = {}
    model.load_state_dict(torch.load(model_weights))
    images = ds.read_dicom_series(30, 150)
    # Preprocess, predict, and postprocess the segmentations
    torch_images = preprocess(images)
    preds = batch_predict(model, torch_images)
    preds = postprocess(preds)
    # Find the volume-wide measures
    liver_data['liver_volume_cm3'] = measure_organ_volume(ds, preds)
    liver_data['liver_volume_mean_hu'] = measure_hounsfields(images, preds)[0]
    # Find the center point of the segmentation
    pred_center = find_seg_center(preds)
    slice_idxs = find_indices(preds, ds.spacing, pred_center)
    pixel_radius = ds.calculate_pixel_radius()
    # Find the slice by slice information
    slice_data = measure_slices(images, preds, slice_idxs, pixel_radius, 'liver', output_dir, ds.filename)
    liver_data.update(slice_data)
    torch.cuda.empty_cache()
    return liver_data


def measure_spleen_hu(ds: DicomSeries, model, model_weights, output_dir):
    """
    measure the hounsfield units of the spleen
    :param ds: the dicom series instance
    :param model: The Unext model
    :param model_weights: the model weights for the spleen segmentation
    :param output_dir: the directory to save the output images
    :return: a dictionary containing the spleen data
    """
    spleen_data = {}
    model.load_state_dict(torch.load(model_weights))
    images = ds.read_dicom_series(30, 150)
    # Preprocess, predict, and postprocess the segmentations
    torch_images = preprocess(images)
    preds = batch_predict(model, torch_images)
    preds = postprocess(preds)
    
    spleen_data['spleen_volume_cm3'] = measure_organ_volume(ds, preds)
    spleen_data['spleen_volume_mean_hu'] = measure_hounsfields(images, preds)[0]
    pred_center = find_seg_center(preds)
    slice_idxs = {'center': int(pred_center)}
    pixel_radius = ds.calculate_pixel_radius()
    slice_data = measure_slices(images, preds, slice_idxs, pixel_radius, 'spleen', output_dir, ds.filename)
    spleen_data.update(slice_data)
    torch.cuda.empty_cache()
    return spleen_data

def preprocess(images: cp.ndarray):
    """
    preprocess the images for segmentation by UNext
    """
    torch_images = normalize_pytorch(images.copy(), images.max())
    torch_images = cp.expand_dims(torch_images, axis=1)
    torch_images = torch.as_tensor(torch_images, device='cuda')
    return torch_images


def postprocess(preds, threshold=100):
    """
    postprocess the predictions for further analysis
    """
    preds = torch.round(torch.squeeze(preds))
    preds = cp.asarray(preds)
    liver_vals = preds.sum(axis=(1, 2))
    above_threshold = cp.where(liver_vals > threshold)[0]
    start, end = longest_consecutive_seg(above_threshold)
    preds[:start, ...] = 0
    preds[end:, ...] - 0 
    return preds


def longest_consecutive_seg(numbers):
    """
    A function to find the longest consecutive cut of liver from a segmentation\n
    :param numbers: a list of indices containing liver segmentations
    :return: the starting and ending index of the longest consecutive cut
    """
    idx = max(
        (
            list(map(itemgetter(0), g))
            for i, g in groupby(enumerate(cp.diff(numbers) == 1), itemgetter(1))
            if i
        ),
        key=len
    )
    return numbers[idx[0]], numbers[idx[-1] + 1]


def erode(preds: cp.ndarray):
    """
    Use 3d binary erosion on a 3d array
    """
    # Create the starting kernel
    kernel = cp.zeros([3, 3, 3])
    kernel[1, 1, 0] = 1
    kernel[0, 1, 1] = 1
    kernel[1, 0, 1] = 1
    kernel[1, 1, 1] = 1
    kernel[1, 2, 1] = 1
    kernel[2, 1, 1] = 1
    kernel[1, 1, 2] = 1
    # Erode the mask down to a center point (y, x, z)
    eroded = ndimage.binary_erosion(preds, structure=kernel).astype(preds.dtype)
    return eroded


def find_seg_center(preds: cp.ndarray, erosion_limit=100):
    """
    Use the erode function to find the center point of the segmentation
    """
    eroded = preds.copy()
    erode_last = eroded
    
    while eroded.sum() > erosion_limit:
        erode_last = eroded.copy()
        eroded = erode(eroded)
    
    if eroded.sum() == 0:
        eroded = erode_last
    
    center = cp.argwhere(eroded)
    center = center[len(center) // 2][0]
    return center


def find_indices(preds, spacing, center_slice, distance=20):
    """
    Find the superior and inferior slices a specified distance (mm) from the center point of the liver segmentation
    """
    slice_thickness = spacing[-1]
    num_slices = distance // slice_thickness
    superior = center_slice - num_slices
    inferior = center_slice + num_slices
    if superior < 0:
        superior = 0
    if inferior >= preds.shape[0]:
        inferior = preds.shape[0] - 1
    return {'superior': int(superior), 'center': int(center_slice), 'inferior': int(inferior)}


def measure_slices(images, preds, slice_indices, pixel_radius, organ, output_dir, filename):
    measures = {}
    for key, v in slice_indices.items():
        pred = preds[v]
        image = images[v]
        slice_mean, _ = measure_hounsfields(image, pred)
        measures[f'{organ}_{key}_slice_mean_hu'] = slice_mean
        slice_center = find_slice_center(pred)
        roi_mask = create_roi_mask(pred, slice_center, pixel_radius, organ)
        roi_mean, roi_std = measure_hounsfields(image, roi_mask)
        measures[f'{organ}_{key}_roi_mean_hu'] = roi_mean
        measures[f'{organ}_{key}_roi_std_hu'] = roi_std
        plot_roi_overlay(image, roi_mask, key, organ, output_dir, filename)
    return measures


def create_roi_mask(pred, mask_center, pixel_radius, organ):
    h, w = pred.shape
    roi_mask = cp.zeros_like(pred)
    if organ == 'liver':
        roi_centers = find_roi_centers(pred, mask_center)
        for center in roi_centers:
            roi = draw_roi(h, w, center, pixel_radius)
            roi_mask = roi_mask + roi
    else:
        roi = draw_roi(h, w, mask_center, pixel_radius)
        roi_mask = roi_mask + roi
    roi_mask[pred == 0] = 0
    return roi_mask
            
            

def find_roi_centers(mask, mask_center, alpha=0.55):
    # Get the center coordinates
    y_center = int(mask_center[0])
    x_center = int(mask_center[1])
    antialpha = 1 - alpha
    # Get the edges
    left_edge = cp.where(mask[y_center, :] == 1)[0].min()
    top_edge = cp.where(mask[:, x_center] == 1)[0].min()
    bottom_edge = cp.where(mask[:, x_center] == 1)[0].max()
    # Calculate the centers
    left_center = (y_center, int(alpha * left_edge + antialpha * x_center))
    top_center = (int(alpha * top_edge + antialpha * y_center), x_center)
    bottom_center = (int(alpha * y_center + antialpha * bottom_edge), x_center)

    return left_center, top_center, bottom_center
    

def find_slice_center(pred):
    center_mask = pred.copy()
    erode_last = center_mask
    
    while center_mask.sum() > 1:
        erode_last = center_mask.copy()
        center_mask = ndimage.binary_erosion(center_mask)
        
    if center_mask.sum() == 0:
        center_mask = erode_last
    center_point = cp.argwhere(center_mask)
    
    if len(center_point.shape) > 1:
        center_point = center_point[0]
    return center_point


def draw_roi(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = cp.ogrid[:h, :w]
    dist_from_center = cp.sqrt((X - center[1])**2 + (Y-center[0])**2)

    mask = dist_from_center <= radius
    return mask.astype(cp.int)


def plot_roi_overlay(image, roi_mask, slice_name, organ, output_dir, filename):
    image[roi_mask == 1] = 255
    plt.imshow(image.get())
    outfile = f'{output_dir}/{filename}_{organ}_{slice_name}_roi.jpg'
    plt.savefig(outfile, dpi=150)
    plt.close()
    
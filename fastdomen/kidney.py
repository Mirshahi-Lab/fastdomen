import numpy as np
import cupy as cp
from cupyx.scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch
import SimpleITK as sitk

from fastdomen.imaging.dicomseries import DicomSeries
# from fastdomen.unext.utils import batch_predict
from fastdomen.imaging.utils import normalize_pytorch, normalize_zero_one
# from fastdomen.liver import longest_consecutive_seg

#### PRE - TOTAL SEGMENTATOR CODE #####
# def measure_kidney(ds: DicomSeries, model, model_weights, output_dir):
#     kidney_data = {}
#     images = ds.read_dicom_series(50, 400)
#     torch_images = preprocess(images)
#     model.load_state_dict(torch.load(model_weights))
#     preds = batch_predict(model, torch_images)
#     left_k, right_k = postprocess(preds, threshold=50)
#     preds = cp.concatenate((left_k, right_k), axis=2)
#     voxel_volumes = measure_direct_volume(left_k, right_k, ds.spacing)
#     kidney_data.update(voxel_volumes)
#     estimated_volume = measure_estimated_volume(left_k, right_k, ds.spacing)
#     kidney_data.update(estimated_volume)
#     return kidney_data


# def preprocess(images):
#     images = 255 * normalize_zero_one(images)
#     images = normalize_pytorch(images, images.max(), 0.445, 0.269)
#     images = torch.as_tensor(images, dtype=torch.float32, device='cuda')
#     images = images.unsqueeze(1)
#     return images


# def postprocess(preds, threshold=100):
#     """
#     postprocess the predictions for further analysis
#     """
#     preds = torch.round(torch.squeeze(preds))
#     preds = cp.asarray(preds)
#     left_k = preds[:, :, :256]
#     right_k = preds[:, :, 256:]
#     try:
#         left_k = process_kidney(left_k, threshold)
#     except ValueError:
#         print('Left Kidney was not found in segmentation')
#     try:
#         right_k = process_kidney(right_k, threshold)
#     except ValueError:
#         print('Right Kidney was not found in segmentation')
#     return left_k, right_k


# def process_kidney(kidney_mask, threshold):
#     # First, only keep the largest connected segments
#     new_mask = cp.zeros_like(kidney_mask)
#     for i in range(kidney_mask.shape[0]):
#         if kidney_mask[i].sum() > 0:
#             new_mask[i] = largest_connected_seg(kidney_mask[i])
#     kidney_vals = new_mask.sum(axis=(1,2))
#     above_thresh = cp.where(kidney_vals > threshold)[0]
#     start, end = longest_consecutive_seg(above_thresh)
#     new_mask[:start, ...] = 0
#     new_mask[end:, ...] = 0
#     return new_mask


def largest_connected_seg(keep_mask):
    labels, _ = ndimage.label(keep_mask)
    unique, counts = cp.unique(labels, return_counts=True)
    unique, counts = unique[1:], counts[1:]
    largest_label = int(unique[int(cp.argmax(counts))])
    keep_mask[labels != largest_label] = 0
    return keep_mask

def measure_direct_volume(left_k, right_k, spacing):
    volumes = {}
    l, w, h = spacing
    # calculate the volume in cm^3
    voxel_volume = cp.float(l * w * h) // 1000
    left_volume = left_k.sum() * voxel_volume
    right_volume = right_k.sum() * voxel_volume
    total_volume = left_volume + right_volume
    volumes['left_voxel_volume'] = round(float(left_volume))
    volumes['right_voxel_volume'] = round(float(right_volume))
    volumes['total_voxel_volume'] = round(float(total_volume))
    return volumes


def max_transverse_axes(arr, spacing):
    x_space, y_space = spacing
    arr = largest_connected_seg(arr)
    img = sitk.GetImageFromArray(arr.get().astype(int))
    filter_label = sitk.LabelShapeStatisticsImageFilter()
    filter_label = sitk.LabelShapeStatisticsImageFilter()
    filter_label.SetComputeFeretDiameter(True)
    filter_label.Execute(img)
    # we have to get a bit smarter for the principal moments
    pc1_x, pc1_y, pc2_x, pc2_y = filter_label.GetPrincipalAxes(1)

    # get the center of mass
    com_y, com_x = filter_label.GetCentroid(1)
    com_y = com_y * y_space
    com_x = com_x * x_space
    # print(com_y, com_x)
    # now trace the distance from the centroid to the edge along the principal axes
    v_x, v_y = np.where(arr.get().astype(int))
    v_x = v_x * x_space
    v_y = v_y * y_space
    # print(v_x, v_y)
    # convert these positions to a vector from the centroid
    v_pts = np.array((v_x - com_x, v_y - com_y)).T

    # project along the first principal component
    distances_pc1 = np.dot(v_pts, np.array((pc1_x, pc1_y)))
    # get the extent
    dmax_1 = distances_pc1.max()
    dmin_1 = distances_pc1.min()
    
    # project along the second principal component
    distances_pc2 = np.dot(v_pts, np.array((pc2_x, pc2_y)))

    # get the extent
    dmax_2 = distances_pc2.max()
    dmin_2 = distances_pc2.min()
    dist1 = dmax_1 - dmin_1
    dist2 = dmax_2 - dmin_2
    major_dist = max(dist1, dist2)
    minor_dist = min(dist1, dist2)
    return round(major_dist, 1), round(minor_dist, 1)


# Note: This is does not work properly for saggital and coronal directions
def find_max_diameter(mask, spacing):
    maj_max = 0
    min_max = 0
    index = 0
    for i in range(mask.shape[1]):
        im = mask[:, :, i]
        if im.sum() > 0:
            tmp_max, tmp_min = max_transverse_axes(im, spacing)
            if tmp_max > maj_max:
                maj_max = tmp_max
                min_max = tmp_min
                index = i
    return round(maj_max, 1), round(min_max, 1), index


def estimate_length(mask, spacing):
    # Get all slices where the mask exists
    l, w, h = spacing
    z_locs = cp.where(mask)[-1]
    min_z = z_locs.min()
    max_z = z_locs.max()
    # Extract top and bottom slices
    min_slice = mask[:, :, min_z]
    max_slice = mask[:, :, max_z]
    # Convert indices into mm for euclidean space
    min_z = min_z * h
    max_z = max_z * h
    # distance = max_z - min_z
    # Find center points of each segmentation
    min_y, min_x = ndimage.center_of_mass(min_slice)
    max_y, max_x = ndimage.center_of_mass(max_slice)
    # Convert to euclidean location
    min_y, max_y = min_y*l, max_y*l
    min_x, max_x = min_x*w, max_x*w
    min_arr = cp.asarray([min_z, min_y, min_x])
    max_arr = cp.asarray([max_z, max_y, max_x])
    distance = cp.linalg.norm(max_arr - min_arr)
    return distance.round(1)


def spheroid_kv(axial_pcs, length):
    width, depth, _ = axial_pcs
    return ((np.pi / 6) * (width * depth * length) / 1000).round(1)


def measure_estimated_volume(organ_seg, organ_map, spacing):
    volumes = {}
    l, w, h = spacing
    right_k = cp.where(organ_seg == organ_map['kidney_right'], 1, 0)
    left_k = cp.where(organ_seg == organ_map['kidney_left'], 1, 0)
    r_ax_axes = find_max_diameter(right_k, [w, l])
    r_avg_length = estimate_length(right_k, spacing)
    l_ax_axes = find_max_diameter(left_k, [w, l])
    l_avg_length = estimate_length(left_k, spacing)
    rkv = spheroid_kv(r_ax_axes, r_avg_length)
    lkv = spheroid_kv(l_ax_axes, l_avg_length)
    tkv = rkv + lkv
    volumes['left_spheroid_volume'] = round(float(lkv), 2)
    volumes['right_spheroid_volume'] = round(float(rkv), 2)
    volumes['total_spheroid_volume'] = round(float(tkv), 2)
    return volumes
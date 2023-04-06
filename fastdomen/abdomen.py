import cupy as cp
from cupyx.scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage import filters, morphology
from skimage.measure import label, find_contours, regionprops

from fastdomen.imaging.utils import normalize_zero_one, normalize_pytorch
from fastdomen.unext.utils import batch_predict

def measure_abdomen(ds, model, locations, model_weights, output_dir):
    abdomen_data = {}
    # Find the start and end of the abdomen
    start, end, logical_order = find_start_and_end(locations)
    abdomen_data['logical_vert_order'] = logical_order
    # Create the necessary arrays that are needed for measurements
    body_array, torch_array, waist_mask = create_needed_arrays(ds, start, end)
    vert_info = adjust_vert_indices(locations)
    model.load_state_dict(torch.load(model_weights), strict=False)
    preds = postprocess(batch_predict(model, torch_array))
    torch.cuda.empty_cache()
    cm_spacing = [x / 10 for x in ds.spacing]
    abd_areas, subq_pixels, subq_area, visc_pixels, visc_area = sum_fat_pixels(body_array, preds, cm_spacing)
    waist_circs, waist_areas = get_waist_data(waist_mask.get(), cm_spacing)
    vert_data = measure_vert_slices(vert_info, subq_pixels, subq_area, visc_pixels, visc_area, waist_circs, waist_areas, abd_areas)
    abdomen_data.update(vert_data)
    for i in range(body_array.shape[0]):
        original_idx = start + i
        abdomen_data[f'slice_{original_idx}'] = get_slice_vals(i, subq_pixels, subq_area, visc_pixels, visc_area, waist_circs, waist_areas, abd_areas)
        if i == vert_info['adj_L3']:
            plot_contours(body_array[i].get(), waist_mask[i].get(), output_dir)
    return abdomen_data

def create_needed_arrays(ds, start, end):
    # Array to measure fat
    body_array = ds.read_dicom_series(100, 1500)[start:end+1]
    # Array to segment abdominal wall
    body_norm = 255*normalize_zero_one(body_array.copy())
    # Array to measure body circumference
    waist_mask = cp.zeros_like(body_norm)
    for i in range(body_norm.shape[0]):
        waist_mask[i] = remove_table(body_norm[i])
    body_array[waist_mask != 1] = body_array.min()
    body_norm[waist_mask != 1] = body_norm.min()
    body_norm = normalize_pytorch(body_norm, body_norm.max())
    body_norm = cp.expand_dims(body_norm, axis=1)
    body_norm = torch.as_tensor(body_norm, dtype=torch.float32, device='cuda')
    return body_array, body_norm, waist_mask
    
    
def remove_table(body_slice):
    tgray = body_slice > filters.threshold_otsu(body_slice.copy().get())
    keep_mask = morphology.remove_small_objects(tgray.get(), min_size=463)
    keep_mask = cp.asarray(morphology.remove_small_holes(keep_mask), dtype=cp.int)
    labels, _ = ndimage.label(keep_mask)
    unique, counts = cp.unique(labels, return_counts=True)
    unique, counts = unique[1:], counts[1:]
    largest_label = int(unique[int(cp.argmax(counts))])
    keep_mask[labels != largest_label] = 0
    keep_mask = ndimage.binary_fill_holes(keep_mask)
    return keep_mask


def find_start_and_end(locations):
    l1_loc = locations['L1']['slice_idx']
    l3_loc = locations['L3']['slice_idx']
    l5_loc = locations['L5']['slice_idx']
    start = min(l1_loc, l3_loc, l5_loc)
    end = max(l1_loc, l3_loc, l5_loc)
    if start != l1_loc or end != l5_loc:
        logical_vert_order = False
    else:
        logical_vert_order = True
    return start, end, logical_vert_order


def adjust_vert_indices(locations):
    vert_info = {}
    # Save the original locs
    l1_loc = locations['L1']['slice_idx']
    l3_loc = locations['L3']['slice_idx']
    l5_loc = locations['L5']['slice_idx']
    vert_info['og_L1'] = l1_loc
    vert_info['og_L3'] = l3_loc
    vert_info['og_L5'] = l5_loc
    # Save the adjusted locs
    new_l3 = l3_loc - l1_loc
    new_l5 = l5_loc - l1_loc
    new_l1 = 0
    vert_info['adj_L1'] = new_l1
    vert_info['adj_L3'] = new_l3
    vert_info['adj_L5'] = new_l5
    return vert_info
    

def postprocess(pred):
    return cp.asarray(torch.round(torch.squeeze(pred)))


def sum_fat_pixels(body_array, preds, spacing):
    visc = body_array.copy()
    subq = body_array.copy()
    visc[preds == 0] = visc.min()
    subq[preds == 1] = subq.min()
    abd_area = preds.sum(axis=(1, 2)) * spacing[0] * spacing[1]
    # [-190, -30] is the standard hounsfield window for fat
    visc_pixels = ((visc >= -190) & (visc <= -30)).sum(axis=(1, 2))
    subq_pixels = ((subq >= -190) & (subq <= -30)).sum(axis=(1, 2))
    visc_area = visc_pixels * spacing[0] * spacing[1]
    subq_area = subq_pixels * spacing[0] * spacing[1]
    return abd_area, subq_pixels, subq_area, visc_pixels, visc_area
    

def get_waist_data(waist_mask, spacing):
    h, w, _ = spacing
    circs = []
    areas = []
    for i in range(waist_mask.shape[0]):
        labels = label(waist_mask[i])
        regions = regionprops(labels)
        waist = max(regions, key=lambda x: x.area)
        waist_circ = round(waist.perimeter * w, 2)
        waist_area = round(waist.area * w * h, 2)
        circs.append(waist_circ)
        areas.append(waist_area)
    return circs, areas


def measure_vert_slices(vert_info, subq_pixels, subq_area, visc_pixels, visc_area, waist_circs, waist_areas, abd_areas):
    vert_data = {}
    for vert in ['L1', 'L3', 'L5']:
        idx = vert_info[f'adj_{vert}']
        vert_data[f'{vert}_measures'] = get_slice_vals(idx, subq_pixels, subq_area, visc_pixels, visc_area, waist_circs, waist_areas, abd_areas)
    return vert_data


def get_slice_vals(i, subq_pixels, subq_area, visc_pixels, visc_area, waist_circs, waist_areas, abd_areas):
    slice_data = {}
    slice_data['body_circ'] = float(waist_circs[i])
    slice_data['body_area'] = round(float(waist_areas[i]), 2)
    slice_data['abd_area'] = round(float(abd_areas[i]), 2)
    slice_data['subq_pixels'] = float(subq_pixels[i])
    slice_data['subq_area'] = round(float(subq_area[i]), 2)
    slice_data['visc_pixels'] = float(visc_pixels[i])
    slice_data['visc_area'] = round(float(visc_area[i]), 2)
    return slice_data


def plot_contours(original_image, body_mask, out_dir):
    """
    A function to plot the contours of the waist circumference around original image
    :param original_image: the original L3 slice
    :param body_mask: the body mask
    :param output_path: the path to save the plot
    :return: the plot of the contours
    """
    contours = find_contours(body_mask, 0.5)
    fig, ax = plt.subplots()
    ax.imshow(original_image, cmap=plt.cm.gray)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=3)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(f'{out_dir}/L3_waist_contour.jpg', dpi=150)
    plt.close()
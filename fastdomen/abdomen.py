import cupy as cp
from cupyx.scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import torch
from cucim.skimage import filters, morphology
from skimage.measure import label, find_contours, regionprops

from fastdomen.imaging.utils import normalize_zero_one, normalize_pytorch
# from fastdomen.unext.utils import batch_predict


def measure_verts(original, tissue_seg, tissue_dict, vert_dict, spacing):
    """
    A function to measure the body circ/area, subq/visceral fat and skeletal muscle at each vertebrae
    """
    out_dict = {}
    l, w, _ = spacing
    for vert in ['vertebrae_L1', 'vertebrae_L3', 'vertebrae_L5']:
        loc = vert_dict[vert]
        # print(f'{loc=}')
        if loc is not None:
            body_slice = remove_table(original[:, :, loc])
            waist_circ, waist_area = get_waist_data(body_slice, l, w)
            fat_slice = tissue_seg[:, :, loc]
            tissue_out = {}
            tissue_out['slice_idx'] = loc
            tissue_out['body_circ'] = float(round(waist_circ, 2))
            tissue_out['body_area'] = float(round(waist_area, 2))
            for tissue, idx in tissue_dict.items():
                # print(f'{tissue=}, {idx=}')
                tissue_msk = cp.where(fat_slice == idx, 1, 0)
                tissue_out[f'{tissue}_area'] = float(cp.round(tissue_msk.sum() * l * w, 2))
            
        else:
            tissue_out = {}
            tissue_out['slice_idx'] = loc
            tissue_out['body_circ'] = None
            tissue_out['body_area'] = None
            for tissue, idx in tissue_dict.items():
                tissue_out[f'{tissue}_area'] = None
        
        out_dict[vert] = tissue_out
    return out_dict


def total_torso_measure(tissue_seg, tissue_dict, vert_dict, spacing):
    """
    Measure the total torso area from the L1 to L5 (or L3) vertebra as well as the total volume
    """
    # Get the l, w, and h for each voxel in mm
    l, w, h = spacing
    # vert_dict.pop('vertebrae_T12')
    vert_dict = {k: v for k, v in vert_dict.items() if v is not None and k != 'vertebrae_T12'}
    # print(f'torso {vert_dict=}')
    # Only do this if more than one vertebrae is in the image
    tissue_out = {}
    if len(vert_dict) > 1:
        min_vert = min(vert_dict, key=vert_dict.get)
        max_vert = max(vert_dict, key=vert_dict.get)
        start_idx = vert_dict[min_vert]
        stop_idx = vert_dict[max_vert] + 1
        
        for tissue, idx in tissue_dict.items():
            tissue_msk = cp.where(tissue_seg == idx, 1, 0)
            tissue_out[f'{tissue}_total_volume'] = float(cp.round(tissue_msk.sum() * l * w * h, 2))
            tissue_msk = tissue_msk[:, :, start_idx:stop_idx]
            tissue_out[f'{tissue}_torso_volume'] = float(cp.round(tissue_msk.sum() * l * w * h, 2))
        tissue_out['min_vertebrae'] = min_vert
        tissue_out['max_vertebrae'] = max_vert
    else:
        for tissue, idx in tissue_dict.items():
            tissue_msk = cp.where(tissue_seg == idx, 1, 0)
            tissue_out[f'{tissue}_total_volume'] = float(cp.round(tissue_msk.sum() * l * w * h, 2))
            tissue_out[f'{tissue}_torso_volume'] = None
        tissue_out['min_vertebrae'] = None
        tissue_out['max_vertebrae'] = None
    return tissue_out

###### PRE - TOTAL SEGMENTATOR CODE #########
# def measure_abdomen(ds, model, locations, model_weights, output_dir):
#     abdomen_data = {}
#     # Find the start and end of the abdomen
#     start, end, logical_order = find_start_and_end(locations)
#     abdomen_data['logical_vert_order'] = logical_order
#     # Create the necessary arrays that are needed for measurements
#     body_array, torch_array, waist_mask = create_needed_arrays(ds, start, end)
#     vert_info = adjust_vert_indices(locations)
#     model.load_state_dict(torch.load(model_weights), strict=False)
#     preds = postprocess(batch_predict(model, torch_array))
#     torch.cuda.empty_cache()
#     cm_spacing = [x / 10 for x in ds.spacing]
#     abd_areas, subq_pixels, subq_area, visc_pixels, visc_area = sum_fat_pixels(body_array, preds, cm_spacing)
#     waist_circs, waist_areas = get_waist_data(waist_mask.get(), cm_spacing)
#     vert_data = measure_vert_slices(vert_info, subq_pixels, subq_area, visc_pixels, visc_area, waist_circs, waist_areas, abd_areas)
#     abdomen_data.update(vert_data)
#     for i in range(body_array.shape[0]):
#         original_idx = start + i
#         abdomen_data[f'slice_{original_idx}'] = get_slice_vals(i, subq_pixels, subq_area, visc_pixels, visc_area, waist_circs, waist_areas, abd_areas)
#         if i == vert_info['adj_L3']:
#             plot_contours(body_array[i].get(), waist_mask[i].get(), output_dir)
#     return abdomen_data

# def create_needed_arrays(ds, start, end):
#     # Array to measure fat
#     body_array = ds.read_dicom_series(100, 1500)[start:end+1]
#     # Array to segment abdominal wall
#     body_norm = 255*normalize_zero_one(body_array.copy())
#     # Array to measure body circumference
#     waist_mask = cp.zeros_like(body_norm)
#     for i in range(body_norm.shape[0]):
#         waist_mask[i] = remove_table(body_norm[i])
#     body_array[waist_mask != 1] = body_array.min()
#     body_norm[waist_mask != 1] = body_norm.min()
#     body_norm = normalize_pytorch(body_norm, body_norm.max())
#     body_norm = cp.expand_dims(body_norm, axis=1)
#     body_norm = torch.as_tensor(body_norm, dtype=torch.float32, device='cuda')
#     return body_array, body_norm, waist_mask
    
    
def remove_table(body_slice):
    # tgray = body_slice > filters.threshold_otsu(body_slice.copy().get())
    tgray = body_slice > filters.threshold_otsu(body_slice)
    keep_mask = morphology.remove_small_objects(tgray, min_size=463)
    keep_mask = morphology.remove_small_holes(keep_mask)
    labels, _ = ndimage.label(keep_mask)
    unique, counts = cp.unique(labels, return_counts=True)
    unique, counts = unique[1:], counts[1:]
    largest_label = int(unique[int(cp.argmax(counts))])
    keep_mask[labels != largest_label] = 0
    keep_mask = cp.asarray(ndimage.binary_fill_holes(keep_mask))
    return keep_mask


# def find_start_and_end(locations):
#     l1_loc = locations['L1']['slice_idx']
#     l3_loc = locations['L3']['slice_idx']
#     l5_loc = locations['L5']['slice_idx']
#     start = min(l1_loc, l3_loc, l5_loc)
#     end = max(l1_loc, l3_loc, l5_loc)
#     if start != l1_loc or end != l5_loc:
#         logical_vert_order = False
#     else:
#         logical_vert_order = True
#     return start, end, logical_vert_order


# def adjust_vert_indices(locations):
#     vert_info = {}
#     # Save the original locs
#     l1_loc = locations['L1']['slice_idx']
#     l3_loc = locations['L3']['slice_idx']
#     l5_loc = locations['L5']['slice_idx']
#     vert_info['og_L1'] = l1_loc
#     vert_info['og_L3'] = l3_loc
#     vert_info['og_L5'] = l5_loc
#     # Save the adjusted locs
#     new_l3 = l3_loc - l1_loc
#     new_l5 = l5_loc - l1_loc
#     new_l1 = 0
#     vert_info['adj_L1'] = new_l1
#     vert_info['adj_L3'] = new_l3
#     vert_info['adj_L5'] = new_l5
#     return vert_info
    

# def postprocess(pred):
#     return cp.asarray(torch.round(torch.squeeze(pred)))


# def sum_fat_pixels(body_array, preds, spacing):
#     visc = body_array.copy()
#     subq = body_array.copy()
#     visc[preds == 0] = visc.min()
#     subq[preds == 1] = subq.min()
#     abd_area = preds.sum(axis=(1, 2)) * spacing[0] * spacing[1]
#     # [-190, -30] is the standard hounsfield window for fat
#     visc_pixels = ((visc >= -190) & (visc <= -30)).sum(axis=(1, 2))
#     subq_pixels = ((subq >= -190) & (subq <= -30)).sum(axis=(1, 2))
#     visc_area = visc_pixels * spacing[0] * spacing[1]
#     subq_area = subq_pixels * spacing[0] * spacing[1]
#     return abd_area, subq_pixels, subq_area, visc_pixels, visc_area
    

def get_waist_data(waist_mask, h, w):
    # h, w = spacing
    # circs = []
    # areas = []
    # for i in range(waist_mask.shape[0]):
    labels = label(waist_mask.get())
    regions = regionprops(labels)
    waist = max(regions, key=lambda x: x.area)
    # waist circ is in cm
    waist_circ = round(waist.perimeter * w / 10, 2)
    waist_area = round(waist.area * w * h, 2)
    # circs.append(waist_circ)
    # areas.append(waist_area)
    return waist_circ, waist_area


# def measure_vert_slices(vert_info, subq_pixels, subq_area, visc_pixels, visc_area, waist_circs, waist_areas, abd_areas):
#     vert_data = {}
#     for vert in ['L1', 'L3', 'L5']:
#         idx = vert_info[f'adj_{vert}']
#         vert_data[f'{vert}_measures'] = get_slice_vals(idx, subq_pixels, subq_area, visc_pixels, visc_area, waist_circs, waist_areas, abd_areas)
#     return vert_data


# def get_slice_vals(i, subq_pixels, subq_area, visc_pixels, visc_area, waist_circs, waist_areas, abd_areas):
#     slice_data = {}
#     slice_data['body_circ'] = float(waist_circs[i])
#     slice_data['body_area'] = round(float(waist_areas[i]), 2)
#     slice_data['abd_area'] = round(float(abd_areas[i]), 2)
#     slice_data['subq_pixels'] = float(subq_pixels[i])
#     slice_data['subq_area'] = round(float(subq_area[i]), 2)
#     slice_data['visc_pixels'] = float(visc_pixels[i])
#     slice_data['visc_area'] = round(float(visc_area[i]), 2)
#     return slice_data


# def plot_contours(original_image, body_mask, out_dir):
#     """
#     A function to plot the contours of the waist circumference around original image
#     :param original_image: the original L3 slice
#     :param body_mask: the body mask
#     :param output_path: the path to save the plot
#     :return: the plot of the contours
#     """
#     contours = find_contours(body_mask, 0.5)
#     fig, ax = plt.subplots()
#     ax.imshow(original_image, cmap=plt.cm.gray)

#     for contour in contours:
#         ax.plot(contour[:, 1], contour[:, 0], linewidth=3)

#     ax.axis('image')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     plt.savefig(f'{out_dir}/L3_waist_contour.jpg', dpi=150)
#     plt.close()

import cupy as cp
import numpy as np
import radiomics
import SimpleITK as sitk
from cucim.skimage import measure

def measure_aorta(original, full_mask, organ_map, vert_dict, spacing):
    """
    A function to measure the aortic data within the aorta mask, in abdominal, thoracic, and total aorta
    """
    out_dict = {}
    # print(f'{vert_dict=}')
    t12_idx = vert_dict['vertebrae_T12']
    out_dict['vertebrae_T12'] = t12_idx
    # Get the l, w, and h for each voxel in mm
    l, w, h = spacing
    aorta_idx = int(organ_map.get('aorta'))
    aorta_hu = cp.where(full_mask == aorta_idx, original, original.min())
    aorta_only = cp.where(full_mask == aorta_idx, 1, 0)
    calcification = cp.where(aorta_hu > 130, 1, 0)
    out_dict['total_aorta_calcification'] = float(cp.round(calcification.sum() * l * w * h, 2))
    out_dict['total_aorta_max_diameter'] = find_max_diameter(aorta_only, l)
    # print(f'{calcification.shape=}, {t12_idx=}')
    if (t12_idx is not None) and (t12_idx != (calcification.shape[-1] - 1)) and (t12_idx != 0):
        thoracic = calcification[:, :, t12_idx:]
        abdominal = calcification[:, :, :t12_idx]
        out_dict['thoracic_aorta_calc_volume'] = float(cp.round(thoracic.sum() * l * w * h, 2))
        out_dict['abdominal_aorta_calc_volume'] = float(cp.round(abdominal.sum() * l * w * h, 2))
        out_dict['thoracic_aorta_max_diameter'] = find_max_diameter(aorta_only[:, :, t12_idx:], l)
        out_dict['abdominal_aorta_max_diameter'] = find_max_diameter(aorta_only[:, :, :t12_idx], l)
        
    elif (t12_idx is not None) and (t12_idx == (calcification.shape[-1] - 1)):
        out_dict['abdominal_aorta_calc_volume'] = out_dict['total_aorta_calcification']
        out_dict['abdominal_aorta_max_diameter'] = out_dict['total_aorta_max_diameter']
        out_dict['thoracic_aorta_calc_volume'] = 0
        out_dict['thoracic_aorta_max_diameter'] = 0
    elif (t12_idx is not None) and (t12_idx == 0):
        out_dict['thoracic_aorta_calc_volume'] = out_dict['total_aorta_calcification']
        out_dict['thoracic_aorta_max_diameter'] = out_dict['total_aorta_max_diameter']
        out_dict['abdominal_aorta_calc_volume'] = 0
        out_dict['abdominal_aorta_max_diameter'] = 0
    else:
        out_dict['thoracic_aorta_calc_volume'] = None
        out_dict['abdominal_aorta_calc_volume'] = None
        out_dict['thoracic_aorta_max_diameter'] = None
        out_dict['abdominal_aorta_max_diameter'] = None
    return out_dict

#### PyRadiomics is better for this #####
def find_max_diameter(aorta, l):
    """
    A function to find the maximum axial diameter of the aorta
    """
    # l, _, _ = spacing
    aorta_slices = list(cp.unique(cp.where(aorta == 1)[-1]))
    max_diameter = 0
    largest_idx = None
    if len(aorta_slices) > 0:
        for i in aorta_slices:
            slice_ = aorta[:, :, i]
            labels = measure.label(slice_)
            props = measure.regionprops(labels)
            major_axis = max([prop.major_axis_length for prop in props])
            if major_axis > max_diameter:
                max_diameter = major_axis
                largest_idx = int(i)
    max_diameter = max_diameter * l
    return {'max_diameter': float(round(max_diameter, 2)), 'max_diameter_idx': largest_idx}

# def find_max_diameter(full_im, aorta, spacing):
#     """
#     Measure the maximum 2d diameter of the aorta using pyradiomics mesh
#     """
#     # Create the full sitk image
#     full_im_sitk = sitk.GetImageFromArray(full_im)
#     full_im_sitk.SetSpacing(tuple([float(x) for x in spacing]))
#     # Create the mask sitk image
#     aorta_sitk = sitk.GetImageFromArray(aorta.astype(np.int8))
#     aorta_sitk.SetSpacing(tuple([float(x) for x in spacing]))
#     # Generate the 3D mesh
#     mesh = radiomics.shape.RadiomicsShape(full_im_sitk, aorta_sitk)
    
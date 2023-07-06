import cupy as cp
import numpy as np


def measure_aortic_calcification(original, full_mask, organ_map, vert_dict, header):
    """
    A function to measure the aortic calcification within the aorta mask, in abdominal, thoracic, and total aorta
    """
    out_dict = {}
    # print(f'{vert_dict=}')
    t12_idx = vert_dict['vertebrae_T12']
    out_dict['vertebrae_T12'] = t12_idx
    # Get the l, w, and h for each voxel in mm
    l, w, h = header['length_mm'], header['width_mm'], header['slice_thickness_mm']
    aorta_idx = int(organ_map.get('aorta'))
    aorta_hu = cp.where(full_mask == aorta_idx, original, original.min())
    calcification = cp.where(aorta_hu > 130, 1, 0)
    out_dict['total_aorta_calcification'] = float(cp.round(calcification.sum() * l * w * h, 2))
    # print(f'{calcification.shape=}, {t12_idx=}')
    if (t12_idx is not None) and (t12_idx != (calcification.shape[-1] - 1)) and (t12_idx != 0):
        thoracic = calcification[:, :, t12_idx:]
        abdominal = calcification[:, :, :t12_idx]
        out_dict['thoracic_aorta_calc_volume'] = float(cp.round(thoracic.sum() * l * w * h, 2))
        out_dict['abdominal_aorta_calc_volume'] = float(cp.round(abdominal.sum() * l * w * h, 2))
    elif (t12_idx is not None) and (t12_idx == (calcification.shape[-1] - 1)):
        out_dict['abdominal_aorta_calc_volume'] = out_dict['total_aorta_calcification']
        out_dict['thoracic_aorta_calc_volume'] = 0
    elif (t12_idx is not None) and (t12_idx == 0):
        out_dict['thoracic_aorta_calc_volume'] = out_dict['total_aorta_calcification']
        out_dict['abdominal_aorta_calc_volume'] = 0
    else:
        out_dict['thoracic_aorta_calc_volume'] = None
        out_dict['abdominal_aorta_calc_volume'] = None
    return out_dict
    
import os
import gzip
import shutil
import numpy as np
import cupy as cp
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.map_to_binary import class_map

def segment_organs(nifti, output_dir, organs, subvals, fast):
    directory = '/'.join(nifti.split('/')[:-1])
    header = f'{directory}/header_info.json'
    if os.path.exists(header):
        name = f"{nifti.split('/')[-1].replace('.nii.gz', '')}_segs"
        seg_dir = f'{output_dir}/{name}'
        seg_name = f'{seg_dir}/{nifti.split("/")[-1].replace(".nii.gz", "")}'
        out = totalsegmentator(nifti, f'{seg_name}_full', roi_subset=organs, statistics=True, fast=fast, radiomics=False, nr_thr_saving=4, ml=True)
        gzip_seg(f'{seg_name}_full')
        if subvals is not None:
            subtask = totalsegmentator(nifti, f'{seg_name}_tissues', roi_subset=subvals, statistics=False, task='bones_tissue_test',nr_thr_saving=4, ml=True)
            gzip_seg(f'{seg_name}_tissues')
        else:
            subtask = None
        return out, subtask, seg_dir
    else:
        print('No header data was found for this nifti file. Will not proceed.')
        return None, None

    
def get_organ_indicies(organs, subvals):
    """
    A function to get the class numbers for the total segmentation organs from the class map
    """
    full_map = {v: k for k, v in class_map['total'].items()}
    tissue_idxs = {v: k for k, v in class_map['bones_tissue_test'].items()}

    organ_map = {organ: full_map[organ] for organ in organs}
    tissue_map = {val: tissue_idxs[val] for val in subvals}
    return organ_map, tissue_map

def gzip_seg(output_file):
    name = f'{output_file}.nii'
    with open(name, 'rb') as f_in:
        with gzip.open(f'{name}.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(name)
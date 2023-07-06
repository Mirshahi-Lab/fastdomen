import argparse
import json
import os
import time
import warnings
import cupy as cp
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd

from glob import glob

parser = argparse.ArgumentParser(description='Fastdomen - Fast and Automatic Quantification of CT Scans using TotalSegmentator')
parser.add_argument('--gpu', '-g', type=str, default='0', help='The gpu device ID to use, default = "0"')
parser.add_argument('--input', '-i', type=str, help='The input file to be analyzed (can be a directory of Dicom files or a Nifti File)')
parser.add_argument('--output-dir', '-o', type=str, help='The directory to store the output into (please use absolute paths)')

parser.add_argument('--fast-mode', action='store_true', help='Use the fast mode of Total Segmentator') 
parser.add_argument('--tmp-dir', type=str, default=None, help='Writable directory for temporary files. Only change if default tmp directory on system is not writable.')

args = parser.parse_args()

if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    raise RuntimeError('Please specify a gpu device to use with the --gpu argument')
if args.tmp_dir is not None:
    os.environ['TMPDIR'] = args.tmp_dir

import torch
from fastdomen.abdomen import measure_verts, total_torso_measure
from fastdomen.aorta import measure_aortic_calcification
from fastdomen.imaging.convert_dicom_to_nifti import convert_dicom
# from fastdomen.kidney import measure_estimated_volume
from fastdomen.segmentor import segment_organs, get_organ_indicies
from fastdomen.vertebra import find_vertebrae_centers, plot_vertebrae_overlay

def main():
    """
    Main function to run the pipeline
    """
    plt.ioff()
    if not torch.cuda.is_available():
        raise OSError('CUDA Device is not detected. Check your installation of pytorch for compatiblity.')
    # If a dicom series is given, convert it to a nifti file with the header preserved in header.json
    # Else, make sure that the output directory is defined properly
    if os.path.isdir(args.input):
        print('Converting Dicom folder to Nifti format...')
        args.output_dir = convert_dicom(args)
        args.input = glob(f'{args.output_dir}/*nii.gz')
        header = f'{args.output_dir}/header_info.json'
    else:
        input_dir = '/'.join(args.input.split('/')[:-1])
        mrn_acc_cut = '/'.join(args.input.split('/')[-4:-1])
        args.output_dir = f'{args.output_dir}/{mrn_acc_cut}'
        header = f'{input_dir}/header_info.json'
        args.input = [args.input]
        
    if os.path.exists(f'{args.output_dir}/totseg_v1_completed.txt'):
        with open(f'{args.output_dir}/totseg_v1_completed.txt', 'r') as f:
            completed = f.read().splitlines()
    else:
        completed = []
            
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    
    # Define the objects to save
    full_organs = [
        'liver',
        'spleen',
        'kidney_left',
        'kidney_right',
        'stomach',
        'aorta',
        'vertebrae_L1',
        'vertebrae_L3',
        'vertebrae_L5',
        'vertebrae_T12'
    ]
    subitems = [
        'subcutaneous_fat',
        'torso_fat',
        'skeletal_muscle'
    ]

    # Need to loop through this if the dicom files convert into multiple niftis
    for input_ in args.input:
        if input_ in completed:
            print(f"{input_} has already been segmented with this version of Fastdomen.")
            continue
        orgs, fats, seg_dir = segment_organs(input_, args.output_dir, full_organs, subitems, args.fast_mode)
        original = cp.asarray(nib.load(input_).get_fdata())
        # Push the segs to gpu for fast calcs
        orgs = cp.asarray(orgs.get_fdata())
        fats = cp.asarray(fats.get_fdata())
        # Get the maps for indices as the masks are multi-indexed
        organ_map, tissue_map = get_organ_indicies(full_organs, subitems)
        # Load the stats for each segmented organ
        with open(f'{seg_dir}/statistics.json', 'r') as f:
            stats = json.load(f)
        # Load the header data 
        with open(header, 'r') as f:
            header_dict = json.load(f)
        header_dict['image_shape'] = original.shape
        spacing = [header_dict['length_mm'], header_dict['width_mm'], header_dict['slice_thickness_mm']]
        # Get vertebrae information
        vertebrae_locs = find_vertebrae_centers(orgs, organ_map, stats)
        # print(vertebrae_locs)
        plot_vertebrae_overlay(input_, vertebrae_locs, seg_dir)
        vert_vals = measure_verts(fats, tissue_map, vertebrae_locs, header_dict)
        torso_vals = total_torso_measure(fats, tissue_map, vertebrae_locs, header_dict)
        # Measure aortic calcification
        aorta_calc = measure_aortic_calcification(original, orgs, organ_map, vertebrae_locs, header_dict)
        
        header_dict.update(vert_vals)
        header_dict['tissue_stats'] = torso_vals
        
        organ_dict = {organ: stats[organ] for organ in full_organs[:6]}
        header_dict.update(organ_dict)
        header_dict['aorta_stats'] = aorta_calc
        with open(f'{seg_dir}/fastdomen_out.json', 'w') as f:
            json.dump(header_dict, f)
        with open(f'{args.output_dir}/totseg_v1_completed.txt', 'a') as f:
            f.write(f'{input_}\n')
    print('Done!')
if __name__ == '__main__':
    main()
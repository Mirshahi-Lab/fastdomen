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
parser.add_argument('--input', '-i', type=str, help='The input file to be analyzed (can be a directory of Dicom files or a Nifti File)')
parser.add_argument('--output-dir', '-o', type=str, help='The directory to store the output into (please use absolute paths)')
parser.add_argument('--gpu', '-g', type=str, default='0', help='The gpu device ID to use, default = "0"')
parser.add_argument('--fast-mode', action='store_true', help='Use the fast mode of Total Segmentator') 
parser.add_argument('--tmp-dir', type=str, default=None, help='Writable directory for temporary files. Only change if default tmp directory on system is not writable.')

args = parser.parse_args()
print(f'{args=}')
if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    raise RuntimeError('Please specify a gpu device to use with the --gpu argument')
if args.tmp_dir is not None:
    os.environ['TMPDIR'] = args.tmp_dir

warnings.filterwarnings('ignore', category=UserWarning)

import torch
from fastdomen.imaging.convert_dicom_to_nifti import convert_dicom
from fastdomen.segmentor import segment_organs

def main(args):
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
        args.input = glob(f'{args.output_dir}/*nii.gz')[0]
        header = f'{args.output_dir}/header.json'
    else:
        input_dir = '/'.join(args.input.split('/')[:-1])
        mrn_acc_cut = '/'.join(args.input.split('/')[-4:-1])
        args.output_dir = f'{args.output_dir}/{mrn_acc_cut}'
        header = f'{input_dir}/header.json'
        
    print(args.input, header, args.output_dir)
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
    
    orgs, fats = segment_organs(args.input, args.output_dir, full_organs, subitems, args.fast_mode)
    
main(args)
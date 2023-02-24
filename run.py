import argparse
import cupy as cp
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import warnings

parser = argparse.ArgumentParser(description='Fastdomen - Fast and Automatic Quantifcation of Abdominal CT Scans')
parser.add_argument('--series-list', '-i', type=str, help='The list of ct series to quantify (please use absolute path names)')
parser.add_argument('--output-dir', '-o', type=str, help='The directory to store the output into (please use absolute path names)')
parser.add_argument('--file-extension', '-e', type=str, help='The file extension of the dicom files, default = "*", matching every file', default='*')
parser.add_argument('--gpu', '-g', default='0', type=str, help='Specified gpu device to use, default = 0')
parser.add_argument('--failed-series', type=str, default=None, help='File to save the names of series that failed quants')
parser.add_argument('--completed-series', type=str, default=None, help='File to save the names of completed series')
args = parser.parse_args()

if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    raise RuntimeError('Please specify a gpu device to use with the --gpu argument.')
warnings.filterwarnings('ignore', category=UserWarning)
# These need to be imported after the argument parsing so gpu selection works properly
import torch
from fastdomen.unext.archs import UNext
from fastdomen.imaging.dicomseries import DicomSeries
from fastdomen.liver import measure_liver_hu, measure_spleen_hu
from fastdomen.vertebra import detect_vertebra_slices
from fastdomen.abdomen import measure_abdomen


def check_series(series, file_extension):
    try:
        ds = DicomSeries(series, file_extension)
        cut = str(ds.cut).lower()
        if ds.num_files < 50:
            usable = False
        elif ('bone' in cut) or ('head' in cut) or ('spine' in cut) or ('neck' in cut):
            usable = False
        elif (ds.series_info['ct_direction'] != 'AX') and (ds.series_info['ct_direction'] is not None):
            usable = False
        else:
            usable = True
        if not usable:
            print(f'Dicom Series {series} is not usable for pipeline.\n')
        return usable
    except Exception as e:
        print(f'Error: {e}\n Dicom Series {series} is not usable for pipeline.\n')
        return False


def main():
    """
    Main function to run the analysis
    """
    # print(args)
    torch.set_num_threads(4)
    plt.ioff()
    if not torch.cuda.is_available():
        raise OSError('CUDA Device is not detected. Check your installation of pytorch for compatiblity.')
        
    with open(args.series_list, 'r') as f:
        series_list = [x.strip('\n') for x in f.readlines()]
    
    if args.failed_series is None:
        args.failed_series = os.path.splitext(args.series_list)[0] + '_failed.txt'
    if args.completed_series is None:
        args.completed_series = os.path.splitext(args.series_list)[0] + '_completed.txt'
    # print(series_list)
    model = UNext(1, 1, img_size=512, in_chans=1).cuda()
    model.eval()
    
    vert_weights = {
        'L1': 'fastdomen/unext/models/vertebra/L1_model.pth',
        'L3': 'fastdomen/unext/models/vertebra/L3_model.pth',
        'L5': 'fastdomen/unext/models/vertebra/L5_model.pth',
    }
    for idx, series in enumerate(series_list):
        print(f'Series {idx+1}/{len(series_list)}')
        start_time = time.time()
        print(f'Analyzing {series}')
        usable = check_series(series, args.file_extension)
        if usable:
            quant_data = {}
            ds = DicomSeries(series, args.file_extension)
            
            # Make output directory if it doesn't exist
            output_dir = f'{args.output_dir}/{ds.mrn}/{ds.accession}/{ds.cut}'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            quant_data['header'] = ds.series_info
            # Segment and measure the liver
            try:
                print('Segmenting and measuring liver...')
                liver_data = measure_liver_hu(ds, model, 'fastdomen/unext/models/liver.pth', output_dir)
                quant_data['liver_data'] = liver_data
                print('Done.\n')
            except Exception as e:
                print(f'Liver measurement failed:\n \t{e}\n')
            # Segment and measure the spleen
            try:
                print('Segmenting and measuring spleen...')
                spleen_data = measure_spleen_hu(ds, model, 'fastdomen/unext/models/spleen.pth', output_dir)
                quant_data['spleen_data'] = spleen_data
                print('Done.\n')
            except Exception as e:
                print(f'Spleen measurement failed:\n \t{e}\n')
            
            
            # Detect vertebra slice indices and quant the abdomen
            try:
                print('Detecting vertebra locations...')
                locations = detect_vertebra_slices(ds, model, vert_weights, output_dir)
                quant_data['vertebra_data'] = locations
                print('Done\n')
                # Quant the abdomen
                print('Measuring abdominal fat...')
                slice_data = measure_abdomen(ds, model, locations, 'fastdomen/unext/models/abdomen.pth', output_dir)
                quant_data['abdomen_data'] = slice_data
                print('Done\n')
            except Exception as e:
                print(f'Vertebra Detection/Abdomen quant failed:\n \t{e}\n')
            runtime = time.time() - start_time
            quant_data['runtime'] = f'{runtime:.2f}'
            print(f'Fastdomen runtime: {runtime:.2f} seconds')
            print(f'Saving measurements to {output_dir}/{ds.filename}_quant.json')
            json.dump(quant_data, open(f'{output_dir}/{ds.filename}_quant.json', 'w'))
            # plt.close()
            torch.cuda.empty_cache()
            cp._default_memory_pool.free_all_blocks()
            print(f'----Done----\n')
            with open(args.completed_series, 'a') as f:
                f.write(f'{series}\n')
        else:
            print(f'Series {series} is not the proper type of CT for measurement.\n')
            with open(args.failed_series, 'a') as f:
                f.write(f'{series}\n')
        
if __name__ == '__main__':
    main()

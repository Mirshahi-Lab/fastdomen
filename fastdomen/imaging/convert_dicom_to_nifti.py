import argparse
import os
import json
from glob import glob
import dicom2nifti
import dicom2nifti.settings as settings
from pydicom import dcmread
from fastdomen.imaging.dicomseries import DicomSeries

# parser = argparse.ArgumentParser(description='Convert Dicom directory to a Nifti Format file for TotalSegmentator')
# parser.add_argument('--input', '-i', type=str, help='The directory to convert')
# parser.add_argument('--output-dir', '-o', type=str, help='The base directory to store the converted files into')
# args = parser.parse_args()

def convert_dicom(args):
    indir = args.input
    outdir = args.output_dir
    dicom_files = glob(f'{indir}/*')
    outdir = outdir[:-1] if outdir[-1] == '/' else outdir
    print(indir)
    print(f'Num. Files in series: {len(dicom_files)}')
    if len(dicom_files) > 20:
        # header = dcmread(dicom_files[0])
        full_info = DicomSeries(indir, '*', make_frontal=False)
        mrn = full_info.mrn
        acc = full_info.accession
        cut = full_info.series_name
        # cut = cut.replace(' ', '_')
        directory = f'{outdir}/{mrn}/{acc}/{cut}'
        print(f'Saving to {directory}')
        if not os.path.exists(directory):
            os.makedirs(directory)
        if os.path.exists(f'{directory}/converted.txt'):
            print('This file has been converted already.')
            return directory
        try:
            dicom2nifti.convert_directory(indir, directory)
            with open(f'{directory}/header_info.json', 'w') as f:
                json.dump(full_info.series_info, f)
            with open(f'{directory}/converted.txt', 'w') as f:
                f.write('True')
            print('Series Successfully Converted!\n')
        except Exception as e:
            print(e)
            os.rmdir(directory)
            print("Series Failed to Convert.\n")
        return directory
    else:
        print(f'Dicom Series {indir} is not long enough to be an abdominal/chest scan\n')
        
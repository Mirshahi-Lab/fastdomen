import logging
import os
from glob import glob
from datetime import datetime
import natsort
import numpy as np
import cupy as cp
import pydicom
# from numba import njit
from fastdomen.imaging.utils import transform_to_hu, window_image


class DicomSeries:
    """
    A class representing a series of dicom files in a specified directory\n

    """
    orientations = {
        'COR': [1, 0, 0, 0, 0, -1],
        'SAG': [0, 1, 0, 0, 0, -1],
        'AX': [1, 0, 0, 0, 1, 0]
    }
    important_attributes = {
        'PatientID': 'mrn',
        'AccessionNumber': 'accession',
        'SeriesDescription': 'series_name',
        'ImageOrientationPatient': 'ct_direction',
        'ImageType': 'image_type',
        'PatientSex': 'sex',
        'PatientBirthDate': 'birthday',
        'AcquisitionDate': 'scan_date',
        'PatientAge': 'age_at_scan',
        'PixelSpacing': 'pixel_spacing',
        'SliceThickness': 'slice_thickness_mm',
        'Manufacturer': 'manufacturer',
        'ManufacturerModelName': 'manufacturer_model',
        'KVP': 'kvp',
        'ContrastBolusAgent': 'contrast_bolus_agent',
        'ContrastBolusRoute': 'contrast_bolus_route',
        'MultienergyCTAcquisitionSequence': 'multienergy_ct'
    }

    def __init__(self, directory, filepattern='*.dcm', make_frontal=True):
        """
        :param directory: the directory containing the series
        :param filepattern: the file pattern of the series
        :param window_center: the center of the hounsfield units to view the images (default: 30)
        :param window_width: the width of the hounsfield units to view the images (default: 150)
        :param read_images: a boolean indicating whether to read the images or not (default: True)
        """
        if not os.path.exists(directory) or not os.path.isdir(directory):
            raise ValueError(f'Given directory does not exist or is not a file: {directory}')

        self.directory = os.path.abspath(directory)
        self.filepattern = filepattern
        self.file_list = glob(os.path.join(directory, filepattern))
        self.num_files = len(self.file_list)
        self.header = pydicom.dcmread(self.file_list[0])
        self.series_info = self._get_image_info(self.header)
        self.series_info['directory'] = self.directory
        self.mrn = self.series_info['mrn']
        self.accession = self.series_info['accession']
        self.series_name = self.series_info['series_name']
        self.filename = f'MRN{self.mrn}_{self.accession}_{self.series_name}'
        self.spacing = [float(self.header.PixelSpacing[0]), float(self.header.PixelSpacing[1]), float(self.header.SliceThickness)]
        if make_frontal:
            self.frontal = self.get_mip(axis=1)

    # @staticmethod
    def read_dicom_series(self, window_center, window_width):
        """
        A function to read a series of dicom files into one numpy array \n
        :param directory: directory containing the files
        :param filepattern: filepattern to match dicom files
        :param slope: the slope rescaling factor from the dicom file (0 if already in hounsfield units)
        :param intercept: the intercept value from the dicom file (depends on the machine)
        :param window_center: hounsfield window center
        :param window_width: hounsfield window width
        :return: a numpy array containing the dicom series
        """
        files = natsort.natsorted(self.file_list)

        # create an empty dictionary to store image and image number
        ct_scans = {}
        # read all the files
        for i, file in enumerate(files):
            ds = pydicom.dcmread(file)
            # get the image number
            if hasattr(ds, 'InstanceNumber'):
                image_number = int(ds.InstanceNumber)
            else:
                image_number = i
            image = cp.asarray(ds.pixel_array, dtype=np.float32)
            # store the image and number in the dictionary
            ct_scans[image_number] = image

        # sort the images by image number
        sorted_ct = [ct_scans[key] for key in sorted(ct_scans.keys())]
        # stack the images into one array
        combined_image = cp.stack(sorted_ct, axis=0)
        # Convert to hounsfield units
        hu_image = transform_to_hu(combined_image, self.header.RescaleSlope, self.header.RescaleIntercept)
        # Window the hounsfield units
        windowed_image = window_image(hu_image, window_center, window_width)
        # return the windowed image
        return windowed_image

    def _get_image_info(self, header):
        """
        A function to get the important information from the dicom header \n
        :param header: the dicom header object
        :return: a dictionary containing the important information
        """
        series_info = {}
        # Loop through all the important attributes
        for tag, column in self.important_attributes.items():
            try:
                value = getattr(header, tag)
                if tag == "SeriesDescription":
                    value = value.replace('/', '_').replace(' ', '_').replace('-', '_').replace(
                        ',', '.')
                elif tag == "ImageOrientationPatient":
                    orientation = np.round(getattr(header, tag))
                    value = None
                    for key, direction in self.orientations.items():
                        if np.array_equal(orientation, direction):
                            value = key
                            break
                elif tag == 'ImageType':
                    value = '_'.join(value)
                elif tag == 'PatientBirthDate':
                    value = datetime.strptime(value, '%Y%m%d').date()
                    value = value.strftime('%Y-%m-%d')
                elif tag == 'AcquisitionDate':
                    value = datetime.strptime(value, '%Y%m%d').date()
                    value = value.strftime('%Y-%m-%d')
                elif tag == 'PatientAge':
                    value = int(value[:-1])
                elif tag == 'PixelSpacing':
                    # Value in this case is a list
                    length, width = value
                    series_info['length_mm'] = length
                    series_info['width_mm'] = width
                    continue
                elif tag == 'PatientSex':
                    if value == 'M':
                        value = 1
                    elif value == 'F':
                        value = 0
                elif tag == 'ContrastBolusAgent' or tag == 'ContrastBolusRoute':
                    value = True
                elif tag == 'MultienergyCTAcquisitionSequence':
                    value = True

                series_info[column] = value

            except AttributeError:
                logging.info(f'{tag} not found in dicom header')
                series_info[column] = None

        return series_info

    def get_mip(self, axis):
        """
        A method to get the Maximum Intensity Projection (MIP) of the dicom pixel array along an axis \n
        :param pixel_array: the dicom pixel array
        :param axis: the axis to project along
        :return: the frontal MIP
        """
        # Get the frontal MIP
        mip = cp.amax(self.read_dicom_series(1000, 3000), axis=axis)
        return mip
    
    def calculate_pixel_radius(self, roi_area=3):
        pixel_width = self.spacing[0] / 10
        radius = cp.sqrt(roi_area / cp.pi)
        pixel_radius = int(np.ceil(radius / pixel_width))
        return pixel_radius

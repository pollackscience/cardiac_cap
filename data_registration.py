import os
import numpy as np
import pandas as pd
import xarray as xr
import pickle as pkl
from datetime import datetime
from scipy import ndimage as ndi
import SimpleITK as sitk
import skimage as skim
from skimage import feature, morphology
import glob


class RegHearts:
    '''Class that generates liver masks for MRE input images'''

    def __init__(self, fixed_subj, moving_subj, tslice=0, verbose=False):

        self.verbose = verbose
        self.fixed_subj = fixed_subj
        self.moving_subj = moving_subj
        self.tslice = tslice

        self.load_niftis()

    def load_niftis(self):
        fixed_ct_name = os.path.join(self.fixed_subj, f'CT_tslice_{self.tslice}.nii')
        fixed_mask_name = os.path.join(self.fixed_subj, f'mask_tslice_{self.tslice}.nii')
        moving_ct_name = os.path.join(self.moving_subj, f'CT_tslice_{self.tslice}.nii')
        moving_mask_name = os.path.join(self.moving_subj, f'mask_tslice_{self.tslice}.nii')

        self.fixed_ct = self.get_sitk_image(fixed_ct_name)
        self.fixed_mask = self.get_sitk_image(fixed_mask_name)
        self.moving_ct = self.get_sitk_image(moving_ct_name)
        self.moving_mask = self.get_sitk_image(moving_mask_name)

    def get_sitk_image(self, nifti_name):
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(nifti_name)
        img = reader.Execute()
        size = img.GetSize()
        dims = img.GetSpacing()
        orig = img.GetOrigin()
        if self.verbose:
            print(f"Image info for {nifti_name}:")
            print("Image size:", size[0], size[1], size[2])
            print("Image dims:", dims[0], dims[1], dims[2])
            print("Image orig:", orig[0], orig[1], orig[2])

        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(sitk.sitkFloat32)
        return caster.Execute(img)

    def gen_param_map(self):
        self.p_map_vector = sitk.VectorOfParameterMap()
        paff = sitk.GetDefaultParameterMap("affine")
        pbsp = sitk.GetDefaultParameterMap("bspline")
        paff['AutomaticTransformInitialization'] = ['true']
        paff['AutomaticTransformInitializationMethod'] = ['GeometricalCenter']
        paff['NumberOfSamplesForExactGradient'] = ['100000']
        pbsp['NumberOfSamplesForExactGradient'] = ['100000']
        # paff['MaximumNumberOfSamplingAttempts'] = ['2']
        # pbsp['MaximumNumberOfSamplingAttempts'] = ['2']
        paff['NumberOfSpatialSamples'] = ['5000']
        pbsp['NumberOfSpatialSamples'] = ['5000']
        paff['NumberOfHistogramBins'] = ['32', '32', '64', '128']
        pbsp['NumberOfHistogramBins'] = ['32', '32', '64', '128']
        paff['MaximumNumberOfIterations'] = ['1024']
        pbsp['MaximumNumberOfIterations'] = ['1024']
        # paff['NumberOfResolutions'] = ['4']
        # pbsp['NumberOfResolutions'] = ['4']
        paff['GridSpacingSchedule'] = ['6', '4', '2', '1.000000']
        pbsp['GridSpacingSchedule'] = ['6', '4', '2', '1.000000']
        # pbsp['FinalGridSpacingInPhysicalUnits'] = ['40', '40', '40']
        pbsp['FinalGridSpacingInPhysicalUnits'] = ['32', '32', '32']
        # pbsp['Metric0Weight'] = ['0.01']
        # pbsp['Metric1Weight'] = ['0.1']
        # paff['FixedImagePyramid'] = ['FixedShrinkingImagePyramid']
        # pbsp['FixedImagePyramid'] = ['FixedShrinkingImagePyramid']

        # attempting to use multiple fixed images at once
        # paff['Registration'] = ['MultiMetricMultiResolutionRegistration']
        # paff['FixedImagePyramid'] = ['FixedSmoothingImagePyramid', 'FixedSmoothingImagePyramid']
        # paff['ImageSampler'] = ['RandomCoordinate', 'RandomCoordinate']
        # paff['Metric'] = ['AdvancedMattesMutualInformation', 'AdvancedMattesMutualInformation']
        # pbsp['Metric'] = ['AdvancedMattesMutualInformation', 'TransformBendingEnergyPenalty',
        #                  'AdvancedMattesMutualInformation', 'TransformBendingEnergyPenalty']
        # pbsp['FixedImagePyramid'] = ['FixedSmoothingImagePyramid', 'FixedSmoothingImagePyramid']
        # pbsp['ImageSampler'] = ['RandomCoordinate', 'RandomCoordinate']
        #                         'RandomCoordinate', 'RandomCoordinate']
        self.p_map_vector.append(paff)
        self.p_map_vector.append(pbsp)
        if self.verbose:
            sitk.PrintParameterMap(self.p_map_vector)

    def register_imgs(self):
        self.elastixImageFilter = sitk.ElastixImageFilter()
        self.elastixImageFilter.SetFixedImage(self.fixed_ct)

        self.elastixImageFilter.SetMovingImage(self.moving_ct)
        self.elastixImageFilter.SetParameterMap(self.p_map_vector)
        self.elastixImageFilter.Execute()
        self.moving_ct_result = self.elastixImageFilter.GetResultImage()
        self.moving_ct_result.CopyInformation(self.fixed_ct)

    def gen_mask(self, smooth=False):
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(
            self.elastixImageFilter.GetTransformParameterMap())
        transformixImageFilter.SetMovingImage(self.moving_mask)
        transformixImageFilter.Execute()
        self.moving_mask_result = transformixImageFilter.GetResultImage()

        if smooth:
            tmp_img = sitk.GetArrayFromImage(self.moving_mask_result)
            tmp_img = np.where((tmp_img > 0), 1, 0)
            self.moving_mask_result = sitk.GetImageFromArray(tmp_img)

        self.moving_mask_result.CopyInformation(self.fixed_ct)
        self.moving_mask_result = sitk.Cast(self.moving_mask_result, sitk.sitkFloat32)

    def recenter_img_z(self, sitk_img, offset=False):
        spacing = sitk_img.GetSpacing()[2]
        layers = sitk_img.GetSize()[2]
        orig = sitk_img.GetOrigin()
        if not offset:
            sitk_img.SetOrigin([orig[0], orig[1], spacing*(-layers/2)])
        else:
            sitk_img.SetOrigin([orig[0], orig[1], spacing*(-layers/1.5)])


def add_liver_mask(ds, moving_name='19', extra_name='extra1'):
    '''Generate a mask from the liver registration method, and place it into the given "extra" slot.
    Assumes you are using an xarray dataset from the MREDataset class.'''

    for sub in tqdm(ds.subject):
        mask_maker = MRELiverMask(str(sub.values), moving_name, verbose=False, center=True,
                                  fixed_seq='T1Pre', moving_seq='T1_inphase')
        mask_maker.gen_param_map()
        mask_maker.register_imgs()
        mask_maker.gen_mask(smooth=True)
        mask = sitk.GetArrayFromImage(mask_maker.moving_mask_result)
        mask = np.where(mask >= 1, 1, 0)
        ds['image'].loc[dict(sequence=extra_name, subject=sub)] = mask

    new_sequence = [a.replace(extra_name, 'liverMsk') for a in ds.sequence.values]
    ds = ds.assign_coords(sequence=new_sequence)
    return ds

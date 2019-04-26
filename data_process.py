import pandas as pd
import xarray as xr
import SimpleITK as sitk
import numpy as np
import glob


class DicomToXArray:
    def __init__(self, patient_dir):
        self.dir = patient_dir
        self.SAs = glob.glob(patient_dir+'/SA*')
        self.raw_image_dict = {}
        self.image_list = []
        self.mask_list = []
        self.metadata_dict = {}
        self.timestamp_dict = {}
        self.weirdness_dict = {}
        self.loc_dict = {}
        self.slice_dict = {}
        self.shape_dict = {}
        self.spacing_dict = {}
        self.direction_dict = {}
        self.bad_SAs = set()
        self.good_SAs = set()

        print(self.SAs)
        self.get_images_and_metadata()
        self.make_xarray()

    def get_reader(self, SA):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(SA)
        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn()  # Get DICOM Info
        reader.LoadPrivateTagsOn()  # Get DICOM Info
        return reader

    def get_reader_and_image(self, SA):
        reader  = self.get_reader(SA)
        image = reader.Execute()
        return reader, image

    def load_metadata(self, SA, reader, slices):
        # load metadata for first timestamp
        # (except the timestamp value, need to get that for all)

        vals = []
        for k in reader.GetMetaDataKeys(0):
            vals.append(reader.GetMetaData(0, k))
        self.metadata_dict[SA] = pd.Series(vals, reader.GetMetaDataKeys(0))

        locs = [float(a) for a in self.metadata_dict[SA]['0020|0032'].split('\\')]
        self.loc_dict[SA] = locs

        self.slice_dict[SA] = float(self.metadata_dict[SA]['0020|1041'])

        spacing = [float(a) for a in self.metadata_dict[SA]['0028|0030'].split('\\')]
        self.spacing_dict[SA] = spacing

        self.timestamp_dict[SA] = []
        for i in range(slices):
            self.timestamp_dict[SA].append(int(reader.GetMetaData(i, '0020|0013')))

    def calc_weirdness(self):
        for SA in self.SAs:
            weirdness = 0
            for SA_test in self.SAs:
                if SA != SA_test:
                    weirdness += np.sum(~self.metadata_dict[SA].eq(self.metadata_dict[SA_test]))
            self.weirdness_dict[SA] = weirdness

    def mark_bads(self):
        # mark slices as bad if any exist
        for SA in self.SAs:
            for SA_test in self.SAs:
                if SA_test == SA or (SA_test in self.bad_SAs):
                    continue
                too_close = np.isclose(self.loc_dict[SA][-1], self.loc_dict[SA_test][-1], atol=0.5)
                too_close_slice = np.isclose(self.slice_dict[SA],
                                             self.slice_dict[SA_test], atol=0.5)
                if too_close or too_close_slice:
                    print(SA, SA_test)
                    print(self.loc_dict[SA], self.loc_dict[SA_test])
                    if self.weirdness_dict[SA] > self.weirdness_dict[SA_test]:
                        self.bad_SAs.add(SA)
                    else:
                        self.bad_SAs.add(SA_test)
        print(self.bad_SAs)
        self.good_SAs = self.bad_SAs.symmetric_difference(self.SAs)

    def get_ideal_params(self):
        x_shapes = [self.shape_dict[SA][0] for SA in self.good_SAs]
        y_shapes = [self.shape_dict[SA][1] for SA in self.good_SAs]
        x_spacings = [self.spacing_dict[SA][0] for SA in self.good_SAs]
        y_spacings = [self.spacing_dict[SA][1] for SA in self.good_SAs]
        x_origins = [self.loc_dict[SA][0] for SA in self.good_SAs]
        y_origins = [self.loc_dict[SA][1] for SA in self.good_SAs]
        directions = [self.direction_dict[SA] for SA in self.good_SAs]

        self.ideal_x_shape = int(np.mean(x_shapes))
        self.ideal_y_shape = int(np.mean(y_shapes))
        self.ideal_x_spacing = np.mean(x_spacings)
        self.ideal_y_spacing = np.mean(y_spacings)
        self.ideal_x_origin = np.mean(x_origins)
        self.ideal_y_origin = np.mean(y_origins)
        directions = np.asarray(directions)
        self.ideal_directions = directions.mean(axis=0)

    def get_images_and_metadata(self):
        for i, SA in enumerate(self.SAs):
            reader, image = self.get_reader_and_image(SA)
            self.shape_dict[SA] = image.GetSize()
            self.direction_dict[SA] = image.GetDirection()
            self.load_metadata(SA, reader, image.GetSize()[-1])
            self.raw_image_dict[SA] = image

        self.calc_weirdness()
        self.mark_bads()
        self.get_ideal_params()

        print(self.good_SAs)
        for SA in self.good_SAs:
            mask = self.get_mask(SA)
            image, mask = self.resample_images(SA, mask)
            self.image_list.append(sitk.GetArrayFromImage(image))
            self.mask_list.append(sitk.GetArrayFromImage(mask))

    def resample_images(self, SA, mask):
        image = self.raw_image_dict[SA]
        ref_image = sitk.Image((int(self.ideal_x_shape), int(self.ideal_y_shape),
                                int(image.GetSize()[-1])), 2)
        ref_image.SetSpacing((self.ideal_x_spacing, self.ideal_y_spacing, 1))
        ref_image.SetOrigin((self.ideal_x_origin, self.ideal_y_origin, image.GetOrigin()[-1]))
        # ref_image.SetOrigin(image.GetOrigin())
        # ref_image.SetDirection(image.GetDirection())
        ref_image.SetDirection(self.ideal_directions)
        ref_image = sitk.Cast(ref_image, image.GetPixelIDValue())
        center = sitk.CenteredTransformInitializer(
            ref_image, image, sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        new_image = sitk.Resample(image, ref_image, center,
                                  sitk.sitkNearestNeighbor)
        new_mask = sitk.Resample(mask, ref_image, center,
                                 sitk.sitkNearestNeighbor)
        return new_image, new_mask

    def get_mask(self, SA):
        image = self.raw_image_dict[SA]
        png_names = SA + '/*.png'
        tmp_mask_list = []
        for i, fn in enumerate(sorted(glob.glob(png_names))):
            tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage(fn))
            tmp_mask_list.append(tmp_mask[:, :, 0])

        mask_array = np.zeros((len(tmp_mask_list), tmp_mask_list[0].shape[0],
                               tmp_mask_list[0].shape[1]), dtype=np.float32)
        for i, m in enumerate(tmp_mask_list):
            mask_array[i, :, :] = m

        mask = sitk.GetImageFromArray(mask_array)
        mask.SetDirection(image.GetDirection())
        mask.SetOrigin(image.GetOrigin())
        mask.SetSpacing(image.GetSpacing())
        return mask

    def make_xarray(self):
        print(np.asarray(self.image_list).shape)

        zs = [self.loc_dict[SA][-1] for SA in self.good_SAs]
        self.xrds = xr.Dataset({'image': (['z', 't', 'y', 'x'], self.image_list),
                                'mask': (['z', 't', 'y', 'x'], self.mask_list)},
                               coords={'t': self.timestamp_dict[list(self.good_SAs)[0]],
                                       'x': range(self.ideal_x_shape),
                                       'y': range(self.ideal_y_shape),
                                       'z': zs})
        self.xrds = self.xrds.sortby(['t', 'z'])

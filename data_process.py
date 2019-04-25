import xarray as xr
import SimpleITK as sitk
import numpy as np
import glob


class DicomToXArray:
    def __init__(self, patient_dir):
        self.dir = patient_dir
        self.SAs = glob.glob(patient_dir+'/SA*')
        self.image_list = []
        self.mask_list = []
        self.metadata_list = []
        self.x_shape = None
        self.y_shape = None
        self.x_spacing = None
        self.y_spacing = None
        self.x_origin = None
        self.y_origin = None
        self.zs_type1 = []
        self.zs_type2 = []

        print(self.SAs)
        self.get_images_and_metadata()
        self.make_xarray()

    def get_images_and_metadata(self):
        for i, SA in enumerate(self.SAs):
            reader, image = self.get_reader_and_image(SA)
            # if reader.GetMetaData(0, '0018|1312').strip() == 'ROW':
            #     print('row')
            #     continue
            if not self.x_shape:
                # Getting shape, spacing, and origin for X-Y
                # Using first temporal slice grabbed, ignoring conflicting
                # info from all other temporal slices
                self.x_shape = image.GetSize()[0]
                self.y_shape = image.GetSize()[1]
                self.x_spacing = image.GetSpacing()[0]
                self.y_spacing = image.GetSpacing()[1]
                self.x_origin = image.GetOrigin()[0]
                self.y_origin = image.GetOrigin()[1]
                self.direction = image.GetDirection()
            metadata = self.get_metadata(reader, image)
            mask = self.get_mask(SA, image)

            # image, mask, metadata = self.check_axis(image, mask, metadata)
            image, mask, metadata = self.resample_images(image, mask, metadata)

            self.image_list.append(sitk.GetArrayFromImage(image))
            self.mask_list.append(sitk.GetArrayFromImage(mask))
            self.metadata_list.append(metadata)

    def get_reader_and_image(self, SA):
        reader = sitk.ImageSeriesReader()
        # dicom_names = reader.GetGDCMSeriesFileNames(self.dir, sid)
        dicom_names = reader.GetGDCMSeriesFileNames(SA)
        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn()  # Get DICOM Info
        reader.LoadPrivateTagsOn()  # Get DICOM Info
        image = reader.Execute()
        return reader, image

    def get_metadata(self, reader, image):
        # Assume metadata for first temporal slice works for everything
        # (except temporal slice position)

        xyz = reader.GetMetaData(5, '0020|0032')
        xyz = [float(a) for a in xyz.split('\\')]
        other_z = float(reader.GetMetaData(5, '0020|1041'))
        x = xyz[0]
        y = xyz[1]
        # z = other_z
        z = xyz[2]
        self.zs_type1.append(z)
        self.zs_type2.append(other_z)

        spacing = reader.GetMetaData(0, '0028|0030')
        spacing = [float(a) for a in spacing.split('\\')]
        x_spacing = spacing[0]
        y_spacing = spacing[1]

        t = []
        for i in range(image.GetSize()[-1]):
            t.append(int(reader.GetMetaData(i, '0020|0013')))

        return [x, y, z, t, x_spacing, y_spacing]

    def check_axis(self, image, mask, metadata):
        im_shape = list(image.GetSize()[0:2])
        true_shape = [self.x_shape, self.y_shape]
        if im_shape == true_shape:
            return image, mask, metadata
        # elif im_shape == true_shape[::-1]:
        elif im_shape != true_shape:
            ref_image = sitk.Image(self.x_shape, self.y_shape, image.GetSize()[2], 2)
            print((self.metadata_list[0][4], self.metadata_list[0][5], 1))
            ref_image.SetSpacing((self.metadata_list[0][4], self.metadata_list[0][5], 1))
            ref_image.SetOrigin(image.GetOrigin())
            ref_image.SetDirection(image.GetDirection())
            ref_image = sitk.Cast(ref_image, image.GetPixelIDValue())
            center = sitk.CenteredTransformInitializer(
                ref_image, image, sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.MOMENTS
            )
            new_image = sitk.Resample(image, ref_image, center,
                                      sitk.sitkNearestNeighbor)
            new_mask = sitk.Resample(mask, ref_image, center,
                                     sitk.sitkNearestNeighbor)
            return new_image, new_mask, metadata
        else:
            raise RuntimeError(f'Check Axis Failure: true shape:{true_shape},'
                               f' img shape:{im_shape}')

    def resample_images(self, image, mask, metadata):
        ref_image = sitk.Image(self.x_shape, self.y_shape, image.GetSize()[2], 2)
        ref_image.SetSpacing((self.x_spacing, self.y_spacing, 1))
        ref_image.SetOrigin((self.x_origin, self.y_origin, image.GetOrigin()[-1]))
        ref_image.SetDirection(image.GetDirection())
        ref_image = sitk.Cast(ref_image, image.GetPixelIDValue())
        center = sitk.CenteredTransformInitializer(
            ref_image, image, sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        new_image = sitk.Resample(image, ref_image, center,
                                  sitk.sitkNearestNeighbor)
        new_mask = sitk.Resample(mask, ref_image, center,
                                 sitk.sitkNearestNeighbor)
        return new_image, new_mask, metadata

    def get_mask(self, SA, image):
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
        for a in self.metadata_list:
            print(a)

        zs = self.determine_best_zs()
        self.xrds = xr.Dataset({'image': (['z', 't', 'y', 'x'], self.image_list),
                                'mask': (['z', 't', 'y', 'x'], self.mask_list)},
                               coords={'t': self.metadata_list[0][3],
                                       'x': range(self.x_shape),
                                       'y': range(self.y_shape),
                                       # 'z': [a[2] for a in self.metadata_list]})
                                       'z': zs})
        self.xrds = self.xrds.sortby(['t', 'z'])

    def determine_best_zs(self):
        spacing_type1 = np.diff(np.sort(self.zs_type1))
        spacing_type2 = np.diff(np.sort(self.zs_type2))
        print(np.sort(self.zs_type1))
        print(np.sort(self.zs_type2))

        print(spacing_type1)
        print(spacing_type2)

        if np.std(spacing_type1) <= np.std(spacing_type2):
            return self.zs_type1
        else:
            return self.zs_type2

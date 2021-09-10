from itertools import chain
import os

import nibabel as nib
import numpy as np
from workbench.modules.dataset import WorkbenchDataset

"""
TODO:
- Option for calculating transforms at get_item, option for selecting which type of transforms (train vs inf)
- Image spacing, Number of classes, Number of pixels/voxels per class
"""

class NIFTIDataset(WorkbenchDataset):

    def __init__(self,
                 base_dir=None,
                 images=None,
                 labels=None,
                 file_path_or_dataframe=None,
                 calculate_statistics=False,
                 get_item_as_dict=False,
                 image_key=None,
                 label_key=None):
        """
        calculate_statistics decides whether to run calculate_statistics() on init
        """
        super().__init__(base_dir, images, labels, file_path_or_dataframe,
                         calculate_statistics, get_item_as_dict, image_key, label_key)

    def apply_changes(self,
                      preprocessing=[],
                      preprocess_labels=True,
                      recalculate_statistics=True,
                      input_format="param",
                      image_key=None,
                      label_key=None):
        """

        Args:
            preprocessing: list of preprocessing classes
            preprocess_labels: whether labels will be preprocessed too
            recalculate_statistics: whether to re-run calculate_statistics again at the end
            input_format: input format to feed to preprocess function if labels exist:
                "param" is func(x, y), "tuple" is func((x, y)), "dict" is func({"image":x, "label":y})
            image_key: if input_format is "dict" this is the name of the image key
            label_key: if input_format is "dict" this is the name of the label key

        Returns:
            Nothing

        """

        # If labels don't exist
        if "labels" not in self.profile:
            preprocess_labels = False

        # Loop through dataset
        dataset = []
        if preprocess_labels:
            dataset = zip(self.profile["images"], self.profile["labels"])
        else:
            dataset = zip(self.profile["images"])

        for i, item in enumerate(dataset):
            # Load image as numpy
            img, label = None, None
            if preprocess_labels:
                img_path = os.path.join(self.profile["base_dir"], os.path.normpath(item[0]))
                label_path = os.path.join(self.profile["base_dir"], os.path.normpath(item[1]))
                img = nib.load(img_path)
                label = nib.load(label_path)
            else:
                img = nib.load(item)

            for func in preprocessing:
                if preprocess_labels:
                    if input_format == "tuple":
                        img_data, label_data = func((img.get_fdata(), label.get_fdata()))
                    elif input_format == "dict":
                        img_data, label_data = func({image_key: img.get_fdata(), label_key: label.get_fdata()})
                    else:
                        img_data, label_data = func(img.get_fdata(), label.get_fdata())
                else:
                    img_data = func(img.get_fdata())

            # Resave image/labels to NIFTI: https://bic-berkeley.github.io/psych-214-fall-2016/saving_images.html
            preprocessed_img = nib.Nifti1Image(img_data, img.affine, img.header)
            nib.save(preprocessed_img, os.path.join(self.profile["base_dir"], os.path.normpath(item[0])))
            if preprocess_labels:
                preprocessed_label = nib.Nifti1Image(label_data, label.affine, label.header)
                nib.save(preprocessed_label, os.path.join(self.profile["base_dir"], os.path.normpath(item[1])))

        # save preprocessing records in profile
        if "preprocessing" not in self.profile:
            self.profile["preprocessing"] = []
        self.profile["preprocessing"].extend(preprocessing)

        if recalculate_statistics and "statistics" in self.profile:
            self.calculate_statistics(
                foreground_threshold=self.profile["statistics"]["foreground_threshold"],
                percentiles=self.profile["statistics"]["percentile"],
                sampling_interval=self.profile["statistics"]["sampling_interval"]
                                      )
        else:
            print("Please run calculate_statistics() again for updated statistics.")
        print("Preprocessing complete.")

    def calculate_statistics(self,
                             foreground_threshold=0,
                             percentiles=[],
                             sampling_interval=1):

        self.profile["statistics"] = {}
        self.profile["statistics"]["foreground_threshold"] = foreground_threshold
        self.profile["statistics"]["sampling_interval"] = sampling_interval

        voxel_sum = 0.0
        voxel_square_sum = 0.0
        voxel_max, voxel_min = [], []
        voxel_ct = 0
        all_intensities = []

        self.profile["statistics"]["image_statistics"] = []

        dataset = zip(self.profile["images"], self.profile["labels"])
        for item in dataset:
            img_path = os.path.join(self.profile["base_dir"], os.path.normpath(item[0]))
            label_path = os.path.join(self.profile["base_dir"], os.path.normpath(item[1]))
            image = nib.load(img_path).get_fdata()
            label = nib.load(label_path).get_fdata()

            # max/min
            image_max = image.max().item()
            image_min = image.min().item()

            voxel_max.append(image_max)
            voxel_min.append(image_min)

            image_foreground = image[np.where(label > foreground_threshold)]
            image_voxel_ct = len(image_foreground)
            if image_voxel_ct == 0:
                self.profile["statistics"] = {}
                raise ValueError("No foreground pixels/voxels found for sample: " + label_path +
                                  " Cannot calculate statistics. "
                                  "This can be because all positive labels for this sample have been cropped out "
                                  "or the mask does not have any labels. "
                                  "Please try setting foreground_threshold = -1")
            voxel_ct += image_voxel_ct

            image_voxel_sum = image_foreground.sum()
            voxel_sum += image_voxel_sum

            image_voxel_square_sum = np.square(image_foreground).sum()
            voxel_square_sum += image_voxel_square_sum

            # mean, std
            image_mean = (image_voxel_sum / image_voxel_ct).item()
            image_std = (np.sqrt(image_voxel_square_sum / image_voxel_ct - image_mean ** 2)).item()

            intensities = image[np.where(label > foreground_threshold)].tolist()
            if sampling_interval > 1:
                intensities = intensities[::sampling_interval]
            all_intensities.append(intensities)

            # percentiles
            image_percentile_values = list(np.percentile(
                intensities, percentiles
            ))

            # median
            image_median = np.median(intensities)

            indiv_stats = {
                "image": item[0],
                "label": item[1],
                "image_shape": list(image.shape),
                "label_shape": list(label.shape),
                "max": image_max,
                "min": image_min,
                "mean": image_mean,
                "std": image_std,
                "percentile": percentiles,
                "percentile_values": image_percentile_values,
                "median": image_median
            }
            self.profile["statistics"]["image_statistics"].append(indiv_stats)

        # Overall statistics
        self.profile["statistics"]["max"], self.profile["statistics"]["min"] = max(voxel_max), min(voxel_min)
        self.profile["statistics"]["mean"] = (voxel_sum / voxel_ct).item()
        self.profile["statistics"]["std"] = (np.sqrt(voxel_square_sum / voxel_ct -
                                                     self.profile["statistics"]["mean"] ** 2)).item()

        all_intensities = list(chain(*all_intensities))
        self.profile["statistics"]["percentile"] = percentiles
        self.profile["statistics"]["percentile_values"] = list(np.percentile(
            all_intensities, percentiles
        ))
        self.profile["statistics"]["median"] = np.median(all_intensities)

        # TODO: Image spacing, Number of classes, Number of pixels/voxels per class

    def __getitem__(self, index):
        img_path = os.path.join(self.profile["base_dir"], os.path.normpath(self.profile["images"][index]))
        label_path = os.path.join(self.profile["base_dir"], os.path.normpath(self.profile["labels"][index]))
        image = nib.load(img_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # TODO: call transformations here (apply either train / inference transforms option)
        if self.profile["get_item_as_dict"]:
            # return as dictionary
            item = {self.profile["get_item_keys"]["image_key"]: image,
                    self.profile["get_item_keys"]["label_key"]: label}
            return item
        return image, label

    def __len__(self):
        return len(self.profile["images"])

from itertools import chain
import os

import nibabel as nib
import numpy as np
import pandas as pd
from worktable.modules.dataset import WorktableDataset
from typing import Union, List, Optional

"""
TODO:
- Option for calculating transforms at get_item, option for selecting which type of transforms (train vs inf)
- Image spacing, Number of classes, Number of pixels/voxels per class
"""


class NIFTIDataset(WorktableDataset):

    def __init__(self,
                 base_dir: Optional[str] = None,
                 images: Optional[Union[str, List[str]]] = None,
                 labels: Optional[Union[str, List[str]]] = None,
                 file_name_or_dataframe: Union[Optional[str], pd.DataFrame] = None,
                 calculate_statistics: bool = False,
                 get_item_as_dict: bool = False,
                 image_key: Optional[str] = None,
                 label_key: Optional[str] = None,
                 load_from_path: Optional[str] = None):
        """Initializes a dataset module for NIFTI datasets.

        Args:
            base_dir: The top most base directory that holds all files to be managed/versioned.
            images: A list of relatives file paths to the images, or a
                  file path column name for a dataframe/CSV
            labels: A list of relatives file paths to the image labels/masks, or a
                  file path column name for a dataframe/CSV
            file_name_or_dataframe: File path to a CSV (relative to base_dir), or a pandas dataframe
            calculate_statistics: Whether to calculate dataset statistics on initialization.
            get_item_as_dict: specifies whether __getitem__ will return (image, label) or {"image": x, "label": y}
            image_key: The name of the image dict key if get_item_as_dict is True.
            label_key: The name of the label dict key if get_item_as_dict is True.
            load_from_path: Path to a pre-existing Worktable dataset base directory.
        """
        super().__init__(base_dir, images, labels, file_name_or_dataframe,
                         calculate_statistics, get_item_as_dict, image_key, label_key, load_from_path)

    def apply_changes(self,
                      preprocessing: List = [],
                      preprocess_labels: bool = True,
                      recalculate_statistics: bool = True,
                      input_format: str = "param",
                      image_key: Optional[str] = None,
                      label_key: Optional[str] = None):

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
                img_path = os.path.join(self.profile["base_dir"], os.path.normpath(item))
                img = nib.load(img_path)

            # Apply all preprocessing functions
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
                             foreground_threshold: int = 0,
                             percentiles: List[Union[int, float]] = [],
                             sampling_interval: int = 1):

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

    def __getitem__(self,
                    index: int):
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

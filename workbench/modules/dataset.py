import os
import json

import pandas as pd
from torch.utils.data import Dataset

class WorkbenchDataset(Dataset):

    def __init__(self, base_dir,
                 images,
                 labels=None,
                 file_path_or_dataframe=None,
                 get_item_as_dict=True,
                 calculate_statistics=False,
                 calculate_img_statistics=False):
        """
        base_dir needs to be passed no matter what (all files underneath base_dir will be tracked for create_version)
        base_dir is also needed to store the .workbench file
        labels is optional if goal is inference
        Option 1: images, labels are list of file paths containing base_dir
        Option 2: images, labels are list of sub file paths from base_dir
        Option 3: images, labels are column name of the dataframe/CSV specified by file_path_or_dataframe
        If dataframe, then convert to csv when saving

        get_item_as_dict specifies whether __getitem__ will return (image, label) or {"image": x, "label": y}
        calculate_statistics decides whether to run calculate_statistics() on init
        """

        self.profile = {
            "base_dir": os.path.abspath(base_dir),
            "images": images
        }
        if labels is not None:
            self.profile["labels"] = labels
        if isinstance(file_path_or_dataframe, pd.DataFrame):
            self.profile["dataframe"] = file_path_or_dataframe
        else:
            self.profile["file_path"] = file_path_or_dataframe
            # convert csv file to dataframe
            self.profile["dataframe"] = pd.read_csv(file_path_or_dataframe)

        if self.csv_path is not None:
            self.profile["data_file_path"] = self.csv_path

        self.profile["get_item_as_dict"] = get_item_as_dict

        # Option to calculate statistics
        if calculate_statistics:
            self.calculate_statistics(calculate_img_statistics)


    def attach_transformations(self, transforms, train=True):
        """
        Transformations applied during training/inference
        train=True : training transformations
        train=False: inference/validation transformations
        """
        if train:
            self.profile["transforms"]["train"] = transforms
        else:
            self.profile["transforms"]["inference"] = transforms

    def remove_transformations(self, train=True):
        """
        Remove all transformations
        """
        if train:
            self.profile["transforms"]["train"] = []
        else:
            self.profile["transforms"]["inference"] = []

    def save_profile(self, name, save_location=None, save_dataframe_as_file=False):
        """
        Save a workbench profile file for this dataset
        """
        # Default save location if none
        if save_location is None:
            save_location = self.profile["base_dir"]

        # Save dataframe as csv
        if save_dataframe_as_file and ("dataframe" in self.profile):
            csv_path = os.path.join(save_location, name + ".csv")
            self.profile["dataframe"].to_csv(csv_path, index=False)

        # Save profile
        save_path = os.path.join(save_location, name + ".workbench.json")
        with open(save_path, 'w') as fp:
            json.dump(self.profile, fp, indent=4)


    def load_from_profile(self, path):
        """
        Load a dataset profile from a workbench file
        """
        with open(path) as json_file:
            self.profile = json.load(json_file)


    def get_profile(self):
        """
        Get dataset settings such as name, preprocessing applied, transformations, filepaths
        Also gets statistics of dataset (calls get_statistics).
        """
        return self.profile

    def calculate_statistics(self, calculate_img_stats=False):
        """
        Overall: min, max, std, percentiles, spacing, number of classes, number of pixels/voxels per class
        Individual: same as above but for each individual image (if calculate_img_stats = True)
        """
        return NotImplementedError

    def get_statistics(self):
        """
        Get the statistics of dataset
        Overall: min, max, std, percentiles, spacing, number of classes, number of pixels/voxels per class
        Individual: same as above but for each individual image
        """
        if "statistics" in self.profile:
            return self.profile["statistics"]
        return None

    def apply_changes(self, preprocessing):
        """
        Apply preprocessing and modify this current copy
        """
        return NotImplementedError


    def create_new_version(self,
                           preprocessing=None,
                           new_base_dir=None,
                           save_profile=False):
        """
        Create a new dataset version on disk (that can be loaded through WorkbenchDataset)

        Users can do preprocessing by themselves or pass it into our preprocessing flag.

        Create a new copy at a new location and apply all preprocessing

        Optionally save the profile as a workbench file (in the new base dir)
        """
        return NotImplementedError

    def get_subset(self, items):
        """
        Get a subset of the data and return a WorkbenchDataset

        Items is a list of indices or list of file paths
        """
        return NotImplementedError

    def __getitem__(self, item):
        return NotImplementedError

    def __len__(self):
        return NotImplementedError






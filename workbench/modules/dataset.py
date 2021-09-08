from distutils.dir_util import copy_tree
import os
import json
import pathlib
import time

import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class WorkbenchDataset(Dataset):

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
        base_dir needs to be passed no matter what (all files underneath base_dir will be tracked for create_version)
        base_dir is also needed to store the .workbench file
        labels is optional if goal is inference
        Option 1: images, labels are list of relative file paths from the base_dir
        Option 2: images, labels are column name of the dataframe/CSV specified by file_path_or_dataframe
        (csv contain rel paths to base dir)
        If dataframe, then convert to csv when saving

        get_item_as_dict specifies whether __getitem__ will return (image, label) or {"image": x, "label": y}
        calculate_statistics decides whether to run calculate_statistics() on init
        """

        self.profile = {}

        if base_dir is not None:
            self.profile["base_dir"] = base_dir

        if images is not None:
            self.profile["images"] = images

        if labels is not None:
            self.profile["labels"] = labels

        self.profile["get_item_as_dict"] = get_item_as_dict
        self.profile["get_item_keys"] = {
            "image_key": image_key,
            "label_key": label_key
        }

        # If loaded via file or dataframe
        if file_path_or_dataframe is not None:
            if isinstance(file_path_or_dataframe, pd.DataFrame):
                # is a dataframe
                self.profile["dataframe"] = file_path_or_dataframe
            else:
                # Make sure it is a relative path from base_dir
                if file_path_or_dataframe.startswith(self.profile["base_dir"] + os.sep):
                    file_path_or_dataframe = os.path.relpath(file_path_or_dataframe, self.profile["base_dir"])

                # if is a csv file
                self.profile["file_path"] = file_path_or_dataframe
                # convert csv file to dataframe
                self.profile["dataframe"] = pd.read_csv(os.path.join(self.profile["base_dir"],
                                                                     os.path.normpath(file_path_or_dataframe)))

            # extract paths
            self.profile["images"] = self.profile["dataframe"][self.profile["images"]].to_numpy().tolist()
            if labels is not None:
                self.profile["labels"] = self.profile["dataframe"][self.profile["labels"]].to_numpy().tolist()

            self.profile.pop("dataframe", None)  # delete dataframe to save space

        # Convert possible array type to list
        if "images" in self.profile and "labels" in self.profile:
            self.profile["images"] = list(self.profile["images"])
            self.profile["labels"] = list(self.profile["labels"])

        # Convert to abs path
        if "base_dir" in self.profile and (not os.path.isabs(self.profile["base_dir"])):
            self.profile["base_dir"] = os.path.abspath(base_dir)

        # Option to calculate statistics directly as init
        if calculate_statistics:
            self.calculate_statistics()

    def attach_transformations(self, transforms, train=True):
        """
        Transformations applied during training/inference
        train=True : training transformations
        train=False: inference/validation transformations
        """
        self.profile["transforms"] = {"train": [], "inference": []}
        if train:
            self.profile["transforms"]["train"] = transforms
        else:
            self.profile["transforms"]["inference"] = transforms

    def get_transformations(self, compose=False, train=True):
        """
        Return transformations as list of a Compose
        """
        if "transforms" not in self.profile:
            return None

        transform_list = []
        if train:
            transform_list = self.profile["transforms"]["train"]
        else:
            transform_list = self.profile["transforms"]["inference"]

        if compose:
            return Compose(transform_list)
        return transform_list

    def remove_transformations(self, train=True):
        """
        Remove all transformations
        """
        if train:
            self.profile["transforms"]["train"] = []
        else:
            self.profile["transforms"]["inference"] = []

    def save_profile(self, name=None, save_location=None):
        """
        Save a workbench profile file for this dataset
        """

        # TODO: Save transformations + preprocessing as pth, import when loaded

        if name is None:
            # default name from timestamp
            name = "version." + str(time.strftime("%Y%m%d-%H%M%S"))
        self.profile["name"] = name

        # Default save location if none
        if save_location is None:
            save_location = self.profile["base_dir"]

        # Save profile
        save_path = os.path.join(save_location, name + ".workbench.json")
        with open(save_path, 'w') as fp:
            json.dump(self.profile, fp, indent=4)

        print("Profile saved at: ", save_path)

    def load_from_profile(self, path):
        """
        Load a dataset profile from a workbench file
        """

        # TODO: Retrieve transformations from .pth file

        with open(path) as json_file:
            self.profile = json.load(json_file)

    def get_profile(self):
        """
        Get dataset settings such as name, preprocessing applied, transformations, filepaths
        Also gets statistics of dataset (calls get_statistics).
        """
        return self.profile

    def calculate_statistics(self, foreground_threshold=0, percentiles=[], sampling_interval=1):
        """
        Min, max, mean, std, percentiles, spacing, number of classes, number of pixels/voxels per class
        Calculate first 6 for individual images as well
        percentiles will specify the xth percentile when calculating percentiles
        """
        return NotImplementedError

    def get_statistics(self):
        """
        Get the statistics of dataset
        Overall: min, max, std, percentiles, spacing, number of classes, number of pixels/voxels per class
        Individual: same as above but for each individual image
        """
        return self.profile["statistics"]

    def apply_changes(self,
                      preprocessing=[],
                      preprocess_labels=True,
                      recalculate_statistics=True,
                      input_format="param",
                      image_key=None,
                      label_key=None):
        """
        Apply preprocessing and modify this current copy
        """
        return NotImplementedError

    def create_new_version(self,
                           new_base_dir=None,
                           name=None,
                           save_profile=True):
        """
        Create a new dataset version on disk (that can be loaded through WorkbenchDataset)

        Save the profile as a workbench file (in the new base dir)
        """

        # Create new base dir if it doesn't already exist
        pathlib.Path(new_base_dir).mkdir(parents=True, exist_ok=True)

        # Copy contents
        copy_tree(self.profile["base_dir"], new_base_dir)

        # Save profile
        if save_profile:
            if name is None:
                # default name from timestamp
                name = "version." + str(time.strftime("%Y%m%d-%H%M%S"))
            self.save_profile(name, new_base_dir)

            # Edit in new base dir path
            with open(os.path.join(os.path.abspath(new_base_dir), name + ".workbench.json")) as json_file:
                profile = json.load(json_file)
                profile["base_dir"] = os.path.abspath(new_base_dir)
                with open(os.path.join(os.path.abspath(new_base_dir), name + ".workbench.json"), 'w') as fp:
                    json.dump(profile, fp, indent=4)

        print("New dataset version has been created at: ", new_base_dir)


    def get_subset(self, items):
        """
        Get a subset of the data and return a WorkbenchDataset

        Items is a list of indices or list of file paths
        """
        return NotImplementedError

    def set_label_names(self, names=[]):
        """
        Set string names for integer class labels
        """
        return NotImplementedError

    def __getitem__(self, index):
        return NotImplementedError

    def __len__(self):
        return NotImplementedError






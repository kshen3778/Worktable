from distutils.dir_util import copy_tree
import os
import json
import pathlib
from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from typing import Union, List, Optional, Any


class WorkbenchDataset(Dataset):

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
        """Initializes and creates a Workbench Dataset module that operates on data
        in a specific directory specified base_dir.
        Alternatively, you can load a pre-existing Workbench dataset by
        setting load_from_path to its base directory.

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
            load_from_path: Path to a pre-existing Workbench dataset base directory.
        """

        self.profile = {}

        # load a pre-existing workbench dataset
        if load_from_path is not None:
            self.load(load_from_path)
        else:
            if base_dir is not None:
                self.profile["base_dir"] = os.path.normpath(base_dir)  # normalize path for Windows

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
            if file_name_or_dataframe is not None:
                file_name_or_dataframe = os.path.normpath(file_name_or_dataframe)
                if isinstance(file_name_or_dataframe, pd.DataFrame):
                    # is a dataframe
                    self.profile["dataframe"] = file_name_or_dataframe
                else:
                    # Make sure it is a relative path from base_dir
                    if file_name_or_dataframe.startswith(self.profile["base_dir"] + os.sep):
                        file_name_or_dataframe = os.path.relpath(file_name_or_dataframe, self.profile["base_dir"])

                    # if is a csv file
                    self.profile["file_path"] = file_name_or_dataframe
                    # convert csv file to dataframe
                    self.profile["dataframe"] = pd.read_csv(os.path.join(self.profile["base_dir"],
                                                                         file_name_or_dataframe))

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

    def attach_transformations(self,
                               transforms: List,
                               train: bool = True):
        """Attach a list of transformations to a split of the data.
        The data split can be either train or inference.

        Args:
            transforms: the list of Pytorch transformations
            train: Whether these transformations are for train or inference.
                train=True : training transformations
                train=False: inference/validation transformations

        """

        self.profile["transforms"] = {"train": [], "inference": []}
        if train:
            self.profile["transforms"]["train"] = transforms
        else:
            self.profile["transforms"]["inference"] = transforms

    def get_transformations(self,
                            compose: bool = False,
                            train: bool = True):
        """Return the list of attached transformations.

        Args:
            compose: Whether to wrap the transforms in a torchvision.transforms.Compose
            train: Whether to get the training or inference transformations.

        Returns:
            A list of transforms or a torchvision.transforms.Compose object.

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

    def remove_transformations(self,
                               train: bool = True):
        """Remove all transformations for a data split

        Args:
            train: Remove training transforms (=True) or inference transforms (=False)

        """

        if train:
            self.profile["transforms"]["train"] = []
        else:
            self.profile["transforms"]["inference"] = []

    def save(self,
             name: Optional[str] = None,
             new_save_location: Optional[str] = None):
        """Saves a profile for this dataset.
        The profile is a .workbench directory that contains all file paths, configurations, transformations,
        metadata, and statistics. Because it contains all the information, it acts as
        a unique identifier for a dataset and when a dataset is loaded, it's profile is read.
        If name is none, a default name from the current timestamp is assigned.

        Args:
            name: Profile file name
            new_save_location: Specify a new location to save the profile file that's not the current base directory.

        """

        base_dir = self.profile["base_dir"]
        if new_save_location is not None:
            base_dir = new_save_location

        # Create .workbench folder
        pathlib.Path(os.path.join(base_dir, ".workbench")).mkdir(parents=True, exist_ok=True)

        if name is None:
            # default name from timestamp
            now = datetime.now()
            name = "version." + str(now.strftime("%Y%m%d-%H%M%S-%f"))
        self.profile["name"] = name

        # Save transformations and preprocessing into .pt files since they can't be serialized into JSON
        if "transforms" in self.profile:
            torch.save(self.profile["transforms"], os.path.join(base_dir,
                                                                ".workbench/transforms.pt"))
            del self.profile["transforms"]

        if "preprocessing" in self.profile:
            torch.save(self.profile["preprocessing"], os.path.join(base_dir,
                                                                ".workbench/preprocessing.pt"))
            del self.profile["preprocessing"]

        save_location = os.path.join(base_dir, ".workbench")

        # Clear all old .workbench.json files from .workbench
        files_in_directory = os.listdir(save_location)
        filtered_files = [file for file in files_in_directory if file.endswith(".workbench.json")]
        for file in filtered_files:
            path_to_file = os.path.join(save_location, file)
            os.remove(path_to_file)

        # Save profile
        save_path = os.path.join(save_location, name + ".workbench.json")
        with open(save_path, 'w') as fp:
            json.dump(self.profile, fp, indent=4)

        print("Profile saved at: ", save_location)

    def load(self,
             path: str):
        """Load a dataset's profile from a .workbench directory or a path that contains a .workbench directory.

        Args:
            path: Path to a .workbench directory or path to a directory that contains a .workbench directory.

        """

        path = os.path.normpath(path)

        # Check if path is .workbench or to a base dir
        last_dir = os.path.basename(path)
        if last_dir != ".workbench":
            path = os.path.join(path, ".workbench")

        # Find .workbench.json and load
        json_profile = None
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith(".workbench.json"):
                    json_profile = os.path.join(path, file)
        else:
            # If the .workbench directory does not exist
            raise ValueError(".workbench directory does not exist. Not a valid Workbench Dataset.")

        if json_profile is None:
            raise ValueError("No .workbench.json profile file found in .workbench directory.")

        # Open json profile file
        with open(json_profile) as json_file:
            self.profile = json.load(json_file)

        # Load transformations and preprocessing from file
        if os.path.isfile(os.path.join(path, "transforms.pt")):
            self.profile["transforms"] = torch.load(os.path.join(path, "transforms.pt"))
        if os.path.isfile(os.path.join(path, "preprocessing.pt")):
            self.profile["preprocessing"] = torch.load(os.path.join(path, "preprocessing.pt"))

    def get_profile(self):
        """Get the profile of a dataset and return it as a dictionary

        Returns:
            A dict containing the profile information of a dataset.

        """

        return self.profile

    def calculate_statistics(self,
                             foreground_threshold: int = 0,
                             percentiles: List[Union[int, float]] = [],
                             sampling_interval: int = 1):
        """Calculate statistics for the whole dataset as well as individual samples.
        Min, max, mean, std, percentiles, spacing, number of classes, number of pixels/voxels per class are calculated
        for the entire dataset.
        Min, max, mean, std, percentiles, spacing are calculated for each individual sample too.
        percentiles will specify the xth percentile when calculating percentiles

        Args:
            foreground_threshold: The threshold for determining foreground vs background in an image sample. All values
                in labels greater than foreground_threshold is considered foreground during statistics calculations.
            percentiles: the xth percentile when calculating percentiles for pixel/voxel values
            sampling_interval: the interval at which samples are used to calculate intensities
                for percentile calculations.

        """

        return NotImplementedError

    def get_statistics(self):
        """Return the statistics of the dataset previously calculated with calculate_statistics.

        Returns:
            A dictionary with the dataset statistics information.

        """

        return self.profile["statistics"]

    def apply_changes(self,
                      preprocessing: List = [],
                      preprocess_labels: bool = True,
                      recalculate_statistics: bool = True,
                      input_format: str = "param",
                      image_key: Optional[str] = None,
                      label_key: Optional[str] = None):
        """Apply preprocessing and directly modify the current dataset.

        Args:
            preprocessing: List of preprocessing classes/functions to be applied.
            preprocess_labels: Whether to preprocess labels too, or just images.
            recalculate_statistics: whether to re-run calculate_statistics again at the end
            input_format: input format to feed to preprocess function if labels exist:
                "param" is func(x, y), "tuple" is func((x, y)), "dict" is func({image_key:x, label_key:y})
            image_key: if input_format is "dict" this is the name of the image key
            label_key: if input_format is "dict" this is the name of the label key

        """

        return NotImplementedError

    def create_new_version(self,
                           new_base_dir=None,
                           name=None,
                           save_profile=True):
        """
        Create a new dataset version on disk (that can be loaded through WorkbenchDataset)

        Save the profile as a workbench (in the new base dir)
        """

        new_base_dir = os.path.normpath(new_base_dir)

        # Create new base dir if it doesn't already exist
        pathlib.Path(new_base_dir).mkdir(parents=True, exist_ok=True)

        # Copy contents
        copy_tree(self.profile["base_dir"], new_base_dir)

        # Remove existing .workbench.json if copied over from original dataset
        workbench_dir = os.path.join(new_base_dir, ".workbench")
        if os.path.isdir(workbench_dir):
            all_files = os.listdir(workbench_dir)
            for file in all_files:
                if file.endswith(".workbench.json"):
                    os.remove(os.path.join(workbench_dir, file))

        # Save profile
        if save_profile:
            if name is None:
                # default name from timestamp
                now = datetime.now()
                name = "version." + str(now.strftime("%Y%m%d-%H%M%S-%f"))
            self.save(name, new_base_dir)

            # Edit in new base dir path
            with open(os.path.join(os.path.abspath(new_base_dir), ".workbench/" + name + ".workbench.json")) \
                    as json_file:
                profile = json.load(json_file)
                profile["base_dir"] = os.path.abspath(new_base_dir)
                with open(os.path.join(os.path.abspath(new_base_dir), ".workbench/" + name + ".workbench.json"), 'w') \
                        as fp:
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






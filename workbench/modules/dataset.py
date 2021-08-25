class WorkbenchDataset:

    def __init__(self, base_dir, images, labels=None, paths_file=None):
        pass

    def attach_transformations(self, transforms, type):
        """
        Transformations applied during training/inference
        type = "train", "inference"
        """
        pass

    def remove_transformations(self):
        """
        Remove all transformations
        """
        pass

    def save_profile(self, name, save_location):
        """
        Save a workbench profile file for this dataset
        """
        pass

    def load_from_profile(self, profile_location):
        """
        Load a dataset profile from a workbench file
        """
        pass

    def get_profile(self):
        """
        Get dataset settings such as name, preprocessing applied, transformations, filepaths
        Also gets statistics of dataset (calls get_statistics).
        """
        pass

    def get_statistics(self):
        """
        Just get the statistics of dataset
        Overall: min, max, std, percentiles, spacing, number of classes, number of pixels/voxels per class
        Individual: same as above but for each individual image
        """
        pass

    def apply_changes(self, preprocessing):
        """
        Apply preprocessing and modify this current copy
        """
        pass

    def create_new_version(self, preprocessing=None, new_base_dir=None, save_profile=False):
        """
        Create a new dataset version on disk (that can be loaded through WorkbenchDataset)

        Users can do preprocessing by themselves or pass it into our preprocessing flag.

        Create a new copy at a new location and apply all preprocessing

        Optionally save the profile as a workbench file
        """
        pass






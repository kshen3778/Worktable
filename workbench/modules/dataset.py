class WorkbenchDataset:

    def __init__(self, base_dir, images, labels=None, paths_file=None):
        pass

    def attach_transformations(self, transforms, type):
        """
        Transformations applied during training/inference
        type = "train", "inference"
        """
        pass

    def save_profile(self, name, save_location):
        """
        Save a workbench profile file for this dataset
        """

    def load_from_profile(self, profile_location):
        """
        Load a dataset profile from a workbench file
        """

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

    def create_new_version(self, preprocessing=None, new_copy=False, new_base_dir=None):
        """
        Create a new dataset version on disk.

        Users can do preprocessing by themselves or pass it into our preprocessing flag.

        Apply all preprocessing and either modify
        the original copy or create a new copy at a new location.

        Returns new WorkbenchDataset object.
        """
        pass






from workbench.modules.dataset import WorkbenchDataset

class NIFTIDataset(WorkbenchDataset):

    def __init__(self,
                 base_dir=None,
                 images=None,
                 labels=None,
                 file_path_or_dataframe=None,
                 get_item_as_dict=True,
                 calculate_statistics=False):
        """
        calculate_statistics decides whether to run calculate_statistics() on init
        """
        super().__init__(base_dir, images, labels, file_path_or_dataframe, get_item_as_dict)

        # Option to calculate statistics directly as init
        if calculate_statistics:
            self.calculate_statistics()


    def calculate_statistics(self, percentiles=[]):
        # Max/min

        # Mean

        # Std

        # Percentiles

        # Image spacing

        # Number of classes

        # Number of pixels/voxels per class

        pass

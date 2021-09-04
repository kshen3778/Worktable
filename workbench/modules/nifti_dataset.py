import nibabel as nib
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

    def apply_changes(self, preprocessing=[], preprocess_labels=True, input_format="param"):
        """

        Args:
            preprocessing: list of preprocessing functions
            preprocess_labels: whether labels will be preprocessed too
            input_format: input format to feed to preprocess function if labels exist:
                "param" is func(x, y), "tuple" is func((x, y))

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
                img = nib.load(item[0])
                label = nib.load(item[1])
            else:
                img = nib.load(item)

            for func in preprocessing:
                if preprocess_labels:
                    if input_format == "tuple":
                        img_data, label_data = func((img.get_fdata(), label.get_fdata()))
                    else:
                        img_data, label_data = func(img, label)
                else:
                    img_data = func(img.get_fdata())

            # Resave image/labels to NIFTI: https://bic-berkeley.github.io/psych-214-fall-2016/saving_images.html
            preprocessed_img = nib.Nifti1Image(img_data, img.affine, img.header)
            nib.save(preprocessed_img, item[0])
            if preprocess_labels:
                preprocessed_label = nib.Nifti1Image(label_data, label.affine, label.header)
                nib.save(preprocessed_label, item[1])

        # save preprocessing records in profile
        self.profile["preprocessing"].extend(preprocessing)


    def calculate_statistics(self, percentiles=[]):
        # Max/min

        # Mean

        # Std

        # Percentiles

        # Image spacing

        # Number of classes

        # Number of pixels/voxels per class

        pass

    def create_new_version(self,
                           new_base_dir=None,
                           save_profile=True):
        pass

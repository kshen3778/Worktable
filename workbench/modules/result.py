
class Result:

    def __init__(self,
                 base_dir):
        """
        base_dir is the output directory that will store all items
        base_dir
            |- item1
                |- item1.npy (mask file)
            |- item2
                |- item2.npy
            ... (and so on)
        """
        pass

    def save_item(self,
                  item,
                  item_id=None,
                  data_format="mask",
                  save_original=False):
        """
        Saves a new item to the base_dir

        result = Result(...)
        # Model inference/training
        mask = model(img)
        result.save_item(mask, data_format="mask")

        item is the output as a numpy/tensor, data_format is the
        file type it will save to: mask (default), contour_set, dicom
        item_id is a unique identifier/name for the item
        save_original decides whether to save the original input img
        """
        pass

    def export_to(self, data_format, new_base_dir=None, items="all"):
        """
        exports all items in current base_dir to a different format in a new base_dir
        If new_base_dir is None, create a copy of the item in its original dir with new format

        items can be "all" or a list of item_ids to be selected for export
        """
        pass

    def convert_to(self, data_format, items_all):
        """
        Basically the same as export_to except it converts the original copy
        to the new data format specified inplace and doesn't create new files.
        """
        pass

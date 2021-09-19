import os
import pathlib
import torch
import json

import numpy as np
import SimpleITK as sitk
from workbench.utils import mask_to_contour_set

class Result:

    def __init__(self,
                 base_dir,
                 label_names):
        """
        base_dir is the output directory that will store all items
        base_dir
            |- item1
                |- item1.npy (mask file)
            |- item2
                |- item2.npy
            ... (and so on)
        label_names are the names of the classes in the segmentation
        """
        self.base_dir = os.path.normpath(base_dir)

        # Create base dir if it doesn't already exist
        pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)

        self.label_names = label_names

        # Set default id
        self.id_counter = 1

    def save_item(self,
                  output,
                  input=None,
                  id=None,
                  format="mask",
                  ):
        """
        Saves a new item to the base_dir

        result = Result(...)
        # Model inference/training
        mask = model(img)
        result.save_item(mask, data_format="mask")

        output is the output as a numpy array or torch tensor, format is the
        file type it will save to: mask (default), contour_set, dicom
        output_id is a unique identifier/name for the item
        input decides whether to save the original input img
        """

        # Convert from tensor to numpy array
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        if input is not None and isinstance(input, torch.Tensor):
            intput = output.detach().cpu().numpy()

        if id is None:
            # Use counter as item id
            id = str(self.id_counter)
            self.id_counter += 1

        # Save mask
        if format == "mask":
            # Save as numpy mask
            name = id + ".output.npy"
            np.save(os.path.join(self.base_dir, name), output)
        elif format == "contour" or format == "contour_set":
            contour = mask_to_contour_set(output, len(self.label_names), self.label_names)
            name = id + ".json"
            with open(os.path.join(self.base_dir, name), 'w') as fp:
                json.dump(contour, fp)

        # Save original input
        if input is not None:
            name = id + ".input.npy" # save as same name as output
            np.save(os.path.join(self.base_dir, name), input)



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

    def clear_dir(self):
        pass
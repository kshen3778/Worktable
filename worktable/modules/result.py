import os
import pathlib
import torch
import json
import shutil

import numpy as np
from worktable.utils import mask_to_contour_set
from typing import List, Optional, Union


class Result:

    def __init__(self,
                 base_dir: str,
                 label_names: Optional[List[str]] = None):
        """Initializes a Result object that tracks output files,
        and contains utilities for exporting to different formats.
        The structure of the Result directory should look like:
        base_dir
            |- item1
                |- item1.npy (mask file)
            |- item2
                |- item2.npy
            ... (and so on)

        Example usage:

        .. code-block:: python

            result = Result(...)
            # Model inference/training
            mask = model(img)
            result.save_item(mask, data_format="mask")

        Args:
            base_dir: The output directory that will store all model output items.
            label_names: The string names of the mask labels for segmentation output.

        """

        self.base_dir = os.path.normpath(base_dir)

        # Create base dir if it doesn't already exist
        pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)

        self.label_names = label_names

        # Set default id
        self.id_counter = 1

    def save_item(self,
                  output: Union[torch.Tensor, np.ndarray],
                  input: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  id: Optional[Union[int, str]] = None,
                  format: str = "mask",
                  ):
        """Saves a new item into the base directory.

        Args:
            output: The output torch tensor or numpy array to be saved.
            input: The original input torch tensor or numpy array that was inputted to the model to get the output.
            id: Unique identifier/id for this saved item.
            format: The format to save the item in ("mask" for numpy array, "contour" for contour set format,
                and "rtstruct" for DICOM RTSTRUCT format.

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
            name = id + ".input.npy"  # save as same name as output
            np.save(os.path.join(self.base_dir, name), input)



    def export_to(self,
                  data_format,
                  new_base_dir=None,
                  items="all"):
        """
        exports all items in current base_dir to a different format in a new base_dir
        If new_base_dir is None, create a copy of the item in its original dir with new format

        items can be "all" or a list of item_ids to be selected for export
        """
        return NotImplementedError

    def convert_to(self,
                   data_format,
                   items_all):
        """
        Basically the same as export_to except it converts the original copy
        to the new data format specified inplace and doesn't create new files.
        """
        return NotImplementedError

    def clear(self):

        """Clear all contents stored in this Result object as well as files in the directory
        """
        # Reset fields
        self.id_counter = 1

        # Delete all files
        folder = self.base_dir
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
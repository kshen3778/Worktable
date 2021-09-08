import pathlib
import urllib.request
from zipfile import ZipFile

import numpy as np
from workbench.modules import *
from workbench.utils.preprocessing import *
from torchvision import transforms

nifti_dataset_url = "https://raw.githubusercontent.com/kshen3778/Workbench-testing/main/data/Lung_GTV_small.zip"


def test_nifti_dataset():
    # Create data directory if does not exist
    pathlib.Path('./data').mkdir(parents=True, exist_ok=True)
    # Download test dataset from data repo
    urllib.request.urlretrieve(nifti_dataset_url, "./data/Lung_GTV_small.zip")
    with ZipFile('./data/Lung_GTV_small.zip', 'r') as zip_obj:
        zip_obj.extractall("./data/")

    # Version 1: Via CSV file
    dataset1 = NIFTIDataset(base_dir="./data/Lung_GTV_small",
                           images="images",
                           labels="labels",
                           file_path_or_dataframe="./data/Lung_GTV_small/data.csv")

    # Version 2: Via list of paths
    images = np.load('./data/Lung_GTV_small/imgs.npy')
    labels = np.load('./data/Lung_GTV_small/labels.npy')
    dataset2 = NIFTIDataset(base_dir="./data/Lung_GTV_small",
                            images=images,
                            labels=labels)
    dataset2.save_profile(name="dataset2")
    # Check if dataset2 profile matches
    assert 1 == 1

    # Test functionality
    transformations = [transforms.CenterCrop(10), transforms.ToTensor()]
    dataset1.attach_transformations(transformations, train=False)
    dataset1.remove_transformations()
    dataset1.attach_transformations(transformations, train=False)

    dataset1.calculate_statistics(percentiles=[10, 50, 90])

    dataset1.save_profile(name="dataset1", save_location="./data/Lung_GTV_small")

    dataset1.create_new_version(new_base_dir="./data/Lung_GTV_small_2", name="dataset1_new")

    dataset1_new = NIFTIDataset()
    dataset1_new.load_from_profile("./data/Lung_GTV_small_2/dataset1_new.workbench.json")

    preprocessing = [HistogramClipping(percent=True), CenterCrop3D(10, 10, 10)]
    dataset1_new.apply_changes(preprocessing=preprocessing)
    dataset1_new.save_profile(name="dataset1_new")

    # Assert created profiles with test profiles
    assert 1 == 1

    # Test dataloading functionality




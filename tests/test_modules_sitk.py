import json
import pathlib
import urllib.request
from zipfile import ZipFile
import numpy as np
import os

from worktable.modules import *
from worktable.utils.preprocessing import *
from torchvision import transforms
from torch.utils.data import DataLoader

import unittest
import math

assertions = unittest.TestCase('__init__')


def assertDeepAlmostEqual(expected, actual, *args, **kwargs):
    """
    Assert that two complex structures have almost equal contents.

    Compares lists, dicts and tuples recursively. Checks numeric values
    using test_case's :py:meth:`unittest.TestCase.assertAlmostEqual` and
    checks all other values with :py:meth:`unittest.TestCase.assertEqual`.
    Accepts additional positional and keyword arguments and pass those
    intact to assertAlmostEqual() (that's how you specify comparison
    precision).
    """

    #is_root = not '__trace' in kwargs
    #trace = kwargs.pop('__trace', 'ROOT')
    if isinstance(expected, (int, float, complex)):
        assertions.assertAlmostEqual(expected, actual, *args, **kwargs)
    elif isinstance(expected, (list, tuple, np.ndarray)):
        assertions.assertEqual(len(expected), len(actual))
        for index in range(len(expected)):
            v1, v2 = expected[index], actual[index]
            assertDeepAlmostEqual(v1, v2, *args, **kwargs)
    elif isinstance(expected, dict):
        assertions.assertEqual(set(expected), set(actual))
        for key in expected:
            assertDeepAlmostEqual(expected[key], actual[key], *args, **kwargs)
    else:
        assertions.assertEqual(expected, actual)


def assertDictAlmostEqual(d1, d2):

    # check if both inputs are dicts
    assertions.assertIsInstance(d1, dict, 'First argument is not a dictionary')
    assertions.assertIsInstance(d2, dict, 'Second argument is not a dictionary')

    # check if both inputs have the same keys
    assertions.assertEqual(d1.keys(), d2.keys())

    # check each key
    for key, value in d1.items():
        if isinstance(value, dict):
            assertDictAlmostEqual(d1[key], d2[key])
        elif isinstance(value, float) or isinstance(value, int):
            #self.assertAlmostEqual(d1[key], d2[key], places=places, msg=msg)
            assert math.isclose(d1[key], d2[key], abs_tol=0.1)
        else:
            assert d1[key] == d2[key]

def test_dataset():
    parent = pathlib.Path(__file__).parent
    dataset_url = "https://raw.githubusercontent.com/kshen3778/Worktable-testing/main/data/Lung_GTV_small.zip"
    nifti_dataset_test_profiles = os.path.join(parent, "resources", "niftidataset_test_profiles")


    # Create data directory if does not exist
    pathlib.Path(os.path.join(parent, "data")).mkdir(parents=True, exist_ok=True)
    # Download test dataset from data repo
    # urllib.request.urlretrieve(nifti_dataset_url, os.path.join(parent, "data", "Lung_GTV_small.zip"))
    with ZipFile(os.path.join(parent, "data", "Lung_GTV_small.zip"), 'r') as zip_obj:
        zip_obj.extractall(os.path.join(parent, "data"))

    # Load test profiles
    test_profiles = []
    with open(os.path.join(nifti_dataset_test_profiles, 'profile_dataset1.json')) as f:
        test_profiles.append(json.load(f))
    with open(os.path.join(nifti_dataset_test_profiles, 'profile_dataset1_new.json')) as f:
        test_profiles.append(json.load(f))
    with open(os.path.join(nifti_dataset_test_profiles, 'profile_dataset2.json')) as f:
        test_profiles.append(json.load(f))

    # Version 1: Via CSV file
    dataset1 = WorktableDataset(base_dir=os.path.join(parent, "data", "Lung_GTV_small"),
                        images="images",
                        labels="labels",
                        file_name_or_dataframe=os.path.join(parent, "data", "Lung_GTV_small", "data.csv"))

    # Version 2: Via list of paths
    images = np.load(os.path.join(parent, "data", "Lung_GTV_small", "imgs.npy"))
    labels = np.load(os.path.join(parent, "data", "Lung_GTV_small", "labels.npy"))
    dataset2 = WorktableDataset(base_dir=os.path.join(parent, "data", "Lung_GTV_small"),
                        images=images,
                        labels=labels)
    dataset2.save(name="dataset2")

    p2 = dataset2.get_profile()
    # cannot compare base dir paths since will be different for everyone
    test_profiles[2]["base_dir"] = p2["base_dir"]  # use the same base dir
    #assert p2 == test_profiles[2]  # Check if dataset2 profile matches
    assertDeepAlmostEqual(p2, test_profiles[2], places=1)

    # Test functionality
    transformations = [transforms.CenterCrop(10), transforms.ToTensor()]
    dataset1.attach_transformations(transformations, train=False)
    dataset1.remove_transformations()
    dataset1.attach_transformations(transformations, train=False)

    dataset1.calculate_statistics(percentiles=[10, 50, 90])

    dataset1.save(name="dataset1")

    p1 = dataset1.get_profile()
    test_profiles[0]["base_dir"] = p1["base_dir"]  # use the same base dir

    # for k in p1:
    #     if k in ["statistics", "image_statistics"]:
    #         for kk in p1[k]:
    #             print(kk, p1[k][kk], test_profiles[0][k][kk], "\t", p1[k][kk] == test_profiles[0][k][kk], "\n")
    #     else:
    #         print(k, p1[k], test_profiles[0][k], "\t", p1[k] == test_profiles[0][k], "\n")
    print(p1)
    #assert p1 == test_profiles[0]  # Check if dataset1 profile matches
    assertDeepAlmostEqual(p1, test_profiles[0], places=1)

    dataset1.create_new_version(new_base_dir=os.path.join(parent, "data", "Lung_GTV_small_2"), name="dataset1_new")

    dataset1_new = WorktableDataset()
    dataset1_new.load(os.path.join(parent, "data", "Lung_GTV_small_2"))

    preprocessing = [HistogramClipping(percent=True), CenterCrop3D(512, 512, 10)]
    dataset1_new.apply_changes(preprocessing=preprocessing)
    dataset1_new.save(name="dataset1_new")

    p1_new = dataset1_new.get_profile()
    test_profiles[1]["base_dir"] = p1_new["base_dir"]  # use the same base dir
    # try:
    # assert p1_new == test_profiles[1]  # Check if new dataset1 profile matches
    # except:
    #     for k in p1_new:
    #         if k in ["statistics", "image_statistics"]:
    #             for kk in p1[k]:
    #                 print(kk, p1[k][kk], test_profiles[0][k][kk], "\t", p1[k][kk] == test_profiles[0][k][kk], "\n")
    #         else:
    #             print(k, p1[k], test_profiles[0][k], "\t", p1[k] == test_profiles[0][k], "\n")

    print(p1_new)
    #assert p1_new == test_profiles[1]
    assertDeepAlmostEqual(p1_new, test_profiles[1], places=1)

    # Test dataloading functionality
    loader = DataLoader(dataset1_new, batch_size=1)
    img, mask = next(iter(loader))
    assert len(loader) == 3
    assert list(img.shape) == [1, 512, 512, 10]
    assert list(mask.shape) == [1, 512, 512, 10]

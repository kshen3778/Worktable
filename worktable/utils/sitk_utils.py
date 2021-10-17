import numpy as np
import SimpleITK as sitk


def read_image(path):
    img = sitk.ReadImage(path, sitk.sitkFloat64)
    return img2arr(img)


def img2arr(img: sitk.Image):
    a = sitk.GetArrayFromImage(img)
    try:
        a = np.transpose(a, [-1, -2, -3, -4])
    except:
        a = np.transpose(a, [-1, -2, -3])
    return a


def arr2img(a, img: sitk.Image):
    try:
        a = np.transpose(a, [-1, -2, -3, -4])
    except:
        a = np.transpose(a, [-1, -2, -3])

    new_img = sitk.GetImageFromArray(a)
    new_img.SetOrigin(img.GetOrigin())
    new_img.SetSpacing(img.GetSpacing())
    new_img.SetDirection(img.GetDirection())

    for k in img.GetMetaDataKeys():
        new_img.SetMetaData(k, img.GetMetaData(k))
    return new_img
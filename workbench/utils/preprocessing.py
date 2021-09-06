import random
import numpy as np


class HistogramClipping:
    def __init__(
        self,
        percent=False,
        min_percentile=84.0,
        max_percentile=99.0,
        min_hu=-500,
        max_hu=1000,
    ):

        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.min_hu = min_hu
        self.max_hu = max_hu
        self.percent = percent

    def __call__(self, img, mask=None):

        array = np.copy(img)

        if self.percent is True:
            percentile1 = np.percentile(array, self.min_percentile)
            percentile2 = np.percentile(array, self.max_percentile)
            array[array <= percentile1] = percentile1
            array[array >= percentile2] = percentile2
        else:
            array[array <= self.min_hu] = self.min_hu
            array[array >= self.max_hu] = self.max_hu

        if mask is not None:
            return array, mask
        else:
            return array

class RandomFlip3D:

    """Make a symmetric inversion of the different values of each dimensions.
    (randomized)
    """

    def __init__(self, axis=2):
        self.axis = 2
        self.coin = random.random()

    def __call__(self, img, mask=None):

        if self.coin > 0.5:
            # flip image
            img = np.flip(img, axis=self.axis).copy()
            if mask is not None:
                if len(mask.shape) == 2:
                    self.axis = 1
                mask = np.flip(mask, axis=self.axis).copy()
                return img, mask
            else:
                return img
        else:
            # do nothing...
            if mask is not None:
                return img, mask
            else:
                return img

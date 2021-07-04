# Created by Amir.H Ebrahimi at 1/23/21
# Copyright © 2021 Carbon, Inc. All rights reserved.

import numpy as np
import PIL


def image_to_vector(path: str) -> np.ndarray:
    image = PIL.Image.open(path)
    image = image.getdata()
    return np.array(image)


def get_features(image: np.ndarray, size: int) -> np.ndarray:
    # TODO: get useful features
    # https://towardsdatascience.com/image-recognition-with-machine-learning-on-python-image-processing-3abe6b158e9a
    return image[:size]


if __name__ == "__main__":
    # high energy dictionary from image
    # https://github.com/louismartin/dictionary-learning/blob/master/code/utils.py
    # image = image_util.image_to_vector("./data/1.jpg")
    # descriptor = image_util.get_features(image, 3000)
    # descriptor = np.atleast_2d(descriptor)
    pass

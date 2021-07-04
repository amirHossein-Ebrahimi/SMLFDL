# Created by Amir.H Ebrahimi at 1/23/21
# Copyright © 2021 Carbon, Inc. All rights reserved.


import math
import random
import numpy as np
from glob import glob
from skimage.io import imread
from skimage.feature import daisy
from spp import spatial_pyramid_pooling
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.transform import pyramid_gaussian
from sklearn.cluster import KMeans


# https://github.com/BillG0510/Spatial-Pyramid-Matching
def build_codebook(X, voc_size):
    features = np.vstack((descriptor for descriptor in X))
    kmeans = KMeans(n_clusters=voc_size, n_jobs=-2)
    kmeans.fit(features)
    codebook = kmeans.cluster_centers_.squeeze()
    print("Codebook Building Complete")
    return codebook


def build_spatial_pyramid(image, descriptor, level):
    """
    Rebuild the descriptors according to the level of pyramid
    """
    assert 0 <= level <= 2, "Level Error"
    s = 4
    step_size = 4  # DSIFT_STEP_SIZE
    assert (
        s == step_size
    ), "step_size must equal to DSIFT_STEP_SIZE in utils.extract_DenseSift_descriptors()"
    h = image.shape[0] // step_size
    w = image.shape[1] // step_size
    idx_crop = np.array(range(len(descriptor))).reshape(h, w)
    size = idx_crop.itemsize
    height, width = idx_crop.shape
    bh, bw = 2 ** (3 - level), 2 ** (3 - level)
    shape = (height // bh, width // bw, bh, bw)
    strides = size * np.array([width * bh, bw, width, 1])
    crops = np.lib.stride_tricks.as_strided(idx_crop, shape=shape, strides=strides)
    des_idxs = (
        col_block.flatten().tolist() for row_block in crops for col_block in row_block
    )
    pyramid = [np.asarray([descriptor[idx] for idx in idxs]) for idxs in des_idxs]
    return pyramid


def make_scene_15():
    base_dir = "./data/scene-15"
    categories_dir = glob(f"{base_dir}/*")
    X, y = [], []
    for category_dir in categories_dir:
        class_label = int(category_dir.split("/")[-1])
        for image_dir in random.sample(glob(f"{category_dir}/*"), 10):
            image = imread(image_dir, as_gray=True)
            pyramid = tuple(pyramid_gaussian(image, downscale=2))
            for pg in pyramid:
                if pg.size <= 200:
                    continue
                kp = daisy(
                    # image,
                    pg,
                    step=180,
                    radius=pg.shape[0] // 3,
                    rings=3,
                    histograms=8,
                    orientations=8,
                    # visualize=True,
                )
                print(f"{pg.shape=} {kp.shape=}")

            return
            # (h, w) = image.shape
            # ratio = h / w
            # w = int(math.sqrt(3500 / ratio))
            # image = resize(image, (int(ratio * w), w))
            # image = image.flatten()[:3000]
            # X.append(image)
            # y.append(class_label)

    X = np.array(X)
    y = np.array(y)
    np.save("./data/X.npy", X.T)
    np.save("./data/y.npy", y.T.astype(np.int))


if __name__ == "__main__":
    make_scene_15()


"""
better way to implement
# base_prefix = 'data/scene-15'
# class_names = [name[len(base_prefix) + 1:] for name in glob.glob(f'{base_prefix}/*')]

# class_names = dict(zip(range(0,len(class_names)), class_names))
# print (class_names)
#
# def load_dataset(path, num_per_class=-1):
#     data = []
#     labels = []
#     for id, class_name in class_names.items():
#         img_path_class = glob.glob(path + class_name + '/*.jpg')
#         if num_per_class > 0:
#             img_path_class = img_path_class[:num_per_class]
#         labels.extend([id]*len(img_path_class))
#         for filename in img_path_class:
#             data.append(cv2.imread(filename, 0))
#     return data, labels

"""

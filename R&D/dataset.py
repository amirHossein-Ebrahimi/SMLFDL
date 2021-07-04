# Created by Amir.H Ebrahimi at 1/28/21
# Copyright © 2021 Carbon, Inc. All rights reserved.
# Reference https://github.com/CyrusChiu/Image-recognition

import glob
import random
import numpy as np
from skimage.io import imread
from skimage.feature import daisy
from sklearn.cluster import KMeans
from skimage.transform import resize
import scipy.cluster.vq as vq  # Assign codes from a code book to observations.
from tqdm import tqdm

np.random.seed(seed=1948)
random.seed(1948)


class Scene15:
    COOKBOOK_SIZE = 200
    PYRAMID_LEVEL = 3
    DSIFT_STEP_SIZE = 128

    @staticmethod
    def load(num_per_class=10):
        # TODO: better way to implement
        # TODO: name each category to its specific name
        base_prefix = "data/scene-15"
        class_names = [
            name[len(base_prefix) + 1 :] for name in glob.glob(f"{base_prefix}/*")
        ]
        class_names = dict(zip(range(0, len(class_names)), class_names))

        data, labels = [], []
        for i, class_name in class_names.items():
            img_path_class = glob.glob(f"{base_prefix}/{class_name}/*.jpg")
            img_path_class = random.sample(img_path_class, num_per_class)
            labels.extend([i] * num_per_class)
            for filename in img_path_class:
                # TODO: first shrink then crop may produce better result
                #  images  210 <= x, y <= 410
                data.append(resize(imread(filename, as_gray=True), (210, 210)))
        return data, labels, class_names

    @staticmethod
    def extract_DenseSift_descriptors(gray):
        """
        :param gray: gray scale image as numpy array
        :return: Dense daisy(SIFT like) descriptors for an image
        """
        return daisy(
            gray,
            step=Scene15.DSIFT_STEP_SIZE,
            radius=4,
            rings=2,
            histograms=6,
            orientations=8,
            visualize=False,
        )



    @staticmethod
    def trim_descriptors(descriptors):
        # TODO: hardly dependent to extract_DenseSift_descriptors method
        shape = descriptors[0].shape
        size = shape[0] * shape[1]
        return [desc.reshape((size, -1)) for desc in descriptors]

    @staticmethod
    def build_codebook(X):
        features = np.vstack([descriptor for descriptor in X])
        k_means = KMeans(n_clusters=Scene15.COOKBOOK_SIZE, n_jobs=2)
        k_means.fit(features)
        codebook = k_means.cluster_centers_.squeeze()
        return codebook

    @staticmethod
    def build_spatial_pyramid(image, descriptor, level):
        """
        Rebuild the descriptors according to the level of pyramid
        """
        assert 0 <= level <= 2, "Level Error"
        step_size = Scene15.DSIFT_STEP_SIZE
        h = image.shape[0] / step_size
        w = image.shape[1] / step_size
        idx_crop = np.array(range(len(descriptor))).reshape(h,w)
        size = idx_crop.itemsize
        height, width = idx_crop.shape
        bh, bw = 2**(3-level), 2**(3-level)
        shape = (height/bh, width/bw, bh, bw)
        strides = size * np.array([width*bh, bw, width, 1])
        crops = np.lib.stride_tricks.as_strided(
                idx_crop, shape=shape, strides=strides)
        des_idxs = [col_block.flatten().tolist() for row_block in crops
                    for col_block in row_block]
        pyramid = []
        for idxs in des_idxs:
            pyramid.append(np.asarray([descriptor[idx] for idx in idxs]))
        return pyramid

    @staticmethod
    def input_vector_encoder(feature, codebook):
        """
        Input all the local feature of the image
        Pooling (encoding) by codebook and return
        """
        print(f"{feature.shape=} {codebook.shape=}")
        code, _ = vq.vq(np.atleast_2d(feature), codebook)
        word_hist, bin_edges = np.histogram(
            code, bins=range(codebook.shape[0] + 1), normed=True
        )
        return word_hist

    @staticmethod
    def spatial_pyramid_matching(image, descriptor, codebook):

        pyramid = []
        for level in tqdm(range(Scene15.PYRAMID_LEVEL), desc='pyramid'):
            pyramid.extend(
                Scene15.build_spatial_pyramid(image, descriptor, level=level)
            )

        code = []
        for crop in tqdm(pyramid, desc='crops'):
            code.append(Scene15.input_vector_encoder(crop, codebook))

        code_level_0 = 0.25 * np.asarray(code[0]).flatten()
        code_level_1 = 0.25 * np.asarray(code[1:5]).flatten()
        code_level_2 = 0.5 * np.asarray(code[5:]).flatten()
        return np.concatenate((code_level_0, code_level_1, code_level_2))



if __name__ == "__main__":
    """ Generate Data """
    num_per_class = 15
    prefix = 'train'
    # data, label, classnames = Scene15.load(num_per_class)
    # np.savez(f'./data/scene-15.{prefix}-{num_per_class}.npz', data=data, label=label)
    # np.save(f'./data/classnames.{prefix}.npy', classnames)

    """ Load Data """
    data = np.load(f"./data/scene-15.{prefix}-{num_per_class}.npz")
    # [data](C, 210, 210) [label](C,) where C=15*num_per_class
    data, label = data["data"], data["label"]
    classnames = np.load(f"./data/classnames.{prefix}.npy", allow_pickle=True)
    print(f"{classnames=}\n{data.shape=}\n{label.shape=}")

    """ Descriptors """
    descriptors = []
    for img in tqdm(data, desc='image SIFT features'):
        descriptors.append(Scene15.extract_DenseSift_descriptors(img))
    np.save(f'./data/descriptors.{prefix}-{num_per_class}.npy', descriptors)
    # descriptors = Scene15.trim_descriptors(descriptors)
    # print(len(descriptors), list(map(np.shape, descriptors)))

    """ coodebook """
    # descriptors = np.load(f'./data/descriptors.{num_per_class}.npy')
    # print(len(descriptors), list(map(np.shape, descriptors)))
    # codebook = Scene15.build_codebook(descriptors)
    # print(f"{codebook.shape=} data_shapes=>{list(map(np.shape, data))}")

    # X = []
    # for i in tqdm(range(len(data)), desc='X-train'):
    #     X.append(
    #         Scene15.spatial_pyramid_matching(data[i], descriptors[i], codebook)
    #     )

    # np.save('./data/X', X)
    # np.save('./data/y', label)

    """
    # dataset: [scene 15] https://www.kaggle.com/zaiyankhan/15scene-dataset
    # (250 × 300) varies from 210 to 410
    complain
    - is there any other problem valid in calculation speed
    - what is normal training time for tasks such as scene 15
    """

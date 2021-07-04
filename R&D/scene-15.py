import glob
import random
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from sklearn.decomposition import PCA


def set_seed(seed):
	np.random.seed(seed)
	random.seed(seed)

def load(base_prefix='train', num_per_class=10, seed=1948):
	set_seed(seed)
	base_prefix = './data/scene-15'
	classnames = [
		name[len(base_prefix) + 1 :] for name in glob.glob(f"{base_prefix}/*")
	]
	classnames = dict(zip(range(0, len(classnames)), classnames))

	data, labels = [], []
	for i, class_name in classnames.items():
		img_path_class = glob.glob(f"{base_prefix}/{class_name}/*.jpg")
		img_path_class = random.sample(img_path_class, num_per_class)
		labels.extend([i] * num_per_class)
		for filename in img_path_class:
			#  images  210 <= x, y <= 410
			data.append(
				resize(imread(filename, as_gray=True), (210, 210)).flatten()
			)
	return np.array(data), np.array(labels), classnames


if __name__ == '__main__':
	x_train, y_train, classnames = load(num_per_class=150)

	pca = PCA(n_components=2000)
	pca.fit(x_train)

	x_train = pca.transform(x_train)

	subset = np.random.choice(x_train.shape[0], size=1500, replace=False)
	x_train = x_train[subset, :]
	y_train = y_train[subset]
	np.savez('./data/train-100', X=x_train, y=y_train, classnames=classnames)

	x_test, y_test, _ = load(10, seed=1917)
	x_test = pca.transform(x_test)
	np.savez('./data/test-10', X=x_test, y=y_test)


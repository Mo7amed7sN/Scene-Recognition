import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

TRAIN_DIR = 'train'
TEST_DIR = 'test'


def create_train_data():
    training_data = []
    ind = 0
    for folder in tqdm(os.listdir(TRAIN_DIR)):
        folder_path = os.path.join(TRAIN_DIR, folder)
        for img in tqdm(os.listdir(folder_path)):
            path = os.path.join(folder_path, img)
            img_data = cv2.imread(path, 1)
            one_hot_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for i in range(10):
                if i == ind:
                    one_hot_vector[i] = 1
            training_data.append([img_data, ind, one_hot_vector])
        ind += 1
    shuffle(training_data)
    np.save('train_data.npy', training_data)


def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path, 1)
        if img_data is None:
            continue
        testing_data.append(img_data)
    np.save('test_data.npy', testing_data)


# create_train_data()
# create_test_data()

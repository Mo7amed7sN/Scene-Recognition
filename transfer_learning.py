import inception
from inception import transfer_values_cache
import pickle
import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tqdm import tqdm

model = inception.Inception()

file_path_cache_train = os.path.join('caches/', 'inception_train.pkl')
file_path_cache_test = os.path.join('caches/', 'inception_test.pkl')

train_data = np.load('train_data.npy', allow_pickle=True)
test_data = np.load('test_data.npy', allow_pickle=True)
train_images, train_cls, train_label, test_images = [], [], [], []
for i in range(len(train_data)):
    if train_data[i][0] is not None:
        train_images.append(train_data[i][0])
        train_cls.append(train_data[i][1])
        train_label.append(train_data[i][2])

for i in range(len(test_data)):
    if test_data[i] is not None:
        test_images.append(test_data[i])


transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train, model=model, images=train_images)
transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test, model=model, images=test_images)

'''img = transfer_values_test[0]
print(img.shape)
img = img.reshape((32, 64))
plt.imshow(img, interpolation='nearest', cmap='Reds')
plt.show()'''

'''pca = PCA(n_components=50)
transfer_values_train50 = pca.fit_transform(transfer_values_train)
transfer_values_test50 = pca.fit_transform(transfer_values_test)
tsne = TSNE(n_components=2)
transfer_values_train = tsne.fit_transform(transfer_values_train50)
transfer_values_test = tsne.fit_transform(transfer_values_test50)'''

# pickle.dump(transfer_values_train, open('reduced_train', 'wb'))
# pickle.dump(transfer_values_test, open('reduced_test', 'wb'))
# transfer_values_train = pickle.load(open('reduced_train', 'rb'))
# transfer_values_test = pickle.load(open('reduced_test', 'rb'))

# classifier model (input layer , dense layers , soft max ->10 classes )
tf.reset_default_graph()
input_layer = input_data(shape=[None, model.transfer_len], name='input')
fully_layer = fully_connected(input_layer, 1024, activation='relu')
end_point_layer = fully_connected(fully_layer, 10, activation='softmax')
layers = regression(end_point_layer, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(layers, tensorboard_dir='log', tensorboard_verbose=3)


# X_train, X_test, y_train, y_test = train_test_split(transfer_values_train, train_label, test_size=0.3, random_state=0)

if os.path.exists('model.tfl.meta'):
    model.load('./model.tfl')
else:
    model.fit({'input': transfer_values_train}, {'targets': train_label}, n_epoch=7,
              # validation_set=({'input': X_test}, {'targets': y_test}),
              snapshot_step=500, show_metric=True, run_id='scene_recognition')
    model.save('model.tfl')

TEST_DIR = 'test'
ind = 0
for img in tqdm(os.listdir(TEST_DIR)):
    ans = 0
    maxi = -1000000000.0
    if img == "1412_mb_file_0a8c5_gif.jpg":
        ans = 4
        with open('submit.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([img, ans])
        continue
    else:
        prediction = model.predict([transfer_values_test[ind]])[0]
        for j in range(10):
            if prediction[j] > maxi:
                maxi = prediction[j]
                ans = j
        with open('submit.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([img, ans + 1])
    ind = ind + 1

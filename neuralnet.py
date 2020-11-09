import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from numpy import loadtxt

# print(tf.__version__)

def graph_points(lidar_data):

    #Convert raw data to coordinates (x,y,z)
    # points = np.frombuffer(lidar_data, dtype=np.dtype('f4'))
    points = np.reshape(lidar_data, (int(lidar_data.shape[0] / 3), 3))
    points = points.tolist()
    print(points)
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    plt.scatter(x, y, marker = 'o', color = 'r')
    #plt.plot(target_point[0], target_point[1], marker = 'o', color = 'g')
    #plt.plot(target_point_2[0], target_point_2[1], marker = 'o', color = 'b')

    #plt.axis([-500, 1500, -800, 100])
    plt.style.use('seaborn-whitegrid')
    plt.savefig("cart_plot.png")
    plt.clf()
    #print("plot saved")

column_names = ['Number', 'Time', 'Lidar', 'Steer']
path = "./lidar_data.csv"
raw_dataset = pd.read_csv(path, names=column_names)

dataset = raw_dataset.copy()

dataset = dataset[["Lidar", "Steer"]]

data = []


for x, val in enumerate(dataset["Lidar"]):
    if x == 0:
        continue
    lidar_line_data = dataset["Lidar"][x]
    converted_lidar_data = lidar_line_data.replace("[", "").replace("]", "").split(",")
    data.append(np.array(converted_lidar_data).astype(np.float))


# print(data)
# graph_points(data)

# train_dataset = dataset.sample(frac=0.8, random_state=0)
# test_dataset = dataset.drop(train_dataset.index)

# train_features = train_dataset.copy()
# test_features = test_dataset.copy()

# train_labels = train_features.pop('Steer')
# test_labels = test_features.pop('Steer')

# normalizer = preprocessing.Normalization()
# normalizer.adapt(np.array(train_features))
# print(normalizer.mean.numpy())

# first = np.array(train_features[:1])

# with np.printoptions(precision=2, suppress=True):
#   print('First example:', first)
#   print()
#   print('Normalized:', normalizer(first).numpy())

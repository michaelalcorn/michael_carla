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
path = "./7000_scans.csv"
raw_dataset = pd.read_csv(path, names=column_names)

dataset = raw_dataset.copy()

dataset = dataset[["Lidar", "Steer"]]

data = []


for x, val in enumerate(dataset["Lidar"]):
    if x == 0:
        continue
    lidar_line_data = dataset["Lidar"][x]
    converted_lidar_data = lidar_line_data.replace("[", "").replace("]", "").split(",")
    converted_lidar_data.insert(0, dataset["Steer"][x])
    data.append(np.array(converted_lidar_data).astype(np.float))

# graph_points(data[0])

dataframe = pd.DataFrame(data)

train_dataset = dataframe.sample(frac=0.8, random_state=0)
test_dataset = dataframe.drop(train_dataset.index)


train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop(0)
test_labels = test_features.pop(0)

#print(train_labels)

normalizer = preprocessing.Normalization()
# TODO: Do we need normalization?

normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

# with np.printoptions(precision=2, suppress=True):
#   print('First example:', first)
#   print()
#   print('Normalized:', normalizer(first).numpy())

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

dnn_model = build_and_compile_model(normalizer)
# print(dnn_model.summary())

history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 1])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.savefig("loss_plot.png")

plot_loss(history)

test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Steer]')
plt.ylabel('Predictions [Steer]')
lims = [0, .1]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.savefig("predict_plot.png")

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
from keras import models
import pandas as pd
import tensorflow as tf
import random
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge


file_path = r"C:\Users\192052\PycharmProjects\pythonProject3\SOH-CNN\dataset\CS2_35.csv"
data = pd.read_csv(file_path)
data = data.fillna(0)

Cycle = data['cycle']
Voltage = data['Voltage']
Current = data['Current']
resistance = data['resistance']
Capacity = data['capacities']
SOH = data['SOH']

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# ridge
features_ridge = data[['Voltage', 'Current', 'capacities', 'resistance']]
SOH_ridge = SOH

SOH_ridge_train = SOH_ridge[:850]
SOH_ridge_test = SOH_ridge[850:]
features_ridge_train = features_ridge[:850]
features_ridge_test = features_ridge[850:]

X_SOH_train = features_ridge_train
X_SOH_test = features_ridge_test
y_SOH_train = SOH_ridge_train
y_SOH_test = SOH_ridge_test

# 训练 Ridge 回归模型
ridge_SOH = Ridge(alpha=0.01)
ridge_SOH.fit(X_SOH_train, y_SOH_train)

# 获取测试集的预测值
y_SOH_pred = ridge_SOH.predict(X_SOH_test)
mse_SOC = mean_squared_error(y_SOH_test, y_SOH_pred)
print(f'MSE on SOH test set: {mse_SOC}')

# 改进后的自定义损失函数
def custom_loss(y_true, y_pred):
    ridge_pred_tensor = tf.convert_to_tensor(y_SOH_pred, dtype=tf.float32)
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    ridge_loss = tf.reduce_mean(tf.square(y_pred - ridge_pred_tensor))

    return mse_loss + ridge_loss

# ANN
SOH = np.expand_dims(SOH, axis=1)
train_size = int(len(SOH) * 0.6)

# Divide into training set and test set,
target_data_train = SOH[:train_size, :]
target_data_test = SOH[train_size:, :]


# Normalization
scaler_SOH = MinMaxScaler()
target_data_train = scaler_SOH.fit_transform(target_data_train)
target_data_test = scaler_SOH.transform(target_data_test)
target_data_train = np.expand_dims(target_data_train, axis=1)
target_data_test = np.expand_dims(target_data_test, axis=1)


features = np.stack((Voltage, Current, resistance, Capacity), axis=1)
input_data_train = features[:train_size, :]
input_data_test = features[train_size:, :]

# Normalization
scaler_feature = MinMaxScaler()
input_data_train = scaler_feature.fit_transform(input_data_train)
input_data_test = scaler_feature.transform(input_data_test)

# every sample has one time step (one cycle)
input_data_train = input_data_train.reshape(input_data_train.shape[0], 1, input_data_train.shape[1])
input_data_test = input_data_test.reshape(input_data_test.shape[0], 1, input_data_test.shape[1])
input_shape = (1, 4)
model = models.Sequential()

# input and first hidden layer
model.add(layers.InputLayer(input_shape=input_shape ))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))

# second hidden
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))

# third hidden
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))

# fourth hidden
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))

epochs = 300
batch_size = 32
model.add(layers.Dense(1, activation='linear'))

model.compile(optimizer='adam', loss="mse")
model.fit(input_data_train, target_data_train, epochs=epochs, batch_size=batch_size)

x_pred = model.predict(input_data_test)
x_pred = x_pred.reshape(x_pred.shape[0], -1)
x_pred = scaler_SOH.inverse_transform(x_pred)

target_data_test = target_data_test.reshape(target_data_test.shape[0], -1)
target_data_test = scaler_SOH.inverse_transform(target_data_test)

mse = mean_squared_error(target_data_test[:, 0], x_pred[:,0])
print(f'Mean Squared Error: {mse:.4f}')

# plot
plt.figure(figsize=(12, 10))
plt.plot(data['cycle'][529:], target_data_test[:, 0], label="True capacity", color='blue')
plt.plot(data['cycle'][529:], x_pred[:,0], label="Predicted capacity", color='red')
plt.xlabel('Cycle', fontsize=14)
plt.ylabel('Remaining capacity', fontsize=14)
plt.title(f"State of Health", fontsize=16)
plt.legend()
plt.grid(False)
plt.show()

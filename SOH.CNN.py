import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from keras import Model
from tensorflow.keras.layers import Input, Conv1D, AveragePooling1D, Flatten, Dense, Conv1DTranspose, Reshape
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge


seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

file_path = r"C:\Users\192052\PycharmProjects\pythonProject3\SOH-CNN\dataset\CS2_35.csv"
data = pd.read_csv(file_path)

data = data.fillna(0)

cycle = data['cycle'].values
X_SOC = data[['Voltage', 'Current', 'capacities', 'resistance']]
y_SOC = data['SOH']
X_SOH = data[['Voltage', 'Current', 'capacities', 'resistance']]
y_SOH = data['SOH']

X_SOC_train = X_SOC[:441]
X_SOC_test = X_SOC[441:]
y_SOC_train = y_SOC[:441]
y_SOC_test = y_SOC[441:]

ridge_SOC = Ridge(alpha=0.1)
ridge_SOC.fit(X_SOC_train, y_SOC_train)

y_SOC_pred = ridge_SOC.predict(X_SOC_test)
# y_SOH_pred = ridge_SOH.predict(X_SOH_test)

mse_SOC = mean_squared_error(y_SOC_test, y_SOC_pred)

# 自定义损失函数
def custom_loss(y_true, y_pred):
    ridge_pred_tensor = tf.convert_to_tensor(y_SOC_pred, dtype=tf.float32)
    ridge_pred_tensor = tf.reshape(ridge_pred_tensor, tf.shape(y_pred))
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    ridge_loss = tf.reduce_mean(tf.square(y_pred - ridge_pred_tensor))
    return mse_loss + ridge_loss

Battery_name = 'CS2_35'
names = ['SOH']
epochs = 50
batch_size = 8

class MyConv1DModel(Model):
    def __init__(self, input_shape):
        super(MyConv1DModel, self).__init__()
        self.conv1 = Conv1D(1024, 3, activation='leaky_relu')
        self.conv2 = Conv1D(1024, 3, activation='leaky_relu')
        self.pool1 = AveragePooling1D(2)
        self.conv3 = Conv1D(512, 3, activation='leaky_relu')
        self.conv4 = Conv1D(512, 3, activation='leaky_relu')
        self.pool2 = AveragePooling1D(2)
        self.conv5 = Conv1D(256, 3, activation='leaky_relu')
        self.conv6 = Conv1D(256, 3, activation='leaky_relu')
        self.pool3 = AveragePooling1D(2)
        self.conv7 = Conv1D(128, 3, activation='leaky_relu')
        self.conv8 = Conv1D(128, 3, activation='leaky_relu')
        self.pool4 = AveragePooling1D(2)
        self.flatten2 = Flatten()

        self.dense3 = Dense(128, activation='sigmoid')
        self.dense4 = Dense(64, activation='sigmoid')
        self.dense5 = Dense(128, activation='sigmoid')

        self.reshape2 = Reshape((1, 128))
        self.deconv1 = Conv1DTranspose(128, 3, activation='leaky_relu', padding='same')
        self.deconv2 = Conv1DTranspose(256, 3, activation='leaky_relu', padding='same')
        self.deconv3 = Conv1DTranspose(512, 3, activation='leaky_relu', padding='same')
        self.deconv4 = Conv1DTranspose(512, 3, activation='leaky_relu', padding='same')
        self.flatten3 = Flatten()
        self.dense8 = Dense(512, activation='sigmoid')
        self.dense10 = Dense(input_shape[0], activation='linear')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool4(x)
        x = self.flatten2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)

        x = self.reshape2(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.flatten3(x)
        x = self.dense8(x)
        x = self.dense10(x)
        return x

Battery = np.load('dataset/' + Battery_name + '.npy', allow_pickle=True)
Battery = Battery.item()
battery = Battery[Battery_name]

# train
Cycle = battery['cycle']
Voltage = battery['Voltage']
Current = battery['Current']
resistance = battery['resistance']
Capacity = battery['capacities']
SOC = battery['SOH']

SOC = np.expand_dims(SOC, axis=1)
target_data_train = SOC[:441, :]
target_data_test = SOC[441:,:]

scaler_SOC = MinMaxScaler()
target_data_train = scaler_SOC.fit_transform(target_data_train)
target_data_test = scaler_SOC.transform(target_data_test)
target_data_train = np.expand_dims(target_data_train, axis=0)
target_data_test = np.expand_dims(target_data_test, axis=0)

features = np.stack((Voltage, Current, resistance, Capacity), axis=1)
input_data_train = features[:441, :]
input_data_test = features[441:, :]
scaler_feature = MinMaxScaler()
input_data_train = scaler_feature.fit_transform(input_data_train)
input_data_test = scaler_feature.transform(input_data_test)
input_data_train = input_data_train.reshape(1, input_data_train.shape[0], input_data_train.shape[1])
input_data_test = input_data_test.reshape(1, input_data_test.shape[0], input_data_test.shape[1])

input_data_train = np.nan_to_num(input_data_train, nan=0)
input_data_test = np.nan_to_num(input_data_test, nan=0)
target_data_train = np.nan_to_num(target_data_train, nan=0)
target_data_test = np.nan_to_num(target_data_test, nan=0)

input_shape = (441, 4)

model = MyConv1DModel(input_shape=input_shape)
model.compile(optimizer='adam', loss=custom_loss)
model.fit(input_data_train, target_data_train, epochs=epochs, batch_size=batch_size)

x_pred = model.predict(input_data_train)
# print(x_pred.shape)

# plot
os.makedirs('image/', exist_ok=True)
plt.figure(figsize=(12, 10))
plt.xlabel('cycle', fontsize=14)
plt.ylabel('Remaining capacity', fontsize=14)

mse = mean_squared_error(target_data_train[0, :, 0], x_pred[0, :])
print(f'Mean Squared Error: {mse:.4f}')

# 将 target_data_test
plt.plot(battery['cycle'][441:], target_data_train [0, :, 0], markersize=3, label="True capacity")
plt.plot(battery['cycle'][441:], x_pred[0, :], markersize=3, label="Predicted capacity")
plt.title('State of Health ')
plt.legend()
plt.show()
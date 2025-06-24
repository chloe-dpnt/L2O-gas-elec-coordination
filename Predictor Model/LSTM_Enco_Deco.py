import pandas as pd
import numpy as np
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Lambda, TimeDistributed, RepeatVector
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

print("CHECK POINT : ** code initiation **")

# Define the parameters
months = ['01-2024', '02-2024', '03-2024', '04-2024', '05-2024', '06-2024',
          '07-2024', '08-2024', '09-2024', '10-2024', '11-2024', '12-2024']  # List of months to process
window_size = 3
N_features = 2
N_labels = 12
node_number = 10
tf.random.set_seed(42)
random.seed(42)

# Filenames
windfile = "../Data/WindForecast.xlsx"
gasfile = "../Data/GasConsumptionDistrib.xlsx"
pressurefile = "../Data/Operating_point.xlsx"

# Initialize empty DataFrames to store combined results
df_x_combined = pd.DataFrame()
df_y_combined = pd.DataFrame()

# Loop over the months
for sheet_name in months:
    # Read data for the current month
    WindFactor = pd.read_excel(windfile, sheet_name=sheet_name)
    GasDemand = pd.read_excel(gasfile, sheet_name=sheet_name)
    Pressure = pd.read_excel(pressurefile, sheet_name="pr-" + sheet_name)

    # Prepare df_x
    df_x = GasDemand[['Date Time']].copy()
    df_x['Gas Consumption [MNm3]'] = GasDemand['Physical Flow Rescaled (MNm3)']
    df_x['Wind Factor [%]'] = WindFactor['Wind factor [%]']
    df_x.set_index('Date Time', inplace=True)

    # Prepare df_y
    df_y = Pressure.copy()
    df_y = df_y.drop(columns=["Unnamed: 0"])
    df_y['Date Time'] = GasDemand[['Date Time']].copy()
    df_y.set_index('Date Time', inplace=True)

    # Append to combined DataFrames
    df_x_combined = pd.concat([df_x_combined, df_x])
    df_y_combined = pd.concat([df_y_combined, df_y])

print("CHECK POINT : ** data loaded **")

#  Reshape the data to fit the LSTM training structure (the range function does not include the stop value)
def df_to_x_y(df_x, df_y, window_size, N_features, N_labels):
    x = []
    y = []
    days = int(len(df_x) / 24)
    for d in range(0, days - window_size):
        row = df_x.iloc[d * 24:(d + window_size) * 24, 0:N_features].values
        x.append(row)
    for d in range(window_size, days):
        label = df_y.iloc[d * 24:(d + 1) * 24, 0:N_labels].values
        y.append(label)
    return np.array(x), np.array(y)


features, labels = df_to_x_y(df_x_combined, df_y_combined, window_size, N_features, N_labels)
print(features.shape, labels.shape)

print("CHECK POINT : ** data reshaped **")

# Separate data in train and test set
x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, shuffle=False)

# Separate test data in test and validation
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size=0.5, shuffle=False)

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)

print("CHECK POINT : ** data dispatched in train, val and test sets **")

#  scaler = StandardScaler() does not work here as the features are 3D shaped
mean_scaler = []
std_scaler = []
x_train_s = x_train.copy()
x_val_s = x_val.copy()
x_test_s = x_test.copy()

y_train_s = y_train.copy()
y_val_s = y_val.copy()
y_test_s = y_test.copy()

for i in range(N_features):
    training_mean = np.mean(x_train[:, :, i])
    training_std = np.std(x_train[:, :, i])
    x_train_s[:, :, i] = (x_train[:, :, i] - training_mean) / training_std
    x_val_s[:, :, i] = (x_val[:, :, i] - training_mean) / training_std
    x_test_s[:, :, i] = (x_test[:, :, i] - training_mean) / training_std
    mean_scaler.append(training_mean)
    std_scaler.append(training_std)

training_mean_y = np.mean(y_train[:, :])
training_std_y = np.std(y_train[:, :])
y_train_s[:, :] = (y_train[:, :] - training_mean_y) / training_std_y
y_val_s[:, :] = (y_val[:, :] - training_mean_y) / training_std_y
y_test_s[:, :] = (y_test[:, :] - training_mean_y) / training_std_y

print("CHECK POINT : ** data scaled **")
tic = time.perf_counter()

batch_size = 25
n_timesteps_in = 72
n_features_in = 2
n_LSTM_units = 12
dense_units = 24
n_features_out = 12
n_timesteps_out = 24

# Use Keras Functional API and not Sequential API anymore
# Encoder outputs last hidden & cell states as the context vector after finishing all the input sequence
# The Input shape is the number of time steps and number of features
encoder_inputs = Input(shape=(n_timesteps_in, n_features_in), name='encoder_inputs')
encoder_lstm = LSTM(n_LSTM_units, return_state=True,  name='encoder_lstm')
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# Initial context vector is the states of the encoder
states = [state_h, state_c]

# Decoder
decoder_inputs = RepeatVector(n_timesteps_out)(state_h)
decoder_lstm = LSTM(n_LSTM_units, return_sequences=True, name='decoder_lstm')
decoder_output = decoder_lstm(decoder_inputs, initial_state=states)
decoder_dense1 = TimeDistributed(Dense(dense_units, activation='relu'))
decoder_dense2 = TimeDistributed(Dense(n_features_out, activation='linear'))
output = decoder_dense1(decoder_output)
output = decoder_dense2(output)

# Define and compile model
model2 = Model(encoder_inputs, output, name='model_encoder_decoder')
model2.summary()

path_checkpoint = "model_checkpoint1.weights.h5"
cp1 = ModelCheckpoint(path_checkpoint, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)
earlystop = EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

print("CHECK POINT : ** NN model created **")

model2.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[MeanAbsoluteError()])
history = model2.fit(x_train_s, y_train_s, batch_size=batch_size, validation_data=(x_val_s, y_val_s), epochs=500, callbacks=[cp1, earlystop])
model2.load_weights(path_checkpoint)

model2.save("model_Enco_Deco_full1.keras")

print("CHECK POINT : ** NN model trained **")
tac = time.perf_counter()
print(f"Neural Network trained in {tac - tic:0.4f} seconds")

# ------------------------------------------- Model Evaluation --------------------------------------------------

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(f"Loss evolution during training phase")
plt.ylabel('Loss [bar]')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

MAE_M = []
MAE_list = []
std_list = []

def plot_predictions1(model, x, y, training_mean_y, training_std_y, node_number, start=216, end=288):
    predictions = model.predict(x)
    predictions = predictions[:, :, node_number]
    predictions = predictions.flatten()
    predictions_rescaled = (predictions * training_std_y) + training_mean_y
    y = y.flatten()
    y_rescaled = (y * training_std_y) + training_mean_y
    y_mean = np.mean(y_rescaled)

    df = pd.DataFrame(data={'Predictions': predictions_rescaled, 'Actuals': y_rescaled})
    x_values = range(0, end - start)
    plt.plot(x_values, df['Predictions'][start:end], label="Predicted pressure", color='#8B0000')
    plt.plot(x_values, df['Actuals'][start:end], label="Actual pressure", color='orange')
    plt.title(f"Gas pressure evolution for a single node ({node_number+1})", fontsize=16)
    plt.xlabel("Time [hours]", fontsize=14)
    plt.ylabel("Gas pressure [bar]", fontsize=14)
    plt.legend(fontsize=14)
    plt.show()

    MSE = mse(y_rescaled, predictions_rescaled)
    MAE = mae(y_rescaled, predictions_rescaled)
    error = y_rescaled - predictions_rescaled
    STD  = np.std(error)
    MAE_MEAN = (MAE / y_mean)*100
    MAE_M.append(MAE_MEAN)
    MAE_list.append(MAE)
    std_list.append(STD)

    print(f"\n=== Test Metrics of node {node_number + 1} ===")
    print(f"Mean Squared Error (MSE): {MSE:.4f}")
    print(f"Mean Absolute Error (MAE): {MAE:.4f}")
    print(f"Mean Absolute Error / Mean (MAE/MEAN): {MAE_MEAN:.4f}")

    return df, MSE, MAE, MAE_MEAN

day = 1

for pipe_number in range(12):
    plot_predictions1(model2, x_test_s, y_test_s[:, :, pipe_number], training_mean_y, training_std_y, pipe_number,
                      start=day*24, end=(day+3)*24)

print(f"\n=== Global Test Metrics ===")
print("MAE :", np.round(MAE_list, 4))
print("Global MAE :", np.round(np.mean(MAE_list), 4))
print("STD :", np.round(std_list, 4))
print("Global STD :", np.round(np.mean(std_list), 4))
print("MAE_MEAN :", np.round(MAE_M, 4))
print("Global MAE_MEAN :", np.round(np.mean(MAE_M), 4))
print(f"Neural Network trained in {tac - tic:0.4f} seconds")


print("CHECK POINT : ** end of process **")

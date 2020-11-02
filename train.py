import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout



#function to normalize data from csv
def process_data():
    data = pd.read_csv('NVDA.csv', date_parser = True)
    data_train = data[data['Date']<='2019-01-01'].copy()
    print("train data")
    print(data_train.head())
    data_test = data[data['Date']>'2019-01-01'].copy()
    print("test data")
    print(data_test.tail())

    training_data = data_train.drop(['Date','Adj Close'], axis = 1)
    print("train data without column Date & Adj Close")
    print(training_data.head())

    test_data = data_test.drop(['Date','Adj Close'], axis = 1)
    print("test data without column Date & Adj Close")
    print(test_data.tail())

    #Scaling data
    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data)
    print("Scaled training data")
    print(training_data)

    test_data1 = scaler.fit_transform(test_data)
    print("Scaled test data")
    print(test_data1)

    build_NN(training_data,test_data1,scaler)

#building neural net
def build_NN(training_data, test_data1,scaler):
    #creating 3D input data for validation
    X_test = []
    Y_test = []


    for i in range(30, len(test_data1)):
        X_test.append(test_data1[i-30:i])
        Y_test.append(test_data1[i, 0])

    X_test, Y_test = np.array(X_test), np.array(Y_test)

    #creating 3D input data for training
    X_train = []
    Y_train = []
    training_data.shape[0]#number of rows in the dataset

    #creating a 3D array for input to the model; in the form batch size,features,timesteps
    for i in range(30, len(training_data)):#we use 1 month as a time step that is 30 days to predict the day ahead.
        X_train.append(training_data[i-30:i])
        Y_train.append(training_data[i, 0])

    #converting the 3D input into numpy array
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    print(X_train.shape)
    print(Y_train.shape)

   #Building LSTM net

   #initializing the neural net with object Sequential()
    regressor = Sequential()

    regressor.add(LSTM(units = 60, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 5)))
    regressor.add(Dropout(0.2))#dropout layers responsible to prevent overfitting; 20% of the layers will be dropped
    regressor.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
    regressor.add(Dropout(0.3))
    regressor.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
    regressor.add(Dropout(0.4))
    regressor.add(LSTM(units = 120, activation = 'relu'))
    regressor.add(Dropout(0.5))
    regressor.add(Dense(units = 1))#specifies output as 1

    regressor.summary()# summary of the built layers
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    history = regressor.fit(X_train, Y_train, epochs = 200, batch_size = 32,validation_data=(X_test, Y_test))#batch size specifies how many samples the neural net should see before updating the weights when looking for the minima

    regressor.save('athman22.model')

    #plotting training loss and validation loss against number of epochs
    loss_train = np.array(history.history['loss'])
    loss_val = np.array(history.history['val_loss'])
    epochs = range(0,200)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()




process_data()

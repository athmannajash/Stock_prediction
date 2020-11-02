import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def predict():
    data = pd.read_csv('NVDA_Test.csv', date_parser = True)

    #predict_data = data[data['Date']>'2020-07-07'].copy()
    predict_data = data.drop(['Date','Adj Close'], axis = 1)
    print("predict data without column Date & Adj Close")
    print(predict_data)


    #Scaling data
    scaler = MinMaxScaler()
    predict_data = scaler.fit_transform(predict_data)
    print("Scaled predict data")
    print(predict_data)


    X_predict = []
    Y_predict = []


    for i in range(30, predict_data.shape[0]):
        X_predict.append(predict_data[i-30:i])
        Y_predict.append(predict_data[i, 0])

    X_predict, Y_predict = np.array(X_predict), np.array(Y_predict)


    model = tf.keras.models.load_model("athman.model")
    prediction = model.predict(X_predict)
    print(prediction)

    ##INVERSE SCALING PREDICTED DATA
    # create empty table with 5 fields as the input data has 5 features
    trainPredict_dataset_like = np.zeros(shape=(len(prediction), 5) )
    # putting predicted values in respective fields
    trainPredict_dataset_like[:,0] = prediction[:,0]
    # inverse transform and then select the right field
    prediction = scaler.inverse_transform(trainPredict_dataset_like)[:,0]

    ##INVERSE SCALING ORIGINAL DATA
    #create empty table with 5 fields as the input data has 5 features
    Y_predict = Y_predict.reshape(-1, 1)
    predict_dataset = np.zeros(shape=(len(Y_predict), 5) )
    # put the test values in respective fields
    predict_dataset[:,0] = Y_predict[:,0]
    # inverse transform and then select the right field
    Y_predict = scaler.inverse_transform(predict_dataset)[:,0]

    print("these are predicted values")
    print(prediction)
    print("these are the real values")
    print(Y_predict)

    ### VISUALIZING THE PREDICTED AND TEST DATA
    plt.figure(figsize = (14,5))
    plt.plot(Y_predict, color = 'red', label = "Test price")
    plt.plot(prediction, color = 'green', label = "Predicted price")
    plt.title("NVIDIA stock prediction")
    plt.xlabel("Date")
    plt.ylabel("NVIDIA STOCK PRICE")
    plt.legend()
    plt.show()

predict()

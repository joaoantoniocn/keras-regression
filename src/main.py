import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.decomposition import PCA
from parser import Parser
import sys


def main(data_traning, labels_training, data_testing, submission_path):

    parser = Parser()

    # loading data
    data_data = pd.read_csv(data_traning) #'../database/train/train_100k.csv'
    data_test = pd.read_csv(data_testing) # '../database/test/test_100k.csv'

    data_labels = pd.read_csv(labels_training) # '../database/train/train_100k.truth.csv'

    train_data = data_data.values
    train_labels = data_labels.values

    test_data = data_test.values[:, 1:]

    td = train_data[:, 1:]

    tl = train_labels[:, 1:]

    #td_new = parser.agrupamento_all(td, 2)


    # building model
    model = Sequential()
    model.add(Dense(50, input_dim=td.shape[1], activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation='linear'))


    model.compile(optimizer='rmsprop', loss='mse', metrics=['acc', 'mae'])

    # training model
    model.fit(td, tl, epochs=300, verbose=2)


    # --------------- result on training data
    #result = model.predict(td)

    #result_dataframe = pd.DataFrame(result, columns=['slope', 'intercept'])
    #result_dataframe.index.name = 'id'
    #result_dataframe.to_csv('../database/submission-training.csv')

    # --------------- result on testing data
    result_test = model.predict(test_data)

    result_test_dataframe = pd.DataFrame(result_test, columns=['slope', 'intercept'])
    result_test_dataframe.index.name = 'id'
    result_test_dataframe.to_csv(submission_path) # '../database/submission_test.csv'



if __name__ == "__main__":
    data_training = str(sys.argv[1])
    labels_traning = str(sys.argv[2])
    data_testing = str(sys.argv[3])
    submission_path = str(sys.argv[4])

    main(data_training, labels_traning, data_testing, submission_path)


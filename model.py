# -*- coding: utf-8 -*-
########################################## DATA PREPROCESSING ################################################
import pandas as pd
# Load dataset
dataset_train = pd.read_csv('dataset_sample/train.csv', nrows=10000)
# Remove useless columns
dataset_train = dataset_train.drop('key', 1)
# Remove 0 coords
dataset_train = dataset_train.dropna(how = 'any', axis = 'rows')
dataset_train = dataset_train[dataset_train.dropoff_longitude !=0]
dataset_train = dataset_train[dataset_train.dropoff_latitude  !=0]
dataset_train = dataset_train[dataset_train.pickup_longitude  !=0]
dataset_train = dataset_train[dataset_train.pickup_latitude   !=0]
dataset_train = dataset_train[dataset_train.dropoff_longitude<-60]
dataset_train = dataset_train[dataset_train.dropoff_longitude>-80]
dataset_train = dataset_train[dataset_train.dropoff_latitude  <50]
dataset_train = dataset_train[dataset_train.dropoff_latitude  >20]
dataset_train = dataset_train[dataset_train.pickup_longitude <-60]
dataset_train = dataset_train[dataset_train.pickup_longitude >-80]
dataset_train = dataset_train[dataset_train.pickup_latitude   <50]
dataset_train = dataset_train[dataset_train.pickup_latitude   >20]
# Convert coords in to points
dataset_train['pickup_point']  = dataset_train[['pickup_latitude', 'pickup_longitude']].apply(
                                                lambda coords: (coords[0],coords[1]), axis=1)
dataset_train['dropoff_point'] = dataset_train[['dropoff_latitude', 'dropoff_longitude']].apply(
                                                lambda coords: (coords[0],coords[1]), axis=1)
# Remove coords
dataset_train = dataset_train.drop(['dropoff_longitude',
                                    'dropoff_latitude' ,
                                    'pickup_longitude' ,
                                    'pickup_latitude'] , axis=1)
# Calculate Taxicab geometry (Manhattan distance)
import math as m
def manhattan_distance(pickup_point, dropoff_point):
    Δ1, λ1 = pickup_point
    Δ2, λ2 = dropoff_point
    Δ1, λ1 = m.radians(Δ1), m.radians(λ1)
    Δ2, λ2 = m.radians(Δ2), m.radians(λ2)
    
    Δφ = abs(Δ2 - Δ1)
    Δλ = abs(λ2 - λ1)
    
    a = m.sqrt(m.sin(Δφ/2))
    c = 2 * m.atan2(m.sqrt(a), m.sqrt(1-a))
    latitudeDistance = 6371 * c
    
    a = m.sqrt(m.sin(Δλ/2))
    c = 2 * m.atan2(m.sqrt(a), m.sqrt(1-a))
    longitudeDistance = 6371 * c
    
    return abs(latitudeDistance) + abs(longitudeDistance)
dataset_train['manhattan_distance'] = dataset_train[['pickup_point', 'dropoff_point']].apply(
                                                lambda coords: manhattan_distance(coords[0],coords[1]), axis=1)
# Remove points
dataset_train = dataset_train.drop(['pickup_point',
                                    'dropoff_point'] , axis=1)
# Remove 0 distances ???
    
# Convert "dates" in to dates
from datetime import datetime as dt
dataset_train['pickup_datetime'] = dataset_train['pickup_datetime'].apply(
                                                lambda date: dt.strptime(date[:19],'%Y-%m-%d %H:%M:%S'))
# Split dates in columns
dataset_train['pickup_year']  = dataset_train['pickup_datetime'].apply(
                                                lambda date: date.year)
dataset_train['pickup_month'] = dataset_train['pickup_datetime'].apply(
                                                lambda date: date.month)
dataset_train['pickup_hour']  = dataset_train['pickup_datetime'].apply(
                                                lambda date: date.hour)
# Remove dates
dataset_train = dataset_train.drop(['pickup_datetime'] , axis=1)
# Get data
X = dataset_train.iloc[:,1:6].values
y = dataset_train.iloc[:,:1].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)
############################################### CREATING THE MODEL ###########################################
# Importing the Keras libraries and packages
from   keras.models import Sequential
from   keras.layers import Dense
from   keras.layers import Dropout
# Initialising the ANN
def model(x_size, y_size):
    model = Sequential()
    model.add(Dense(100, activation="tanh", input_shape=(x_size,)))
    model.add(Dropout(0.1))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(y_size))
    model.compile(loss='mean_squared_error',
        optimizer='nadam',
        metrics=['accuracy'])
    return model
predictor  = model(5,1)
history     = predictor.fit(X_train, y_train, batch_size = 10, epochs = 100)
predictions = predictor.predict(X_test)

predictor.save('predictor.h5')
############################################ DISPLAYING THE RESULTS ##########################################
import matplotlib.pyplot as plt
def plot_hist(h):
    plt.plot(h['loss'])
    plt.plot(h['acc'])
    plt.draw()
    plt.show()
    return
plot_hist(history.history)

results = pd.DataFrame(data=predictions, columns=['predicted_fare'])
results['actual_fare'] = y_test
results['difference']  = results[['predicted_fare','actual_fare']].apply(
                                            lambda values: abs(values[0]-values[1]), axis=1)
def plot_results(res):
    plt.scatter(res['actual_fare'],   res.index.values, marker = '1', color = '#1FD91F')
    plt.scatter(res['predicted_fare'],res.index.values, marker = '1', color = '#D91F1F')
    plt.draw()
    plt.show()
    return
plot_results(results)
def plot_difference(res):
    plt.scatter(res['difference'],    res.index.values, marker = '1', color = '#CB1FD9')
    plt.draw()
    plt.show()
    return
plot_difference(results)


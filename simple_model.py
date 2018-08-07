# -*- coding: utf-8 -*-
########################################## DATA PREPROCESSING ################################################
import pandas as pd
# Load dataset
dataset_train = pd.read_csv('dataset_sample/train.csv', nrows=100000)
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
                                    'dropoff_point',
                                    'passenger_count'] , axis=1)
dataset_train['manhattan_distance']  = dataset_train['manhattan_distance'].apply(
                                                lambda distance: round(distance))
# Convert "dates" in to dates
from datetime import datetime as dt
dataset_train['pickup_datetime'] = dataset_train['pickup_datetime'].apply(
                                                lambda date: dt.strptime(date[:19],'%Y-%m-%d %H:%M:%S'))
# Split dates in columns
dataset_train['pickup_hour']  = dataset_train['pickup_datetime'].apply(
                                                lambda date: date.hour)
# Remove dates
dataset_train = dataset_train.drop(['pickup_datetime'] , axis=1)
# Get data
# Splitting the dataset into the Training set and Test set
X = dataset_train.iloc[:,1:].values
y = dataset_train.iloc[:,:1].values

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
    model.add(Dense(20, activation="tanh", input_shape=(x_size,)))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(40, activation="sigmoid"))
    model.add(Dropout(0.2))
    model.add(Dense(y_size))
    model.compile(loss='mean_squared_error',
        optimizer='nadam',
        metrics=['accuracy'])
    return model
predictor   = model(2,1)
history     = predictor.fit(X_train, y_train, batch_size = 20, epochs = 100)
predictions = predictor.predict(X_test)

#predictor.save('not_so_simple_predictor.h5')
############################################ DISPLAYING THE RESULTS ##########################################
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
def plot_hist(h):
    plt.subplot(GridSpec(2, 2)[0, 0])
    plt.ylabel('Accuracy & Loss')
    plt.plot(h['loss'])
    plt.plot(h['acc'])
    plt.draw()
    return
#plot_hist(history.history)
#plt.show()
results = pd.DataFrame(data=predictions, columns=['predicted_fare'])
results['actual_fare'] = y_test
results['difference']  = results[['predicted_fare','actual_fare']].apply(
                                            lambda values: (values[0]-values[1]), axis=1)
### Score
from sklearn.metrics import mean_squared_error
from math import sqrt

predictions = predictor.predict(X_test)
rms = sqrt(mean_squared_error(y_test, predictions))
# Plots
def plot_results(res):
    plt.scatter(res['actual_fare'],   res.index.values, marker = '1', color = '#1FD91F')
    plt.scatter(res['predicted_fare'],res.index.values, marker = '1', color = '#D91F1F')
    plt.draw()
    return
#plot_results(results)
#plt.show()
def plot_difference(res):
    plt.subplot(GridSpec(2, 2)[0, 1])
    plt.ylabel('Difference')
    plt.xlabel('$')
    plt.scatter(res['difference'],    res.index.values, marker = '1', color = '#CB1FD9')
    plt.draw()
    return
#plot_difference(results)
#plt.show()
def plot_accuracy(accuracy):
    plt.subplot(GridSpec(2, 2)[1, 0],aspect=1)
    plt.pie(accuracy,labels=['Correct','Wrong'],autopct='%1.1f%%',colors=['g','r'])
    plt.draw()
    return
def plot_error(difference):
    plt.subplot(GridSpec(2, 2)[1, 1])
    difference  = difference.apply(lambda values: abs(values))
    difference.sort_values()
    plt.plot(difference)
def plot_rmse(rms):
    plt.subplot(GridSpec(2, 2)[1, 1])
    plt.grid()
    plt.scatter(rms,0)
def display_results(h, res,rmse):
    plot_hist(h)
    plot_difference(res)
    r1 = len(results[results['difference']<= 0.01])
    r2 = len(results[results['difference']>=-0.01])
    r  = abs(r1-r2)
    #d1 = len(results[results['difference']<=  10])
    #d2 = len(results[results['difference']>=- 10])
    #d  = abs(d1-d2) 
    #d  = abs(d-r)
    w = len(results)-r
    plot_accuracy([r,w])
    #plot_error(results['difference'])
    plot_rmse(rms)
    plt.show()
    return
display_results(history.history, results, rms)

### Load old model
#from keras.models import load_model
#predictor = load_model('not_so_simple_predictor.h5')

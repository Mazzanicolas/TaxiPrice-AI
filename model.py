# -*- coding: utf-8 -*-
import pandas as pd
# Load dataset
dataset_train = pd.read_csv('train.csv', nrows=1000)
# Remove useless columns
dataset_train = dataset_train.drop('key', 1)
# Remove 0 coords
dataset_train = dataset_train.dropna(how = 'any', axis = 'rows')
dataset_train = dataset_train[dataset_train.dropoff_longitude!=0]
dataset_train = dataset_train[dataset_train.dropoff_latitude !=0]
dataset_train = dataset_train[dataset_train.pickup_longitude !=0]
dataset_train = dataset_train[dataset_train.pickup_latitude  !=0]
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

# Discretize time
from datetime import datetime as dt
dataset_train['pickup_datetime'] = dataset_train['pickup_datetime'].apply(
                                                lambda date: dt.strptime(date[:19],'%Y-%m-%d %H:%M:%S'))

dataset_train['pickup_datetime'] = dataset_train[dataset_train['pickup_datetime']]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
from geopy.distance import vincenty
def get_distance(row): 
    p1 = (row['pickup_longitude'])

# Display data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plot = dataset_test.iloc[:].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')

dataset_test = dataset_test[dataset_test['abs_diff_longitude']<0.4]
dataset_test = dataset_test[dataset_test['abs_diff_latitude']<0.4]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dataset_test['abs_diff_longitude'], dataset_test['abs_diff_latitude'], dataset_test['fare_amount'])
plt.show()

mean = dataset_test['fare_amount'].mean()








# Utils
def remove_noise(dataset, colum_name, noise_value):
    return dataset_test[dataset_test[colum_name]!=noise_value]
def add_distance_vector_features(dataFrame):
    dataFrame['abs_diff_longitude'] = (dataFrame.dropoff_longitude - dataFrame.pickup_longitude).abs()
    dataFrame['abs_diff_latitude']  = (dataFrame.dropoff_latitude  - dataFrame.pickup_latitude ).abs()
add_distance_vector_features(dataset_test)

#remove abs sum 0
dataset_train = dataset_train[dataset_train.dropoff_longitude.abs()+
                             dataset_train.dropoff_latitude.abs() +
                             dataset_train.pickup_longitude.abs() +
                             dataset_train.pickup_latitude.abs()  !=0]
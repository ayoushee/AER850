import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from learn.model  import train_test_split, StratifiedShuffleSplit

df = pd.read_csv("data/housing.csv")


"""OneHotEncoding"""
my_encoder = OneHotEncoder(sparse_output=False)
my_encoder.fit(df[['ocean_proximity']])
encoded_data = my_encoder.transform(df[['ocean_proximity']])

category_names = my_encoder.get_feature_names_out()
encoded_data_df= pd.DataFrame(encoded_data, columns = category_names)

df = pd.concat([df,encoded_df],axis=1)
df = df.drop(columns = 'ocean proximity')

"""Define X and y"""
x_columns = ['longitude',
             'latitude',
             ' housing_median_age'
             'total_rooms'
             'total_bedrooms'
             'population'
             'households',
             'median_income'
             'ocean_proximity_<1H OCEAN',
             'ocean_proximity_INLAND',
             'ocean_proximity_ISLAND'
             'ocean_proximity_NEAR BAY'
             'ocean_proximity_NEAR OCEAN'
             ]
y_column = ['median_house_value']
x = df[x_columns]  ## independant variables, predictor
y = df[y_columns]  ##outcome measure, dependent variable
 
"""Test Split"""
(X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                     test_size=0.2,
                                                     random_state = 134535)

print(df.columns)

x_columns = ['longitude',
             ]

import json
import pickle
import numpy as np

__locations = None
__model = None
__data_columns= None

def get_location_names():
    return __locations

def get_estimate_price(location,bhk,bath,sqft):
    try:
        location_index = __data_columns.index(location.lower())
    except:
        location_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1]= bath
    x[2] = bhk

    if location_index >=0:
        x[location_index] = 1

    return round(__model.predict([x])[0], 2)


def load_saved_artifacts():
    print("loading saved artifacts.....START")
    global __data_columns
    global __locations

    with open("./server/artifacts/common.json","r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

    global __model
    if __model is None:
        with open('./server/artifacts/banglore_home_price_model.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimate_price('Rajaji Nagar',2,2,1000))
    print(get_estimate_price('Rajaji Nagar', 3, 2,1000))
    print(get_estimate_price('Indira Nagar', 2, 2, 1000))
    print(get_estimate_price('Indira Nagar', 3, 2, 1000))
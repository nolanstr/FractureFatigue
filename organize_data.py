import numpy as np
import time
import copy
import pickle
import glob
import matplotlib.pyplot as plt


def get_E(FILE):
    if "AISI" in FILE:
        return 29700 #ksi
    else:
        return 10200 #ksi

def store_data():
    
    files = glob.glob("../../../../FF_data/*.npy")
    data_sets = [np.load(FILE) for FILE in files]
    E_vals = [get_E(FILE) * np.ones((data_sets[i].shape[0],1)) for i, FILE in \
                            enumerate(files)]
    for i in range(len(files)):
        if np.where(np.isnan(data_sets[i]))[0].shape[0] != 0:
            data_sets[i] = np.hstack((E_vals[i],
                data_sets[i]))[~np.where(np.isnan(data_sets[i]))[0],:]
        else:
            data_sets[i] = np.hstack((E_vals[i],
                data_sets[i]))

    for FILE, data_set in zip(files, data_sets):

        new_name = "cleaned_" + FILE.split('/')[-1].split('.')[0]
        np.save(new_name, data_set)


if __name__ == "__main__":
    store_data()

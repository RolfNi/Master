import numpy as np
import glob
from pandas import DataFrame, read_feather

def file_search(directory, filetype, gamma, lmda, dt, t):
    filename = f'gamma{gamma}_lambda{lmda}_dt{dt}_t{t}'
    paths = glob.glob(f'{directory}/**/*{filename}.{filetype}', recursive=True)
    return paths

def load_series_feather(path):
    return read_feather(path).to_numpy().transpose()

def save_series_feather(times, series, directory, gamma, lmda, dt, t, n):
    filename = f'gamma{gamma}_lambda{lmda}_dt{dt}_t{t}_n{n}'
    DataFrame(np.array((times, series)).transpose(), columns=['t', 'x']).to_feather(filename)
    
def save_series_npy(times, series, directory, filename):
    storage_array = np.stack((times, series))
    np.save(directory+filename, storage_array)
    
def file_searchexp(directory, filetype, gamma, lmda, dur, ntype, ntsratio, dt, t):
    filename = f'gamma{gamma}_lambda{lmda}_duration{dur}_noisetype{ntype}_ratio{ntsratio}_dt{dt}_t{t}*'
    paths = glob.glob(f'{directory}/**/*{filename}.{filetype}', recursive=True)
    return paths
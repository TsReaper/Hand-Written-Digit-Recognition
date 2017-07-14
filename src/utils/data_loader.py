import pickle
import gzip
import os
import numpy as np

def load_data():
    # Download data if not exist
    if not os.path.exists('mnist.pkl.gz'):
        print('Data not found. Downloading data...')

        try:
            import requests
            url = 'https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/master/data/mnist.pkl.gz'
            response = requests.get(url, stream = True)
            total_length = int(response.headers.get('content-length'))

            f = open('mnist.pkl.gz', 'wb')
            dl = 0
            for data in response.iter_content(chunk_size = 4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                print('\r[%s%s]' % ('=' * done, ' ' * (50 - done)), end = '', flush = True)
            f.close()

            print('\nData successfully downloaded')
        except:
            print('\nFailed to download data. Please download manually from ' + url)
            exit()
    
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding = 'bytes')
    f.close()
    return change_data_type(training_data), change_data_type(validation_data), change_data_type(test_data)

def change_data_type(data):
    inputs = [np.reshape(x, (784)) for x in data[0]]
    labels = []
    for x in data[1]:
        labels.append(np.zeros((10)))
        labels[-1][x] = 1
    return inputs, labels

import numpy as np

import utils.data_loader as data_loader
from network.network import Network
from network.network_improved import NetworkImproved

training_data, validation_data, test_data = data_loader.load_data()

# Train network
print('Which network do you want to train?')
print('1. Basic network')
print('2. Network improved')
n = int(input())

if n == 1:
    my_network = Network((784, 50, 10))
    my_network.train(*training_data, 3.0, 30, 10)
elif n == 2:
    my_network = NetworkImproved((784, 50, 10))
    my_network.setAttribs(l2_norm = True)
    my_network.train(*training_data, 0.5, 30, 10)

# Test network
correct, tot = 0, len(test_data[0])
for data, label in zip(*test_data):
    h, ans = np.argmax(my_network.predict(data)), np.argmax(label)
    if h == ans:
        correct += 1
print('Correctness: %d/%d = %.2f%%' % (correct, tot, correct/tot*100))

# Store network in json file
f = open('json.js', 'w')
f.write('let json = ' + my_network.get_json() + ';')
f.close()

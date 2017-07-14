import utils.data_loader as data_loader
from network.network import Network
import numpy as np

training_data, validation_data, test_data = data_loader.load_data()

# Train network
my_network = Network((784, 50, 10))
my_network.train(*training_data, 3.0, 30, 10)

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

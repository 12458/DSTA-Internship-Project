import pickle
import numpy as np


# Load RadioML dataset
with open('RML2016.10a_dict.pkl', 'rb') as f:
    data = pickle.load(f, encoding='bytes')


print(sorted(data.keys()))


#Keys to extract
keys = [(b'QAM16', 18), (b'QAM64', 18), (b'QPSK', 18), (b'8PSK', 18), (b'CPFSK', 18), (b'GFSK', 18), (b'BPSK', 18), (b'PAM4', 18)]

for key in keys:
    print(key)
    data[key][0].tofile(f'{str(key[0])}.iq')
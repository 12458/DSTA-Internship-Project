import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
import adi
import numpy as np

np.seterr(divide='ignore', invalid='ignore')

# Create radio
sdr = adi.Pluto()

# Configure properties
sdr.rx_rf_bandwidth = int(600e3)
sdr.rx_lo = int(2.437e9)
sdr.sample_rate = sdr.rx_rf_bandwidth
sdr.rx_buffer_size = 128
sdr.gain_control_mode_chan0 = "hybrid"

# Read properties
print("RX LO %s" % (sdr.rx_lo))

mod_types = ['a16QAM', 'a64QAM', 'b8PSK', 'bQPSK', 'cCPFSK', 'cGFSK', 'd4PAM', 'dBPSK']

# fit a label binarizer
mod_to_onehot = LabelBinarizer()
mod_to_onehot.fit(mod_types)

# transform the y values to one-hot encoding
# y_train = mod_to_onehot.transform(y_train)

# Normalisation is very important


def iq2ampphase(inphase, quad):
    amplitude = np.sqrt(np.square(inphase) + np.square(quad))
    amp_norm = np.linalg.norm(amplitude)  # L2 norm
    amplitude = amplitude/amp_norm  # normalise
    phase = np.arctan(np.divide(quad, inphase))
    phase = 2.*(phase - np.min(phase))/np.ptp(phase)-1  # rescale phase to range [-1, 1]
    return amplitude, phase

# convert array of multiple iq samples into array of multiple ampphase samples


def arr_iq2ap(X):
    X_ap = []
    I = X[0, :]
    Q = X[1, :]
    amp, phase = iq2ampphase(I, Q)
    ap = np.array([amp, phase])
    return ap


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model/cldnn_ap_model.tflite")
interpreter.allocate_tensors()

count = dict.fromkeys(mod_types, 0)
for i in range(1000):
    iq = np.array(sdr.rx())
    iq_np = np.array([iq.real, iq.imag])
    qpsk = arr_iq2ap(iq_np)
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(input_details, '\n', output_details)

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    input_data = qpsk.reshape(input_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.

    output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)
    modulation_guess = mod_to_onehot.inverse_transform(output_data)[0]
    # time.sleep(0.1)
    count[modulation_guess] += 1

print(count)

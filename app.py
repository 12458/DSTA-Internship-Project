import numpy as np
import adi
import iio
import tflite_runtime.interpreter as tflite
import subprocess
import shlex

# import sys

#from sklearn.preprocessing import LabelBinarizer

np.seterr(divide='ignore', invalid='ignore')


# iio_attr -u ip:192.168.2.1 -d
# iio_attr -u ip:192.168.2.1 -c ad9361-phy RX_LO frequency 1410000000
# iio_attr -u ip:192.168.2.1 -c ad9361-phy voltage0 rf_bandwidth 600000
# iio_attr -u ip:192.168.2.1 -c ad9361-phy voltage0 sampling_frequency 600000
# iio_attr -u ip:192.168.2.1 -c ad9361-phy voltage0 gain_control_mode hybrid
# iio_readdev -u ip:192.168.2.1 -b 128 -s 1024 cf-ad9361-lpc

# Show deivce attributes
process = subprocess.Popen(['iio_attr', '-u', 'ip:192.168.2.1', '-d'], stdout=subprocess.PIPE)
stdout = process.communicate()[0]
print('DEVICE:')
print(stdout.decode('utf-8'))

# SETUP
subprocess.Popen(['iio_attr', '-u', 'ip:192.168.2.1', '-c', 'ad9361-phy', 'RX_LO', 'frequency', '1410000000'], stdout=subprocess.PIPE)
subprocess.Popen(['iio_attr', '-u', 'ip:192.168.2.1', '-c', 'ad9361-phy', 'voltage0', 'rf_bandwidth', '600000'], stdout=subprocess.PIPE)
subprocess.Popen(['iio_attr', '-u', 'ip:192.168.2.1', '-c', 'ad9361-phy', 'voltage0', 'sampling_frequency', '600000'], stdout=subprocess.PIPE)
subprocess.Popen(['iio_attr', '-u', 'ip:192.168.2.1', '-c', 'ad9361-phy', 'voltage0', 'gain_control_mode', 'hybrid'], stdout=subprocess.PIPE)

mod_types = ['a16QAM', 'a64QAM', 'b8PSK', 'bQPSK', 'cCPFSK', 'cGFSK', 'd4PAM', 'dBPSK']


def iq2ampphase(inphase, quad):
    amplitude = np.sqrt(np.square(inphase) + np.square(quad))
    amp_norm = np.linalg.norm(amplitude)  # L2 norm
    amplitude = amplitude/amp_norm  # normalise
    phase = np.arctan(np.divide(quad, inphase))
    phase = 2.*(phase - np.min(phase))/np.ptp(phase)-1  # rescale phase to range [-1, 1]
    return amplitude, phase


def arr_iq2ap(X):
    X_ap = []
    I = X[:, 0]
    Q = X[:, 1]
    amp, phase = iq2ampphase(I, Q)
    ap = np.array([amp, phase])
    return ap


# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

count = dict.fromkeys(mod_types, 0)
for i in range(10):
    process = subprocess.Popen(['iio_readdev', '-u', 'ip:192.168.2.1', '-b', '128', '-s', '128', 'cf-ad9361-lpc'], stdout=subprocess.PIPE, bufsize=0)
    stdout = process.communicate()[0]
    iq = np.array(np.frombuffer(stdout, dtype=np.int16))
    iq = np.reshape(iq, (-1, 2))
    
    iq_np = iq
    ap = arr_iq2ap(iq_np)

    print(ap)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(input_details, '\n', output_details)

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    input_data = ap.reshape(input_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.

    output_data = interpreter.get_tensor(output_details[0]['index'])
    index = np.argmax(output_data)
    # print(output_data)
    # modulation_guess = mod_to_onehot.inverse_transform(output_data)[0]
    count[mod_types[index]] += 1

print(count)

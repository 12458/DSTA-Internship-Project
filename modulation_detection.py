import numpy as np
import time
import tflite_runtime.interpreter as tflite
import subprocess
import adi


np.seterr(divide='ignore', invalid='ignore')

# Show deivce attributes
process = subprocess.Popen(['iio_attr', '-u', 'ip:192.168.2.1', '-d'], stdout=subprocess.PIPE)
stdout = process.communicate()[0]
print('DEVICE:')
print(stdout.decode('utf-8'))

# SETUP
# Create radio
sdr = adi.Pluto()

# Configure properties
sdr.rx_rf_bandwidth = int(4e6)
sdr.rx_lo = int(1.41e9)
sdr.sample_rate = sdr.rx_rf_bandwidth
sdr.rx_buffer_size = 128
sdr.gain_control_mode_chan0 = "hybrid"

# Read properties
print("RX LO %s" % (sdr.rx_lo))

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
    I = X[0, :]
    Q = X[1, :]
    amp, phase = iq2ampphase(I, Q)
    ap = np.array([amp, phase])
    return ap


# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

count = dict.fromkeys(mod_types, 0)

time.sleep(1)

while True:
    iq = np.array(sdr.rx())

    iq_np = np.array([iq.real, iq.imag])
    ap = arr_iq2ap(iq_np)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_data = ap.reshape(input_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    index = np.argmax(output_data)

    print(f"Current modulation type detected: {mod_types[index]}")

print()
print(count)

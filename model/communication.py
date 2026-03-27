import serial
import numpy as np

PORT = "COM20"  # change this to your port

def synchronise_UART(serial_port):
    while (1):
        serial_port.write(b"\xAB")
        ret = serial_port.read(1)
        if (ret == b"\xCD"):
            serial_port.read(1)
            break

def send_inputs_to_STM32(inputs, serial_port):
    inputs = inputs.astype(np.float32)
    buffer = b""
    for x in inputs:
        buffer += x.tobytes()
    serial_port.write(buffer)

def read_output_from_STM32(serial_port):
    output = serial_port.read(5)
    float_values = [int(out)/255 for out in output]
    return float_values

def evaluate_model_on_STM32(iterations, serial_port):
    accuracy = 0
    for i in range(iterations):
        print(f"----- Iteration {i+1} -----")
        send_inputs_to_STM32(X_test[i], serial_port)
        output = read_output_from_STM32(serial_port)
        if (np.argmax(output) == np.argmax(Y_test[i])):
            accuracy += 1 / iterations
        print(f"   Expected output: {Y_test[i]}")
        print(f"   Received output: {output}")
        print(f"   Accuracy so far: {accuracy:.2f}\n")
    return accuracy

if __name__ == '__main__':
    X_test = np.load("X_test_pred.npy")
    Y_test = np.load("Y_test_pred.npy")

    with serial.Serial(PORT, 115200, timeout=1) as ser:
        print("Synchronising...")
        synchronise_UART(ser)
        print("Synchronised!")
        print("Evaluating model on STM32...")
        accuracy = evaluate_model_on_STM32(100, ser)
        print(f"Final accuracy: {accuracy:.2f}")

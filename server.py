import socket
from threading import Thread
import numpy as np
import os
import argparse
import config
import util

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train_dir', help='Path to data train directory')
args = parser.parse_args()
train_dir = args.train_dir

if train_dir:
    config.train_dir = train_dir

config.classes = util.get_classes_from_train_dir()
classes_in_keras_format = util.get_classes_in_keras_format()

model = util.load_model(nb_class=len(config.classes))
model.load_weights(config.fine_tuned_weights_path)

print 'Model loaded'

FILE_DOES_NOT_EXIST = '-1'
UNKNOWN_ERROR = '-2'


def handle(clientsocket):
    while 1:
        buf = clientsocket.recv(config.buffer_size)
        if buf == 'exit':
            return  # client terminated connection

        if os.path.isfile(buf):
            try:
                out = model.predict(np.array([util.load_img(buf)]))
                prediction = np.argmax(out, axis=1)

                class_indices = dict(zip(config.classes, range(len(config.classes))))
                keys = class_indices.keys()
                values = class_indices.values()

                answer = keys[values.index(prediction[0])]

                response = '{"probability":"%s","class":"%s"}' % (out[0][prediction[0]], answer)
                print response
                clientsocket.sendall(response)
            except Exception as e:
                print e.message
                clientsocket.sendall(UNKNOWN_ERROR)
        else:
            clientsocket.sendall(str(FILE_DOES_NOT_EXIST))


serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
serversocket.bind(config.server_address)
serversocket.listen(10)

while 1:
    # accept connections from outside
    (clientsocket, address) = serversocket.accept()

    ct = Thread(target=handle, args=(clientsocket,))
    ct.run()

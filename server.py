import socket
from threading import Thread
import numpy as np
import os
import argparse
import config
import util
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', help='Path to data train directory')
parser.add_argument('--model', type=str, default=config.MODEL_VGG16, help='Base model architecture')

args = parser.parse_args()
train_dir = args.train_dir
if train_dir:
    config.train_dir = train_dir
if args.model:
    config.model = args.model

config.classes = util.get_classes_from_train_dir()
classes_in_keras_format = util.get_classes_in_keras_format()

model_module = util.get_model_module()
model = model_module.load_trained()
print 'Model loaded'

try:
    print 'Loading activation function...'
    af = util.get_activation_function(model, model_module.RELATIVITY_LAYER)
    relativity_clf = joblib.load(config.get_relativity_model_path())
except Exception as e:
    print e

FILE_DOES_NOT_EXIST = '-1'
UNKNOWN_ERROR = '-2'


def handle(clientsocket):
    while 1:
        buf = clientsocket.recv(config.buffer_size)
        if buf == 'exit':
            return  # client terminated connection

        if os.path.isfile(buf):
            try:
                img = [util.load_img(buf)]

                out = model.predict(np.array(img))
                prediction = np.argmax(out, axis=1)

                class_indices = dict(zip(config.classes, range(len(config.classes))))
                keys = class_indices.keys()
                values = class_indices.values()

                answer = keys[values.index(prediction[0])]

                if 'relativity_clf' in locals():
                    acts = util.get_activations(af, img)
                    predicted_relativity = relativity_clf.predict(acts)[0]
                    relativity_class = relativity_clf.__classes[predicted_relativity]
                else:
                    relativity_class = 'plant'

                response = '{"probability":"%s","class":"%s","relativity_class":"%s"}' % (out[0][prediction[0]], answer, relativity_class)
                print response
                clientsocket.sendall(response)
            except Exception as e:
                print e.message
                clientsocket.sendall(UNKNOWN_ERROR)
        else:
            clientsocket.sendall(str(FILE_DOES_NOT_EXIST))


serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.bind(config.server_address)
serversocket.listen(10)

while 1:
    # accept connections from outside
    (clientsocket, address) = serversocket.accept()

    ct = Thread(target=handle, args=(clientsocket,))
    ct.run()

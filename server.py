import socket
from threading import Thread
import numpy as np
import os
import argparse
import config
import util
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Base model architecture',
                    choices=[config.MODEL_RESNET50, config.MODEL_RESNET152, config.MODEL_INCEPTION_V3,
                             config.MODEL_VGG16])
args = parser.parse_args()
config.model = args.model

model_module = util.get_model_class_instance()
model = model_module.load()
print 'Model loaded'

try:
    print 'Loading activation function'
    af = util.get_activation_function(model, model_module.noveltyDetectionLayerName)
    print 'Loading relativity classifier'
    novelty_detection_clf = joblib.load(config.get_novelty_detection_model_path())
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
                img = [model_module.load_img(buf)]

                out = model.predict(np.array(img))
                prediction = np.argmax(out)
                top10 = out[0].argsort()[-10:][::-1]

                class_indices = dict(zip(config.classes, range(len(config.classes))))
                keys = class_indices.keys()
                values = class_indices.values()

                answer = keys[values.index(prediction)]

                try:
                    acts = util.get_activations(af, img)
                    predicted_relativity = novelty_detection_clf.predict(acts)[0]
                    nd_class = novelty_detection_clf.__classes[predicted_relativity]
                except Exception as e:
                    print e.message
                    nd_class = 'related'

                top10_json = "["
                for i, t in enumerate(top10):
                    top10_json += '{"probability":"%s", "class":"%s"}%s' % (
                        out[0][t], keys[values.index(t)], '' if i == 9 else ',')
                top10_json += "]"

                response = '{"probability":"%s","class":"%s","relativity":"%s","top10":%s}' % (
                    out[0][prediction], answer, nd_class, top10_json)
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
serversocket.listen(10000)

print('Ready for requests')

while 1:
    # accept connections from outside
    (clientsocket, address) = serversocket.accept()

    ct = Thread(target=handle, args=(clientsocket,))
    ct.run()

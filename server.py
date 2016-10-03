import socket
from threading import Thread
import config
import util
import numpy as np
import os

weight_path = config.top_model_weights_path.format(19, config.output_dim)
model = util.load_model(weight_path)
print 'Model loaded.'

FILE_DOES_NOT_EXIST = -1
UNKNOWN_ERROR = -2


def handle(clientsocket):
    while 1:
        buf = clientsocket.recv(config.buffer_size)
        if buf == 'exit':
            return  # client terminated connection

        if os.path.isfile(buf):
            try:
                out = model.predict(np.array([util.load_img(buf)]))
                prediction = np.argmax(out, axis=1)

                response = '{"probability":"%s","class":"%s"}' % (out[0][prediction[0]], prediction[0])
                print response
                clientsocket.sendall(str(response))
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

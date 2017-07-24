import socket
import argparse
import config

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='Path to image that should be predicted by the model')
args = parser.parse_args()
img_path = args.path

s = socket.socket()
s.connect(config.server_address)

s.send(img_path.encode())
while 1:
    data = s.recv(config.buffer_size)
    if data:
        print(data)
        break

s.send('exit'.encode())
s.close()

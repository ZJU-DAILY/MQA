import numpy as np
import struct


def ivecs_read(filename):
    a = np.fromfile(filename, dtype='int32')
    ret = []
    index = 0
    while index < len(a):
        x = int(a[index])
        index += 1
        current_array = a[index:index + x]
        ret.append(current_array)
        index += x
    return ret.copy()


def to_ivecs(filename, data):
    with open(filename, 'wb') as file:
        for row in data:
            file.write(struct.pack('I', len(row)))
            for item in row:
                file.write(struct.pack('i', item))


def fvecs_read(filename):
    a = np.fromfile(filename, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view('float32')


def to_fvecs(filename, data):
    with open(filename, 'wb') as file:
        for row in data:
            file.write(struct.pack('I', len(row)))
            for item in row:
                file.write(struct.pack('f', item))


def fbin_read(filename):
    a = np.fromfile(filename, dtype='int32')
    ret = []
    index = 0
    while index < len(a):
        x = int(a[index])
        index += 1
        current_array = a[index:index + x]
        ret.append(current_array)
        index += x
    return ret.copy()


def to_fbin(filename, data):
    with open(filename, 'wb') as file:
        for row in data:
            file.write(struct.pack('I', len(row)))
            for item in row:
                file.write(struct.pack('i', item))

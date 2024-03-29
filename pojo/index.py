import os
import subprocess
import sys

from pojo.embedding import Embedding
from pojo.encoder import get_encoder

root = os.getcwd()
if sys.platform.startswith('win'):
    root = '\\\\?\\' + root
base_path = os.path.join(root, 'dataset', 'base')
result_path = os.path.join(root, 'dataset', 'search', 'result.txt')
meta_path = os.path.join(root, 'dataset', 'meta')
search_path = os.path.join(root, 'dataset', 'search')
delete_id_path = os.path.join(root, 'dataset', 'delete.ivecs')


def get_index(algorithm, neighbor, candidate, index_weight):
    if algorithm == 'Flat':
        return IndexFlat(neighbor, candidate, index_weight)


class BaseIndex:
    def __init__(self, neighbor, candidate, index_weight):
        self.neighbor = neighbor
        self.candidate = candidate
        self.index_weight = index_weight

    def index(self):
        raise NotImplementedError


class IndexFlat(BaseIndex):
    def __init__(self, neighbor, candidate, index_weight):
        super().__init__(neighbor, candidate, index_weight)

    def index(self):
        pass

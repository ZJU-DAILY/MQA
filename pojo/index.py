import os
import subprocess
import sys
import json

from pojo.embedding import Embedding

root = os.getcwd()
if sys.platform.startswith('win'):
    root = '\\\\?\\' + root
dataset_path = os.path.join(root, 'dataset')
base_path = os.path.join(root, 'dataset', 'base')
index_config = os.path.join(root, 'dataset', 'index.json')


def get_index():
    with open(index_config, 'r') as file:
        config = json.load(file)
        return config['index_method'], config['index_path']


def set_index(algorithm, neighbor, candidate, index_weight):
    index_path = os.path.join(dataset_path, 'index', f"{algorithm}_{neighbor}_{candidate}")

    if algorithm == 'Flat':
        data = {'index_method': algorithm, 'index_path': index_path}
        print(data)
        with open(index_config, 'w') as file:
            json.dump(data, file, indent=4)
        return

    embedding = Embedding.get()
    args = f'{len(embedding.modalities)}'
    for i, modality in enumerate(embedding.modalities):
        base_name = f'{modality.encoder}'
        for modal in modality.modals:
            base_name += f'_{modal}'
            index_path += f'_{modal}'
        args = args + ' ' + os.path.join(base_path, f"{base_name}.fvecs").replace('\\\\?\\', '')
        args += f' {index_weight[i]}'
        index_path += f'_{index_weight[i]}'
    args += f' {algorithm}'
    args += f' {neighbor}'
    args += f' {candidate}'
    index_path += '.index'
    print(index_path)
    args += ' ' + index_path.replace('\\\\?\\', '')

    print(args)
    if not os.path.exists(index_path):
        if sys.platform.startswith('win'):
            proc = subprocess.run(f'./index_and_search/index.exe {args}')
        else:
            proc = subprocess.run(f'./index_and_search/index {args}')
        if proc.returncode != 0:
            raise Exception(f'Index Error')

    data = {'index_method': algorithm, 'index_path': index_path}
    with open(index_config, 'w') as file:
        json.dump(data, file, indent=4)

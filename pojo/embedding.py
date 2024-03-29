import json
import os
import sys

from pojo.encoder import get_encoder

root = os.getcwd()
if sys.platform.startswith('win'):
    root = '\\\\?\\' + root
dataset_path = os.path.join(root, 'dataset')
query_path = os.path.join(dataset_path, 'query')
meta_path = os.path.join(dataset_path, 'meta')
base_path = os.path.join(dataset_path, 'base')
embedding_config = os.path.join(dataset_path, 'config.json')


class Modality:
    def __init__(self, encoder, modals, weight):
        self.encoder = encoder
        self.modals = modals  # array
        self.weight = 1.0


class Embedding:
    def __init__(self, data):
        self.id = 0
        self.modalities = []
        self.deleted = data['deleted']

        # init class with json file
        for item in data['modalities']:
            modality = Modality(encoder=item['encoder'], modals=item['modals'], weight=item['weight'])
            self.modalities.append(modality)

    def create_embedding(self, data):
        # encode modalities if it hasn't been embedded
        for modality in self.modalities:
            encoder = modality.encoder
            modals = modality.modals

            # the file_path is like f'{encoder}_{modal0}_{modal1}.fvecs'
            file_path = f'{encoder}'
            dataset = []
            for modal in modals:
                file_path += f'_{modal}'
                # create meta file to prevent MLE
                meta_path = os.path.join(dataset_path, 'meta', f'{modal}.txt')
                if not os.path.exists(meta_path):
                    from pojo.dataset import create_meta
                    create_meta(id=modal)
                dataset.append(meta_path)
            file_path = os.path.join(dataset_path, 'base', f'{file_path}.fvecs')

            if not os.path.exists(file_path):
                # retrieve encoder from its id
                encoder = get_encoder(encoder)
                encoder.encode(path=file_path, dataset=dataset, size=32)

        with open(embedding_config, 'w') as file:
            json.dump(data, file, indent=4)
        # config = self.list()
        # data['id'] = len(config)
        # config.append(data)
        # with open(embedding_config, 'w') as file:
        #     json.dump(config, file, indent=4)

    # @classmethod
    # def from_saved(cls):
    #     with open(embedding_config, 'r') as file:
    #         config = json.load(file)
    #     return cls(config)

    # @classmethod
    # def from_id(cls, id):
    #     with open(embedding_config, 'r') as file:
    #         config = json.load(file)
    #     for item in config:
    #         if item['id'] == id:
    #             return cls(item)

    # get a list of existing embeddings
    # @staticmethod
    # def list():
    #     with open(embedding_config, 'r') as file:
    #         config = json.load(file)
    #     return config

    @staticmethod
    def get():
        with open(embedding_config, 'r') as file:
            config = json.load(file)
            return Embedding(data=config)

    # @staticmethod
    # def get(id):
    #     with open(embedding_config, 'r') as file:
    #         config = json.load(file)
    #         for item in config:
    #             if item['id'] == id:
    #                 return Embedding(data=item)

    @staticmethod
    def modify_weight(weight):
        with open(embedding_config, 'r') as file:
            config = json.load(file)
        for j, modality in enumerate(config['modalities']):
            modality['weight'] = weight[j]
        with open(embedding_config, 'w') as file:
            json.dump(config, file, indent=4)

    # def modify_weight(weight):
    #     with open(embedding_config, 'r') as file:
    #         config = json.load(file)
    #     for i, item in enumerate(config):
    #         if item['id'] == self.id:
    #             for j, modality in enumerate(item['modalities']):
    #                 modality['weight'] = weight[j]
    #     with open(embedding_config, 'w') as file:
    #         json.dump(config, file, indent=4)

    def learning(self):
        # clear tmp file, or else they will accumulate in the directory
        for filename in os.listdir(query_path):
            if filename.endswith('.tmp'):
                filepath = os.path.join(query_path, filename)
                os.remove(filepath)

        from vector_weight_learning import fvecs_converter
        with open(os.path.join(query_path, 'MitStates.txt'), 'r') as query_file:
            ground_truth = []

            for query_id, pair in enumerate(query_file):
                first, second = pair.strip().split(',')
                first = int(first)
                second = int(second)
                appear_time = {}
                # query -> modality -> modal
                dataset = [[] for _ in range(len(self.modalities))]
                # i = 0 => target modal, i > 0 => aux modal
                for i, modality in enumerate(self.modalities):
                    # identity name
                    name = f'{modality.encoder}'
                    for modal in modality.modals:
                        name += f'_{modal}'
                    # dataset_tmp = [[] for _ in range(len(modality.modals))]
                    for j, modal in enumerate(modality.modals):
                        dataset[i].append(os.path.join(query_path, f'{i}_{j}.tmp'))
                        with open(os.path.join(meta_path, f'{modal}.txt'), 'r') as file:
                            # extract data to i_j.tmp for embedding
                            for line_num, line in enumerate(file):
                                if i == 0:
                                    target_line = first
                                else:
                                    target_line = second
                                if line_num == target_line:
                                    with open(os.path.join(query_path, f'{i}_{j}.tmp'), 'a') as oFile:
                                        oFile.write(line)
                                        back_up = line
                                        break
                        # reopen file to reset the file_pointer
                        with open(os.path.join(meta_path, f'{modal}.txt'), 'r') as file:
                            if i > 0:
                                # object whose aux modal is same as second, will appear the same times as second
                                for line_num, line in enumerate(file):
                                    if line == back_up:
                                        if line_num in appear_time:
                                            appear_time[line_num] += 1
                                        else:
                                            appear_time[line_num] = 1

                ground_truth_part = []
                for key, value in appear_time.items():
                    if value == appear_time[second]:
                        ground_truth_part.append(key)
                ground_truth.append(ground_truth_part)

            # embedding query
            for i, modality in enumerate(self.modalities):
                # identity name
                name = f'{modality.encoder}'
                for modal in modality.modals:
                    name += f'_{modal}'
                if not os.path.exists(os.path.join(query_path, f'{name}.fvecs')):
                    encoder = get_encoder(modality.encoder)
                    encoder.encode(path=os.path.join(query_path, f'{name}.fvecs'), dataset=dataset[i], size=32)
            # embedding ground_truth
            fvecs_converter.to_ivecs(os.path.join(query_path, f'{self.id}_ground_truth.ivecs'), ground_truth)

        base_modal = []
        query_modal = []
        for i, modality in enumerate(self.modalities):
            name = f'{modality.encoder}'
            for modal in modality.modals:
                name += f'_{modal}'
            base_modal.append(fvecs_converter.fvecs_read(os.path.join(base_path, f'{name}.fvecs')))
            query_modal.append(fvecs_converter.fvecs_read(os.path.join(query_path, f'{name}.fvecs')))
        ground_truth = fvecs_converter.ivecs_read(os.path.join(query_path, f'{self.id}_ground_truth.ivecs'))

        from vector_weight_learning import learning
        weight = learning.weight_learning(base_modal=base_modal, query_modal=query_modal, ground_truth=ground_truth)
        self.modify_weight(weight)
        return weight

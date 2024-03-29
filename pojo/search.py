import os
import subprocess
import sys

import numpy as np

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


def get_search(retrieval_framework, selected_target, retrieval_number, retrieval_weight):
    if retrieval_framework == 'MR':
        return SearchMr(selected_target, retrieval_number, retrieval_weight)
    elif retrieval_framework == 'JE':
        return SearchJe(selected_target, retrieval_number, retrieval_weight)
    elif retrieval_framework == 'MUST':
        return SearchMust(selected_target, retrieval_number, retrieval_weight)


class BaseSearch:
    def __init__(self, selected_target, retrieval_number, retrieval_weight):
        self.selected_target = selected_target
        self.retrieval_number = retrieval_number
        self.retrieval_weight = retrieval_weight

    def search(self):
        selected_target = self.selected_target

        embedding = Embedding.get()
        for i, modality in enumerate(embedding.modalities):
            with open(os.path.join(search_path, f'{i}.fvecs'), "w") as file:
                file.truncate(0)

            if i == 0:
                base_id = 0
                # if the select_target does not exist at first, default use the first image
                # the weight of it will be set to 0
                if selected_target != -1:
                    res = []
                    with open(result_path, 'r') as file:
                        for line in file:
                            for item in line.strip().split(';'):
                                res.append(item.split(','))
                    base_id = int(res[selected_target][0])
                # read user's select data to {id_modal}.tmp file
                # get the meta data from meta.txt
                dataset = []
                encoder = get_encoder(modality.encoder)
                flag = False
                for modal in modality.modals:
                    dataset.append(os.path.join(search_path, f'0_{modal}.tmp'))
                    # if user selects an object, then use it directly in MR and MUST, while user can modify it in JE
                    # if user doesn't select, then use the 0-vector or the object they offer
                    if selected_target != -1:
                        flag = True
                        with open(os.path.join(meta_path, f'{modal}.txt'), 'r') as file:
                            for line_num, line in enumerate(file):
                                if line_num == base_id:
                                    with open(os.path.join(search_path, f'0_{modal}.tmp'), 'w') as f:
                                        f.write(line)
                                        break
                    else:
                        with (open(os.path.join(search_path, f'0_{modal}.tmp'), 'w') as w,
                              open(os.path.join(search_path, f'{modal}.tmp'), 'r') as r):
                            for line in r:
                                w.write(line)
                                flag = True
                if not flag:
                    self.retrieval_weight[i] = 0
                encoder.encode(path=os.path.join(search_path, f'{i}.fvecs'), dataset=dataset, size=1, flag=flag)
            else:
                encoder = get_encoder(modality.encoder)
                dataset = []
                flag = False
                for modal in modality.modals:
                    dataset.append(os.path.join(search_path, f'{modal}.tmp'))
                    with open(os.path.join(search_path, f'{modal}.tmp'), 'r') as r:
                        flag |= len(list(r))
                if not flag:
                    self.retrieval_weight[i] = 0
                encoder.encode(path=os.path.join(search_path, f'{i}.fvecs'), dataset=dataset, size=1, flag=flag)


class SearchMr(BaseSearch):
    def __init__(self, selected_target, retrieval_number, retrieval_weight):
        super().__init__(selected_target, retrieval_number, retrieval_weight)

    def search(self):
        super().search()
        selected_target = self.selected_target
        retrieval_number = self.retrieval_number
        retrieval_weight = self.retrieval_weight

        embedding = Embedding.get()
        # if selected_target == -1:
        #     retrieval_weight[0] = 0

        args = f'{len(embedding.modalities)}'
        for i, modality in enumerate(embedding.modalities):
            base_name = f'{modality.encoder}'
            for modal in modality.modals:
                base_name += f'_{modal}'
            args = args + ' ' + os.path.join(base_path, f"{base_name}.fvecs").replace('\\\\?\\', '')
            args = args + ' ' + os.path.join(search_path, f"{i}.fvecs").replace('\\\\?\\', '')
            args += f' {retrieval_weight[i]}'
        args += f' {retrieval_number}'
        args += ' ' + result_path.replace('\\\\?\\', '')
        args += ' ' + delete_id_path.replace('\\\\?\\', '')

        print(args)
        if sys.platform.startswith('win'):
            proc = subprocess.run(f'./indexing_and_search/search_mr.exe {args}')
        else:
            proc = subprocess.run(f'./indexing_and_search/search_mr {args}')
        if proc.returncode != 0:
            raise Exception(f'Search Error')

        res = []
        with open(result_path, 'r') as file:
            for line in file:
                for item in line.strip().split(';'):
                    res.append(item.split(','))

        ids = []
        images = []
        for item in res:
            images.append({
                'id': f'http://127.0.0.1:4523/m1/4132394-0-default/image?meta={0}&id={item[0]}'
            })
            ids.append(int(item[0]))

        # delete id who has been searched
        from vector_weight_learning import fvecs_converter
        delete = fvecs_converter.ivecs_read(delete_id_path)
        ids = np.array(ids)
        delete[0] = np.concatenate((delete[0], ids))
        fvecs_converter.to_ivecs(delete_id_path, delete)

        return images


class SearchJe(BaseSearch):
    def __init__(self, selected_target, retrieval_number, retrieval_weight):
        super().__init__(selected_target, retrieval_number, retrieval_weight)

    def search(self):
        selected_target = self.selected_target

        embedding = Embedding.get()
        modality = embedding.modalities[0]
        with open(os.path.join(search_path, f'0.fvecs'), "w") as file:
            file.truncate(0)

        base_id = -1
        if selected_target != -1:
            res = []
            with open(result_path, 'r') as file:
                for line in file:
                    for item in line.strip().split(';'):
                        res.append(item.split(','))
            base_id = int(res[selected_target][0])
        dataset = []
        encoder = get_encoder(modality.encoder)
        for modal in modality.modals:
            dataset.append(os.path.join(search_path, f'{modal}.tmp'))
            if os.path.exists(os.path.join(search_path, f'{modal}.tmp')):
                continue
            with open(os.path.join(meta_path, f'{modal}.txt'), 'r') as file:
                for line_num, line in enumerate(file):
                    if line_num == base_id:
                        with open(os.path.join(search_path, f'{modal}.tmp'), 'w') as f:
                            f.write(line)
                            break
        encoder.encode(path=os.path.join(search_path, f'0.fvecs'), dataset=dataset, size=1)

        # super().search()
        retrieval_number = self.retrieval_number

        embedding = Embedding.get()
        modality = embedding.modalities[0]
        base_name = f'{modality.encoder}'
        for modal in modality.modals:
            base_name += f'_{modal}'
        args = os.path.join(base_path, f"{base_name}.fvecs").replace('\\\\?\\', '')
        args = args + ' ' + os.path.join(search_path, "0.fvecs").replace('\\\\?\\', '')
        args += f' {retrieval_number}'
        args += ' ' + result_path.replace('\\\\?\\', '')
        args += ' ' + delete_id_path.replace('\\\\?\\', '')

        print(args)
        if sys.platform.startswith('win'):
            proc = subprocess.run(f'./indexing_and_search/search_je.exe {args}')
        else:
            proc = subprocess.run(f'./indexing_and_search/search_je {args}')
        if proc.returncode != 0:
            raise Exception(f'Search Error. The target modal cannot use JE framework to search.')

        res = []
        with open(result_path, 'r') as file:
            for line in file:
                for item in line.strip().split(';'):
                    res.append(item.split(','))

        ids = []
        images = []
        for item in res:
            images.append({
                # 'http://127.0.0.1:4523/m1/4132394-0-default/image?meta=2&id=0'
                'id': f'http://127.0.0.1:4523/m1/4132394-0-default/image?meta={0}&id={item[0]}'
                # 'similarity': round(float(item[1]), 4),
            })
            ids.append(int(item[0]))

        # delete id who has been searched
        from vector_weight_learning import fvecs_converter
        delete = fvecs_converter.ivecs_read(delete_id_path)
        ids = np.array(ids)
        delete[0] = np.concatenate((delete[0], ids))
        fvecs_converter.to_ivecs(delete_id_path, delete)

        return images


class SearchMust(BaseSearch):
    def __init__(self, selected_target, retrieval_number, retrieval_weight):
        super().__init__(selected_target, retrieval_number, retrieval_weight)

    def search(self):
        super().search()
        selected_target = self.selected_target
        retrieval_number = self.retrieval_number
        retrieval_weight = self.retrieval_weight

        embedding = Embedding.get()
        args = f'{len(embedding.modalities)}'
        for i, modality in enumerate(embedding.modalities):
            base_name = f'{modality.encoder}'
            for modal in modality.modals:
                base_name += f'_{modal}'
            args = args + ' ' + os.path.join(base_path, f"{base_name}.fvecs").replace('\\\\?\\', '')
            args = args + ' ' + os.path.join(search_path, f"{i}.fvecs").replace('\\\\?\\', '')
            args += f' {retrieval_weight[i]}'
        args += f' {retrieval_number}'
        args += ' ' + result_path.replace('\\\\?\\', '')
        args += ' ' + delete_id_path.replace('\\\\?\\', '')

        print(args)
        if sys.platform.startswith('win'):
            proc = subprocess.run(f'./indexing_and_search/search_must.exe {args}')
        else:
            proc = subprocess.run(f'./indexing_and_search/search_must {args}')
        if proc.returncode != 0:
            raise Exception(f'Search Error')

        res = []
        with open(result_path, 'r') as file:
            for line in file:
                for item in line.strip().split(';'):
                    res.append(item.split(','))

        ids = []
        images = []
        for item in res:
            images.append({
                'id': f'http://127.0.0.1:4523/m1/4132394-0-default/image?meta={0}&id={item[0]}'
            })
            ids.append(int(item[0]))

        # delete id who has been searched
        from vector_weight_learning import fvecs_converter
        delete = fvecs_converter.ivecs_read(delete_id_path)
        ids = np.array(ids)
        delete[0] = np.concatenate((delete[0], ids))
        fvecs_converter.to_ivecs(delete_id_path, delete)

        return images

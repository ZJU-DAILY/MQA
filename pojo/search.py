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


def get_search(retrieval_framework, selected_target, retrieval_number, retrieval_weight, index_method, index_path):
    if retrieval_framework == 'MR':
        return SearchMr(selected_target, retrieval_number, retrieval_weight, index_method, index_path)
    elif retrieval_framework == 'JE':
        return SearchJe(selected_target, retrieval_number, retrieval_weight, index_method, index_path)
    elif retrieval_framework == 'MUST':
        return SearchMust(selected_target, retrieval_number, retrieval_weight, index_method, index_path)


class BaseSearch:
    def __init__(self, selected_target, retrieval_number, retrieval_weight, index_method, index_path):
        self.selected_target = selected_target
        self.retrieval_number = retrieval_number
        self.retrieval_weight = retrieval_weight
        self.index_method = index_method
        self.index_path = index_path

    def get_selected_target(self):
        return self.selected_target

    def preprocessing(self):
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
                        with open(os.path.join(search_path, f'0_{modal}.tmp'), 'w') as w, open(
                                os.path.join(search_path, f'{modal}.tmp'), 'r') as r:
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

    def add_delete(self, res):
        ids = []
        for item in res:
            ids.append(int(item[0]))

        # delete id who has been searched
        from vector_weight_learning import fvecs_converter
        delete = fvecs_converter.ivecs_read(delete_id_path)
        ids = np.array(ids)
        delete[0] = np.concatenate((delete[0], ids))
        fvecs_converter.to_ivecs(delete_id_path, delete)

    def search(self):
        raise NotImplementedError

    @staticmethod
    def _search(args):
        print(args)
        if sys.platform.startswith('win'):
            proc = subprocess.run(f'./index_and_search/search.exe {args}')
        else:
            proc = subprocess.run(f'./index_and_search/search {args}')
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

        return res, images


class SearchMr(BaseSearch):
    def __init__(self, selected_target, retrieval_number, retrieval_weight, index_method, index_path):
        super().__init__(selected_target, retrieval_number, retrieval_weight, index_method, index_path)

    def search(self):
        super().preprocessing()

        retrieval_number = self.retrieval_number
        index_method = self.index_method
        index_path = self.index_path

        embedding = Embedding.get()
        results = []
        for i, modality in enumerate(embedding.modalities):
            args = '1'  # modality number
            base_name = f'{modality.encoder}'
            for modal in modality.modals:
                base_name += f'_{modal}'
            args = args + ' ' + os.path.join(base_path, f"{base_name}.fvecs").replace('\\\\?\\', '')
            args = args + ' ' + os.path.join(search_path, f"{i}.fvecs").replace('\\\\?\\', '')
            args += ' 1'  # retrieval weight
            args += f' {retrieval_number}'
            args += ' ' + result_path.replace('\\\\?\\', '')
            args += ' ' + delete_id_path.replace('\\\\?\\', '')
            args += f' {index_method}'
            args += ' ' + index_path.replace('\\\\?\\', '')
            res, images = super()._search(args)
            results.append(res)

        count_dict = {}
        for i, result in enumerate(results):
            for item in result:
                if item[1] == 'nan':
                    continue
                count_dict[item[0]] = count_dict.get(item[0], 0) + 1
        sorted_counts = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)

        res, images = [], []
        for item in sorted_counts[:self.retrieval_number]:
            res.append([item[0], str(item[1])])
            images.append({'id': f'http://127.0.0.1:4523/m1/4132394-0-default/image?meta={0}&id={item[0]}'})

        self.add_delete(res)
        return res, images


class SearchJe(BaseSearch):
    def __init__(self, selected_target, retrieval_number, retrieval_weight, index_method, index_path):
        super().__init__(selected_target, retrieval_number, retrieval_weight, index_method, index_path)

    def search(self):
        retrieval_number = self.retrieval_number
        selected_target = self.selected_target
        index_method = self.index_method
        index_path = self.index_path

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
            if os.path.getsize(os.path.join(search_path, f'{modal}.tmp')) != 0:
                continue
            with open(os.path.join(meta_path, f'{modal}.txt'), 'r') as file:
                for line_num, line in enumerate(file):
                    if line_num == base_id:
                        with open(os.path.join(search_path, f'{modal}.tmp'), 'w') as f:
                            f.write(line)
                            break
        encoder.encode(path=os.path.join(search_path, f'0.fvecs'), dataset=dataset, size=1)

        base_name = f'{modality.encoder}'
        for modal in modality.modals:
            base_name += f'_{modal}'
        args = '1'  # modal number
        args += ' ' + os.path.join(base_path, f"{base_name}.fvecs").replace('\\\\?\\', '')
        args += ' ' + os.path.join(search_path, "0.fvecs").replace('\\\\?\\', '')
        args += ' 1'  # weight
        args += f' {retrieval_number}'
        args += ' ' + result_path.replace('\\\\?\\', '')
        args += ' ' + delete_id_path.replace('\\\\?\\', '')
        args += f' {index_method}'
        args += ' ' + index_path.replace('\\\\?\\', '')

        res, images = super()._search(args)
        self.add_delete(res)
        return res, images


class SearchMust(BaseSearch):
    def __init__(self, selected_target, retrieval_number, retrieval_weight, index_method, index_path):
        super().__init__(selected_target, retrieval_number, retrieval_weight, index_method, index_path)

    def search(self):
        super().preprocessing()

        retrieval_number = self.retrieval_number
        retrieval_weight = self.retrieval_weight
        index_method = self.index_method
        index_path = self.index_path

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
        args += f' {index_method}'
        args += ' ' + index_path.replace('\\\\?\\', '')

        res, images = super()._search(args)
        self.add_delete(res)
        return res, images

from queue import PriorityQueue

import numpy as np
from tqdm import tqdm

from vector_weight_learning.distance import *

'''
num = modal[X].shape()[0]
dim = modal[X].shape()[1]
'''
"""
about the query_modal and base_modal, 
the first dimension is modality, such as base_modal[0] stands for the first modality which is called the target modality
the second is object, such as base_modal[0][0] stands for the first object of the target modality, stores a pack of test
generally, use the second dimension as a pack of vector in a whole
"""


class OptSet:
    def __init__(self, metric: Metric, thread_num=1, is_norm=True):
        self.query_modal = None
        self.base_modal = None
        self.weight = None
        if metric == Metric.L2_FLOAT:
            self.dist_op_float = EuclideanDistanceFloat()
        elif metric == Metric.IP_FLOAT:
            self.dist_op_float = InnerProductFloat()
        elif metric == Metric.AS_INT_SKIP_0:
            self.dist_op_int = AttributeSimilarityIntSkip0()
        elif metric == Metric.AS_INT:
            self.dist_op_int = AttributeSimilarityInt()
        elif metric == Metric.IP_FLOAT_AS_INT_SKIP_0:
            self.dist_op_float = InnerProductFloat()
            self.dist_op_int = AttributeSimilarityIntSkip0()
        elif metric == Metric.IP_FLOAT_AS_INT:
            self.dist_op_float = InnerProductFloat()
            self.dist_op_int = AttributeSimilarityInt()
        else:
            self.dist_op_float = InnerProductFloat()

        self.is_norm = is_norm
        self.thread_num = thread_num

    # store modal in this class and do normalization with float test
    def set_modal(self, base_modal, query_modal):
        self.base_modal = base_modal
        self.query_modal = query_modal
        if self.is_norm:
            for i, modal in enumerate(self.base_modal):
                if np.issubdtype(modal.dtype, np.floating):
                    norm = np.linalg.norm(modal, ord=2, axis=1)
                    norm_r = norm.reshape(-1, 1)
                    # if norm != 0:
                    self.base_modal[i] = self.base_modal[i] / norm_r
                    # self.base_modal[i] = (modal - np.mean(modal)) / np.std(modal)
            for i, modal in enumerate(self.query_modal):
                if np.issubdtype(modal.dtype, np.floating):
                    norm = np.linalg.norm(modal, ord=2, axis=1)
                    norm_r = norm.reshape(-1, 1)
                    # if norm != 0:
                    self.query_modal[i] = self.query_modal[i] / norm_r
                    # self.query_modal[i] = (modal - np.mean(modal)) / np.std(modal)

    # set the vector weight
    def set_weight(self, weight):
        self.weight = weight

    # get the number of objects in base_modal
    def get_base_num(self):
        return self.base_modal[0].shape[0]

    # get the number of queries in query_modal
    def get_query_num(self):
        return self.query_modal[0].shape[0]

    # get the dimension of object in base modal
    # and the dimension is more than query modal
    def get_dim(self, index):
        return self.base_modal[index].shape[1]

    # return the top-k nearest item's id and distance
    def top_k(self, k):
        id_list = []
        for i in tqdm(range(self.get_query_num()), desc='top_k', unit='queries'):
            query_id_list = []
            priority_queue = PriorityQueue()
            for j in range(self.get_base_num()):
                d = self.dist(i, j)
                # use negative priors to make it a max_heap
                priority_queue.put((d, j))
                # and keep the minimum k items stay in the heap
                if priority_queue.qsize() > k[i]:
                    priority_queue.get()
            for j in range(k[i]):
                priority, item = priority_queue.get()
                # change the order of to make it easily understood
                query_id_list.append([item, priority])
            id_list.append(query_id_list)
        return id_list

    def top_k_id(self, k):
        id_list = []
        for i in tqdm(range(self.get_query_num()), desc='top_k_id', unit='queries'):
            query_id_list = []
            priority_queue = PriorityQueue()
            for j in range(self.get_base_num()):
                d = self.dist(i, j)
                # use negative priors to make it a max_heap
                priority_queue.put((d, j))
                # and keep the minimum k items stay in the heap
                if priority_queue.qsize() > k[i]:
                    priority_queue.get()
            for j in range(k[i]):
                priority, item = priority_queue.get()
                # change the order of to make it easily understood
                query_id_list.append(item)
            id_list.append(query_id_list)
        return id_list

    def dist(self, query_id, base_id):
        d = 0
        for modal in range(len(self.query_modal)):
            if np.issubdtype(self.query_modal[modal].dtype, np.floating):
                d += (self.weight[modal]) * self.dist_op_float.calculate(self.query_modal[modal][query_id],
                                                                              self.base_modal[modal][base_id])
            else:
                d += (self.weight[modal]) * self.dist_op_int.calculate(self.query_modal[modal][query_id],
                                                                            self.base_modal[modal][base_id])
        return d

    # return the distance from a query to each base with id
    # the base_id is a list of id's indexed by base_modal
    def dist_by_id(self, modal, query_id, base_id):
        # print(base_id)
        dist_list = []
        query_modal = self.query_modal[modal]
        base_modal = self.base_modal[modal]
        for i in range(len(base_id)):
            bid = base_id[i]
            if np.issubdtype(query_modal.dtype, np.floating):
                d = self.dist_op_float.calculate(query_modal[query_id], base_modal[bid])
            else:
                d = self.dist_op_int.calculate(query_modal[query_id], base_modal[bid])
            dist_list.append(d)
        return dist_list

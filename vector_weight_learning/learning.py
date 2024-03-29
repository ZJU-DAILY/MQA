import torch
from vector_weight_learning.opt_set import *
from vector_weight_learning import model

k = model.k


def eva_recall(res, gt, k):
    count = 0
    for i in range(len(res)):
        res_intersection = list(set(res[i]) & set(gt[i]))
        count += min(len(res_intersection), k)
    return count / (k * len(res))


# check test format in modals
def check_modal(base_modal, query_modal, ground_truth):
    assert len(base_modal) > 0 and len(base_modal) == len(query_modal)
    base_num = base_modal[0].shape[0]
    query_num = query_modal[0].shape[0]
    for b_modal, q_modal in zip(base_modal, query_modal):
        assert b_modal.shape[0] == base_num
        assert q_modal.shape[0] == query_num
        assert b_modal.shape[1] == q_modal.shape[1]
    assert query_num == len(ground_truth)


def weight_learning(base_modal, query_modal, ground_truth):
    check_modal(base_modal, query_modal, ground_truth)
    modal_num = len(base_modal)
    ground_truth_len = [len(array) for array in ground_truth]

    # set modal and the calculation methods
    dist_opt = OptSet(Metric.IP_FLOAT, thread_num=1, is_norm=True)
    dist_opt.set_modal(base_modal=base_modal, query_modal=query_modal)
    print('Set modal Successfully.')

    # print(base_modal[0])

    # calculate initial ground truth dist
    ground_truth_modal_dist = []
    for modal in range(modal_num):
        dist = [dist_opt.dist_by_id(modal, i, ground_truth[i][:k]) for i in range(len(ground_truth))]
        ground_truth_modal_dist.append(dist)

    # set the aggregation function
    aggre_func = model.AggregationFunctionModel(modal_num=modal_num)
    criteria = model.AggregationFunctionLoss()
    # optimizer = model.AggregationFunctionOptimizer(params=aggre_func.omega, lr=1)
    optimizer = model.torch.optim.Adam(aggre_func.omega, lr=0.2, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                       amsgrad=False)
    print('Set aggregation function Successfully.')

    max_recall = 0
    opt = aggre_func.omega
    for epoch in range(2):
        print("Epoch: #", epoch)
        # initialize the weight
        # dist_opt.set_weight([(x[0]).item() for x in aggre_func.omega])
        dist_opt.set_weight([(x[0]).item() for x in aggre_func.omega])
        # k is same as the number of positive examples

        reca = dist_opt.top_k_id(ground_truth_len)

        print(reca[:1])
        reca_modal_dist = []
        for modal in range(modal_num):
            dist = [dist_opt.dist_by_id(modal, i, reca[i]) for i in range(len(reca))]
            reca_modal_dist.append(dist)

        phi_positive = []
        phi_negative = []
        # res = []

        for query_id in range(len(ground_truth)):
            cur_ground_truth_modal_dist = []
            for modal in range(modal_num):
                cur_ground_truth_modal_dist.append(torch.FloatTensor(ground_truth_modal_dist[modal][query_id]))

            phi_p = aggre_func(cur_ground_truth_modal_dist)
            phi_positive.append(phi_p)

            d = []
            for modal in range(modal_num):
                d.append(torch.FloatTensor(reca_modal_dist[modal][query_id]))

            phi_negative.append(aggre_func(d))

        optimizer.zero_grad()
        loss = criteria(phi_positive, phi_negative)
        # print(phi_positive)
        # print(phi_negative)
        # Back propagate gradients along the computational graph
        loss.backward()

        # for p in aggre_func.parameters():
        #     p.data.clamp_(0.001, 1.0)

        aggre_func.clamp_parameters()

        print("Parameter: ", [x.item() for x in aggre_func.omega])

        cur_recall = eva_recall(reca, ground_truth, k)
        print("Recall@%d: %lf" % (k, cur_recall))
        if cur_recall > max_recall:
            max_recall = cur_recall
            opt = [x.item() for x in aggre_func.omega]
        optimizer.step()

    print('Max Recall@%d: %lf' % (k, max_recall))
    print(opt)
    return opt

//
// Created by MaouSanta on 2024/2/18.
//

#ifndef INDEXING_AND_SEARCH_C7_ROUTING_GREEDY_H
#define INDEXING_AND_SEARCH_C7_ROUTING_GREEDY_H

#include "../c7_routing_basic.h"

class C7RoutingGreedy : public C7RoutingBasic {
public:
    DAnnFuncType prepareParam() override {
        auto *s_param = CGRAPH_GET_GPARAM(NPGSearchParam, GA_ALG_NPG_SEARCH_PARAM_KEY);
        auto *a_param = CGRAPH_GET_GPARAM(AlgParamBasic, GA_ALG_PARAM_BASIC_KEY);
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
        if (nullptr == model_ || nullptr == s_param) {
            return DAnnFuncType::ANN_PREPARE_ERROR;
        }

        num_ = model_->train_meta_modal_[0].num;

        for (int i = 0; i < model_->train_meta_modal_.size(); ++i) {
            dim_.emplace_back(model_->train_meta_modal_[i].dim);
            data_modal_.emplace_back(model_->train_meta_modal_[i].data);
            query_modal_.emplace_back(model_->search_meta_modal_[i].data);
        }

        search_L_ = s_param->search_L;
        K_ = a_param->top_k;
        query_id_ = s_param->query_id;
        if (Params.is_delete_id_) {
            delete_num_each_query_ = model_->delete_meta_.dim;
        }
        return DAnnFuncType::ANN_SEARCH;
    }

    CStatus search() override {
        auto s_param = CGRAPH_GET_GPARAM(NPGSearchParam, GA_ALG_NPG_SEARCH_PARAM_KEY);
        if (nullptr == s_param) {
            CGRAPH_RETURN_ERROR_STATUS("C7RoutingGreedy search get param failed")
        }

        std::vector<char> flags(num_, 0);

        int visited = 0;
        unsigned k = 0;
        while (k < (int) search_L_) {
            visited++;
            unsigned nk = search_L_;

            if (s_param->sp[k].flag_) {
                s_param->sp[k].flag_ = false;
                IDType n = s_param->sp[k].id_;

                for (unsigned int id: model_->graph_m_[n]) {
                    if (flags[id]) continue;
                    flags[id] = 1;
                    bool is_delete = false;
                    if (delete_num_each_query_) {
                        for (IDType kk = 0; kk < delete_num_each_query_; kk++) {
                            if (id == model_->delete_meta_.data[kk]) {
                                is_delete = true;
                                break;
                            }
                        }
                    }
                    if (is_delete) continue;

                    DistResType dist = 0;
                    std::vector<float *> modal_a;
                    std::vector<float *> modal_b;
                    for (int j = 0; j < query_modal_.size(); ++j) {
                        modal_a.emplace_back(data_modal_[j] + (size_t) id * dim_[j]);
                        modal_b.emplace_back(query_modal_[j] + (size_t) query_id_ * dim_[j]);
                    }
                    dist_op_.calculate(modal_a, modal_b, dim_, dim_, dist);

                    s_param->ss.insert(Neighbor(id, dist));
                    if (dist >= s_param->sp[search_L_ - 1].distance_) continue;
                    NeighborFlag nn(id, dist, true);
                    int r = InsertIntoPool(s_param->sp.data(), search_L_, nn);

                    if (r < nk) nk = r;
                }
            }
            nk <= k ? (k = nk) : (++k);
        }

//        std::cerr << "visited = " << visited << std::endl;

        for (size_t i = 0; i < K_; i++) {
            res_.emplace_back(s_param->sp[i].distance_, s_param->sp[i].id_);
        }
//        for (auto item: res_) {
//            std::cerr << item.first << "," << item.second << std::endl;
//        }
        return CStatus();
    }

    CStatus refreshParam() override {
        auto a_param = CGRAPH_GET_GPARAM(AlgParamBasic, GA_ALG_PARAM_BASIC_KEY);
        CGRAPH_ASSERT_NOT_NULL(a_param)

        {
            CGRAPH_PARAM_WRITE_CODE_BLOCK(a_param)
            a_param->results_.emplace_back(res_);
        }
        return CStatus();
    }
};

#endif //INDEXING_AND_SEARCH_C7_ROUTING_GREEDY_H
/*
 * 2
F:\Code\Python\MaouSanta\MQA2\dataset\base\0_0.fvecs
F:\Code\Python\MaouSanta\MQA2\dataset\query\0_0.fvecs
0.22
F:\Code\Python\MaouSanta\MQA2\dataset\base\1_1.fvecs
F:\Code\Python\MaouSanta\MQA2\dataset\query\1_1.fvecs
0.78
5
F:\Code\Python\MaouSanta\MQA2\dataset\search\result.txt
F:\Code\Python\MaouSanta\MQA2\dataset\delete.ivecs
Flat
F:\Code\Python\MaouSanta\MQA2\dataset\index\Vamana_30_200_0_0.22_1_0.78.index
 */
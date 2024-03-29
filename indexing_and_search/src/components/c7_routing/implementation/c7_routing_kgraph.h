//
// Created by MaouSanta on 2024/2/18.
//

#ifndef INDEXING_AND_SEARCH_C7_ROUTING_KGRAPH_H
#define INDEXING_AND_SEARCH_C7_ROUTING_KGRAPH_H

#include "../c7_routing_basic.h"

class C7RoutingKGraph : public C7RoutingBasic {
public:
    DAnnFuncType prepareParam() override {
        auto *s_param = CGRAPH_GET_GPARAM(NPGSearchParam, GA_ALG_NPG_SEARCH_PARAM_KEY);
        auto *a_param = CGRAPH_GET_GPARAM(AlgParamBasic, GA_ALG_PARAM_BASIC_KEY);
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
        if (nullptr == model_ || nullptr == s_param) {
            return DAnnFuncType::ANN_PREPARE_ERROR;
        }

        num_ = model_->train_meta_modal1_.empty() ? model_->train_meta_modal2_[0].num : model_->train_meta_modal1_[0].num;

        for (int i = 0; i < model_->train_meta_modal1_.size(); ++i) {
            dim1_.emplace_back(model_->train_meta_modal1_[i].dim);
            data_modal1_.emplace_back(model_->train_meta_modal1_[i].data);
            query_modal1_.emplace_back(model_->search_meta_modal1_[i].data);
        }
        for (int i = 0; i < model_->train_meta_modal2_.size(); ++i) {
            dim2_.emplace_back(model_->train_meta_modal2_[i].dim);
            data_modal2_.emplace_back(model_->train_meta_modal2_[i].data);
            query_modal2_.emplace_back(model_->search_meta_modal2_[i].data);
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
            CGRAPH_RETURN_ERROR_STATUS("C7RoutingKGraph search get param failed")
        }

        std::vector<char> flags(num_, 0);
        res_.clear();

        unsigned k = 0;
        while (k < (int) search_L_) {
            unsigned nk = search_L_;

            if (s_param->sp[k].flag_) {
                s_param->sp[k].flag_ = false;
                IDType n = s_param->sp[k].id_;

                for (unsigned int id : model_->graph_m_[n]) {
                    if (flags[id]) continue;
                    flags[id] = 1;
                    bool is_delete = false;
                    if (delete_num_each_query_) {
                        for (IDType k = 0; k < delete_num_each_query_; k++) {
                            if (id == model_->delete_meta_.data[s_param->query_id * delete_num_each_query_ + k]) {
                                is_delete = true;
                                break;
                            }
                        }
                    }
                    if (is_delete) continue;

                    DistResType dist = 0;

                    std::vector<float*> modal1_a;
                    std::vector<float*> modal1_b;
                    for (int j = 0; j < query_modal1_.size(); ++j) {
                        modal1_a.emplace_back(data_modal1_[j] + (size_t)id * dim1_[j]);
                        modal1_b.emplace_back(query_modal1_[j] + (size_t)query_id_ * dim1_[j]);
                    }
                    std::vector<int*> modal2_a;
                    std::vector<int*> modal2_b;
                    for (int j = 0; j < query_modal2_.size(); ++j) {
                        modal2_a.emplace_back(data_modal2_[j] + (size_t)id * dim2_[j]);
                        modal2_b.emplace_back(query_modal2_[j] + (size_t)query_id_ * dim2_[j]);
                    }
                    dist_op_.calculate(modal1_a, modal1_b, dim1_, dim1_,
                                       modal2_a, modal2_b, dim2_, dim2_,
                                       dist);

                    if (dist >= s_param->sp[search_L_ - 1].distance_) continue;
                    NeighborFlag nn(id, dist, true);
                    int r = InsertIntoPool(s_param->sp.data(), search_L_, nn);

                    if (r < nk) nk = r;
                }
            }
            nk <= k ? (k = nk) : (++k);
        }

        res_.reserve(K_);
        res_dist_.reserve(K_);
        for (size_t i = 0; i < K_; i++) {
            res_.emplace_back(s_param->sp[i].id_);
            res_dist_.emplace_back(s_param->sp[i].distance_);
        }
        return CStatus();
    }

    CStatus refreshParam() override {
        auto a_param = CGRAPH_GET_GPARAM(AlgParamBasic, GA_ALG_PARAM_BASIC_KEY);
        CGRAPH_ASSERT_NOT_NULL(a_param)

        {
            CGRAPH_PARAM_WRITE_CODE_BLOCK(a_param)
            a_param->results.push_back(res_);
            a_param->results_dist.push_back(res_dist_);
        }
        return CStatus();
    }
};

#endif //INDEXING_AND_SEARCH_C7_ROUTING_KGRAPH_H

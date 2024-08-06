//
// Created by MaouSanta on 2024/2/18.
//

#ifndef INDEXING_AND_SEARCH_C2_CANDIDATE_FETCH_H
#define INDEXING_AND_SEARCH_C2_CANDIDATE_FETCH_H

#include "../c2_candidate_basic.h"

class C2CandidateFetch : public C2CandidateBasic {
public:
    DAnnFuncType prepareParam() override {
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
        auto *t_param = CGRAPH_GET_GPARAM(NPGTrainParam, GA_ALG_NPG_TRAIN_PARAM_KEY)
        if (nullptr == model_ || nullptr == t_param) {
            return DAnnFuncType::ANN_PREPARE_ERROR;
        }

        num_ = model_->train_meta_modal_[0].num;
        for (const auto &modal: model_->train_meta_modal_) {
            dim_.emplace_back(modal.dim);
            data_modal_.emplace_back(modal.data);
        }

        L_ = t_param->L_candidate;
        return DAnnFuncType::ANN_TRAIN;
    }

    CStatus train() override {
        graph_pool_.resize(num_);

#pragma omp parallel for num_threads(Params.thread_num_) schedule(dynamic) default(none)
        for (IDType i = 0; i < num_; i++) {
            std::vector<bool> flags(num_, false);
            flags[i] = true;
            for (unsigned j = 0; j < model_->graph_n_[i].size(); j++) {
                if (flags[j]) continue;
                flags[j] = true;
                graph_pool_[i].emplace_back(model_->graph_n_[i][j]);
            }

            std::set<std::pair<DistResType, IDType>> dist_id;
            for (unsigned j = 0; j < model_->graph_n_[i].size(); j++) {
                IDType nid = model_->graph_n_[i][j].id_;
                for (unsigned nn = 0; nn < model_->graph_n_[nid].size(); nn++) {
                    IDType nnid = model_->graph_n_[nid][nn].id_;
                    if (flags[nnid]) continue;
                    flags[nnid] = true;
                    DistResType dist = 0;

                    std::vector<float *> modal_a;
                    std::vector<float *> modal_b;
                    for (int tmp = 0; tmp < data_modal_.size(); ++tmp) {
                        const auto& modal = data_modal_[tmp];
                        modal_a.emplace_back(modal + (size_t) i * dim_[tmp]);
                        modal_b.emplace_back(modal + (size_t) nnid * dim_[tmp]);
                    }
                    dist_op_.calculate(modal_a, modal_b, dim_, dim_, dist);
                    if (graph_pool_[i].size() >= L_) break;
                }
                if (graph_pool_[i].size() >= L_) break;
            }
        }

        model_->graph_n_ = graph_pool_;
        eval_quality(graph_pool_, model_->sample_points_, model_->knn_set_);
        return CStatus();
    }

    CStatus refreshParam() override {
        std::vector<std::vector<Neighbor>>().swap(graph_pool_);
        return CStatus();
    }

protected:
    std::vector<std::vector<Neighbor>> graph_pool_;
};

#endif //INDEXING_AND_SEARCH_C2_CANDIDATE_FETCH_H

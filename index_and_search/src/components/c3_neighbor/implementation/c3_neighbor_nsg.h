//
// Created by MaouSanta on 2024/2/18.
//

#ifndef INDEXING_AND_SEARCH_C3_NEIGHBOR_NSG_H
#define INDEXING_AND_SEARCH_C3_NEIGHBOR_NSG_H

#include "../c3_neighbor_basic.h"

#if GA_USE_OPENMP

#include <omp.h>

#endif

class C3NeighborNSG : public C3NeighborBasic {
public:
    DAnnFuncType prepareParam() override {
        auto *t_param = CGRAPH_GET_GPARAM(NPGTrainParam, GA_ALG_NPG_TRAIN_PARAM_KEY)
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
        if (nullptr == model_ || nullptr == t_param) {
            return DAnnFuncType::ANN_PREPARE_ERROR;
        }

        num_ = model_->train_meta_modal_[0].num;
        for (const auto &modal: model_->train_meta_modal_) {
            dim_.emplace_back(modal.dim);
            data_modal_.emplace_back(modal.data);
        }
        graph_pool_ = model_->graph_n_;
        C_ = t_param->C_neighbor;
        R_ = t_param->R_neighbor;

        return DAnnFuncType::ANN_TRAIN;
    }

    CStatus train() override {
        auto t_param = CGRAPH_GET_GPARAM(NPGTrainParam, GA_ALG_NPG_TRAIN_PARAM_KEY)
        CGRAPH_ASSERT_NOT_NULL(t_param)

#pragma omp parallel for num_threads(Params.thread_num_) schedule(dynamic) shared(threshold) default(none)
        for (unsigned i = 0; i < num_; i++) {
            unsigned start = 0;
            std::sort(graph_pool_[i].begin(), graph_pool_[i].end());
            std::vector<Neighbor> result;
            if (graph_pool_[i][start].id_ == i) start++;
            result.push_back(graph_pool_[i][start]);

            while (result.size() < R_ && (++start) < graph_pool_[i].size() && start < C_) {
                auto &p = graph_pool_[i][start];
                bool occlude = false;
                for (const auto &res: result) {
                    if (p.id_ == res.id_) {
                        occlude = true;
                        break;
                    }
                    DistResType djk = 0;
                    std::vector<float *> modal_a;
                    std::vector<float *> modal_b;
                    for (int tmp = 0; tmp < data_modal_.size(); ++tmp) {
                        const auto& modal = data_modal_[tmp];
                        modal_a.emplace_back(modal + (size_t) res.id_ * dim_[tmp]);
                        modal_b.emplace_back(modal + (size_t) p.id_ * dim_[tmp]);
                    }
                    dist_op_.calculate(modal_a, modal_b, dim_, dim_, djk);

                    if (djk < p.distance_) {
                        occlude = true;
                        break;
                    }
                }
                if (!occlude) result.push_back(p);
            }

            model_->graph_n_[i] = result;
        }
//        eval_quality(model_->graph_n_, model_->sample_points_, model_->knn_set_);

        return CStatus();
    }

    CStatus refreshParam() override {
        std::vector<std::vector<Neighbor>>().swap(graph_pool_);
        return CStatus();
    }

protected:
    std::vector<std::vector<Neighbor>> graph_pool_;
};

#endif //INDEXING_AND_SEARCH_C3_NEIGHBOR_NSG_H

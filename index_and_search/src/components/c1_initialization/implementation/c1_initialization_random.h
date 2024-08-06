//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_C1_INITIALIZATION_RANDOM_H
#define INDEXING_AND_SEARCH_C1_INITIALIZATION_RANDOM_H

#include "../c1_initialization_basic.h"

class C1InitializationRandom : public C1InitializationBasic {
public:
    DAnnFuncType prepareParam() override {
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
        auto* t_param = CGRAPH_GET_GPARAM(NPGTrainParam, GA_ALG_NPG_TRAIN_PARAM_KEY);
        if (nullptr == t_param || nullptr == model_) {
            return DAnnFuncType::ANN_PREPARE_ERROR;
        }

        num_ = model_->train_meta_modal_[0].num;
        for (const auto& modal: model_->train_meta_modal_) {
            dim_.emplace_back(modal.dim);
            data_modal_.emplace_back(modal.data);
        }
        out_degree_ = t_param->k_init_graph;
        graph_pool_.resize(num_);
        return DAnnFuncType::ANN_TRAIN;
    }


    CStatus train() override {
#pragma omp parallel for num_threads(Params.thread_num_) schedule(dynamic) default(none)
        for (size_t i = 0; i < num_; i++) {
            std::vector<IDType> cur(out_degree_ + 1);
            GenRandomID(cur.data(), num_, cur.size());
            for (unsigned j = 0; j < out_degree_; j++) {
                DistResType dist = 0;
                IDType id = cur[j];
                if (id == i) continue;

                std::vector<float *> modal_a;
                std::vector<float *> modal_b;
                for (int tmp = 0; tmp < data_modal_.size(); ++tmp) {
                    auto modal = data_modal_[tmp];
                    modal_a.emplace_back(modal + (size_t) i * dim_[tmp]);
                    modal_b.emplace_back(modal + (size_t) id * dim_[tmp]);
                }
                dist_op_.calculate(modal_a, modal_b, dim_, dim_, dist);
                graph_pool_[i].emplace_back(id, dist);
            }
        }
        model_->graph_n_ = graph_pool_;
        eval_quality(graph_pool_, model_->sample_points_, model_->knn_set_);
        return CStatus();
    }

protected:
    std::vector<std::vector<Neighbor>> graph_pool_;
};

#endif //INDEXING_AND_SEARCH_C1_INITIALIZATION_RANDOM_H

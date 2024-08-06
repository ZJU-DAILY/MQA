//
// Created by MaouSanta on 2024/5/23.
//

#ifndef INDEXING_AND_SEARCH_C6_SEED_CENTROID_H
#define INDEXING_AND_SEARCH_C6_SEED_CENTROID_H

#include "../c6_seed_basic.h"

class C6SeedCentroid : public C6SeedBasic {
public:
    DAnnFuncType prepareParam() override {
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY)
        auto *s_param = CGRAPH_GET_GPARAM(NPGSearchParam, GA_ALG_NPG_SEARCH_PARAM_KEY)
        if (nullptr == model_ || nullptr == s_param) {
            return DAnnFuncType::ANN_PREPARE_ERROR;
        }

        num_ = model_->train_meta_modal_[0].num;
        for (const auto &modal: model_->train_meta_modal_) {
            dim_.emplace_back(modal.dim);
        }
        search_L_ = s_param->search_L;
        return DAnnFuncType::ANN_SEARCH;
    }

    CStatus search() override {
        auto *s_param = CGRAPH_GET_GPARAM(NPGSearchParam, GA_ALG_NPG_SEARCH_PARAM_KEY)
        if (nullptr == s_param) {
            CGRAPH_RETURN_ERROR_STATUS("C6SeedRandom search find param failed")
        }

        s_param->sp.resize(search_L_ + 1);
        std::vector<IDType> init_ids(search_L_);
        std::vector<bool> flags(num_, false);

        unsigned tmp_l = 0;
        for (; tmp_l < search_L_ && tmp_l < model_->graph_m_[model_->ep_].size(); ++tmp_l) {
            init_ids[tmp_l] = model_->graph_m_[model_->ep_][tmp_l];
            flags[init_ids[tmp_l]] = true;
        }
        std::random_device rd;
        std::mt19937 rng(rd());
        for (; tmp_l < search_L_; tmp_l++) {
            unsigned id = rng() % num_;
            while (flags[id]) id = rng() % num_;
            flags[id] = true;
            init_ids[tmp_l] = id;
        }
        for (unsigned i = 0; i < init_ids.size(); ++i) {
            unsigned id = init_ids[i];
            DistResType dist = 0;
            std::vector<float *> modal_a;
            std::vector<float *> modal_b;
            for (int j = 0; j < model_->search_meta_modal_.size(); ++j) {
                auto train_modal = model_->train_meta_modal_[j].data;
                auto search_modal = model_->search_meta_modal_[j].data;
                modal_a.emplace_back(train_modal + (size_t) id * dim_[j]);
                modal_b.emplace_back(search_modal + (size_t) s_param->query_id * dim_[j]);
            }
            dist_op_.calculate(modal_a, modal_b, dim_, dim_, dist);
            s_param->sp[i] = NeighborFlag(id, dist, true);
        }
        std::sort(s_param->sp.begin(), s_param->sp.end());

        return CStatus();
    }

};

#endif //INDEXING_AND_SEARCH_C6_SEED_CENTROID_H

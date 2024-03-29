//
// Created by MaouSanta on 2024/2/18.
//

#ifndef INDEXING_AND_SEARCH_C6_SEED_KGRAPH_H
#define INDEXING_AND_SEARCH_C6_SEED_KGRAPH_H

#include "../c6_seed_basic.h"
#include <cstring>

class C6SeedKGraph : public C6SeedBasic {
public:
    DAnnFuncType prepareParam() override {
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY)
        auto *s_param = CGRAPH_GET_GPARAM(NPGSearchParam, GA_ALG_NPG_SEARCH_PARAM_KEY)
        if (nullptr == model_ || nullptr == s_param) {
            return DAnnFuncType::ANN_PREPARE_ERROR;
        }

        num_ = model_->train_meta_modal1_.empty() ? model_->train_meta_modal2_[0].num : model_->train_meta_modal1_[0].num;
        for (const auto& modal1: model_->train_meta_modal1_) {
            dim1_.emplace_back(modal1.dim);
        }
        for (const auto& modal2: model_->train_meta_modal2_) {
            dim2_.emplace_back(modal2.dim);
        }

        search_L_ = s_param->search_L;
        return DAnnFuncType::ANN_SEARCH;
    }

    CStatus search() override {
        auto *s_param = CGRAPH_GET_GPARAM(NPGSearchParam, GA_ALG_NPG_SEARCH_PARAM_KEY)
        if (nullptr == s_param) {
            CGRAPH_RETURN_ERROR_STATUS("C6SeedKGraph search find param failed")
        }

        s_param->sp.reserve(search_L_ + 1);
        std::vector<IDType> init_ids(search_L_);

        GenRandomID(init_ids.data(), num_, search_L_);
        std::vector<char> flags(num_);
        memset(flags.data(), 0, num_ * sizeof(char));
        for (unsigned i = 0; i < search_L_; i++) {
            IDType id = init_ids[i];
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
            for (int j = 0; j < model_->search_meta_modal1_.size(); ++j) {
                auto train_modal1 = model_->train_meta_modal1_[j].data;
                auto search_modal1 = model_->search_meta_modal1_[j].data;
                modal1_a.emplace_back(train_modal1 + (size_t)id * dim1_[j]);
                modal1_b.emplace_back(search_modal1 + (size_t)s_param->query_id * dim1_[j]);
            }
            std::vector<int*> modal2_a;
            std::vector<int*> modal2_b;
            for (int j = 0; j < model_->search_meta_modal2_.size(); ++j) {
                auto train_modal2 = model_->train_meta_modal2_[j].data;
                auto search_modal2 = model_->search_meta_modal2_[j].data;
                modal2_a.emplace_back(train_modal2 + (size_t)id * dim2_[j]);
                modal2_b.emplace_back(search_modal2 + (size_t)s_param->query_id * dim2_[j]);
            }
            dist_op_.calculate(modal1_a, modal1_b, dim1_, dim1_,
                               modal2_a, modal2_b, dim2_, dim2_,
                               dist);

            s_param->sp[i] = NeighborFlag(id, dist, true);
        }

        std::sort(s_param->sp.begin(), s_param->sp.end());
        return CStatus();
    }

};

#endif //INDEXING_AND_SEARCH_C6_SEED_KGRAPH_H

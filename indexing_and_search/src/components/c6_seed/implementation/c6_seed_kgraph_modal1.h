//
// Created by MaouSanta on 2024/2/18.
//

#ifndef INDEXING_AND_SEARCH_C6_SEED_KGRAPH_MODAL1_H
#define INDEXING_AND_SEARCH_C6_SEED_KGRAPH_MODAL1_H

/*
#include "../c6_seed_basic.h"
#include <cstring>

class C6SeedKGraphModal1 : public C6SeedBasic {
public:
    DAnnFuncType prepareParam() override {
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY)
        auto *s_param = CGRAPH_GET_GPARAM(NPGSearchParam, GA_ALG_NPG_SEARCH_PARAM_KEY)
        if (nullptr == model_ || nullptr == s_param) {
            return DAnnFuncType::ANN_PREPARE_ERROR;
        }

        query_num_ = model_->train_meta_modal1_[0].num;
        for (const auto& modal1: model_->train_meta_modal1_) {
            dim1_.emplace_back(modal1.dim);
        }
        search_L_ = s_param->search_L;

        std::vector<float> w2_(model_->train_meta_modal2_.size(), 0);
        dist_op_.set_weight(Params.w1_, w2_);
        return DAnnFuncType::ANN_SEARCH;
    }

    CStatus search() override {
        auto *s_param = CGRAPH_GET_GPARAM(NPGSearchParam, GA_ALG_NPG_SEARCH_PARAM_KEY)
        if (nullptr == s_param) {
            CGRAPH_RETURN_ERROR_STATUS("C6SeedKGraph search find param failed")
        }

        s_param->sp_modal1.reserve(search_L_ + 1);
        std::vector<IDType> init_ids(search_L_);

        GenRandomID(init_ids.data(), query_num_, search_L_);
        std::vector<char> flags(query_num_);
        memset(flags.data(), 0, query_num_ * sizeof(char));

        s_param->sp_modal1.emplace_back();
        for (unsigned i = 0; i < search_L_; i++) {
            IDType id = init_ids[i];
            bool is_delete = false;
            if (delete_num_each_query_) {
                for (IDType k = 0; k < delete_num_each_query_; k++) {
                    if (id == model_->delete_meta_.data[s_param->modal1_query_id * delete_num_each_query_ + k]) {
                        is_delete = true;
                        break;
                    }
                }
            }
            if (is_delete) continue;
            DistResType dist = 0;


            dist_op_.calc_vec1(model_->search_meta_modal1_.data + (s_param->modal1_query_id * (size_t)dim1_),
                               model_->train_meta_modal1_.data + (size_t)id * (size_t)dim1_,
                               dim1_, dim1_, dist);


            s_param->sp_modal1.back()[i] = NeighborFlag(id, dist, true);
        }

        std::sort(s_param->sp_modal1.begin(), s_param->sp_modal1.begin() + search_L_);
        return CStatus();
    }
};
*/

#endif //INDEXING_AND_SEARCH_C6_SEED_KGRAPH_MODAL1_H

//
// Created by MaouSanta on 2024/2/18.
//

#ifndef INDEXING_AND_SEARCH_C6_SEED_KGRAPH_MODAL2_H
#define INDEXING_AND_SEARCH_C6_SEED_KGRAPH_MODAL2_H

#include "../c6_seed_basic.h"
#include <cstring>
/*
class C6SeedKGraphModal2 : public C6SeedBasic {
public:
    DAnnFuncType prepareParam() override {
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY)
        auto *s_param = CGRAPH_GET_GPARAM(NPGSearchParam, GA_ALG_NPG_SEARCH_PARAM_KEY)
        if (nullptr == model_ || nullptr == s_param) {
            return DAnnFuncType::ANN_PREPARE_ERROR;
        }

        query_num_ = model_->train_meta_modal2_.num;
        dim2_ = model_->train_meta_modal2_.dim;
        search_L_ = s_param->search_L;
        dist_op_.set_weight(0, Params.w2_);
        return DAnnFuncType::ANN_SEARCH;
    }

    CStatus search() override {
        auto *s_param = CGRAPH_GET_GPARAM(NPGSearchParam, GA_ALG_NPG_SEARCH_PARAM_KEY)
        if (nullptr == s_param) {
            CGRAPH_RETURN_ERROR_STATUS("C6SeedKGraph search find param failed")
        }

        s_param->sp_modal2.reserve(search_L_ + 1);
        std::vector<IDType> init_ids(search_L_);

        GenRandomID(init_ids.data(), query_num_, search_L_);
        std::vector<char> flags(query_num_);
        memset(flags.data(), 0, query_num_ * sizeof(char));
        for (unsigned i = 0; i < search_L_; i++) {
            IDType id = init_ids[i];
            bool is_delete = false;
            if (delete_num_each_query_) {
                for (IDType k = 0; k < delete_num_each_query_; k++) {
                    if (id == model_->delete_meta_.data[s_param->modal2_query_id * delete_num_each_query_ + k]) {
                        is_delete = true;
                        break;
                    }
                }
            }
            if (is_delete) continue;
            DistResType dist = 0;
            dist_op_.calc_vec2(model_->search_meta_modal2_.data + (s_param->modal2_query_id * (size_t)dim2_),
                               model_->train_meta_modal2_.data + (size_t)id * (size_t)dim2_,
                               dim2_, dim2_, dist);
            s_param->sp_modal2[i] = NeighborFlag(id, dist, true);
        }

        std::sort(s_param->sp_modal2.begin(), s_param->sp_modal2.begin() + search_L_);
        return CStatus();
    }
};
*/
#endif //INDEXING_AND_SEARCH_C6_SEED_KGRAPH_MODAL2_H

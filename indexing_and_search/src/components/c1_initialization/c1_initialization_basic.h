//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_C1_INITIALIZATION_BASIC_H
#define INDEXING_AND_SEARCH_C1_INITIALIZATION_BASIC_H

#include <string>
#include <cassert>

#include "../components_basic.h"
#include "../../utils/utils.h"

class C1InitializationBasic : public ComponentsBasic {
protected:
    CStatus init() override {
        auto *model_param = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY)
        if (!model_param) {
            CGRAPH_RETURN_ERROR_STATUS("C1InitializationBasic init get param failed")
        }

        CStatus status;
        IDType num = 0;
        model_param->train_meta_modal1_.resize(Params.GA_ALG_BASE_MODAL1_PATH_.size());
        for (int i = 0; i < Params.GA_ALG_BASE_MODAL1_PATH_.size(); ++i) {
            status += model_param->train_meta_modal1_[i].load(Params.GA_ALG_BASE_MODAL1_PATH_[i], Params.is_norm_modal1_[i]);
            num = std::max(num, model_param->train_meta_modal1_[i].num);
        }
        model_param->train_meta_modal2_.resize(Params.GA_ALG_BASE_MODAL2_PATH_.size());
        for (int i = 0; i < Params.GA_ALG_BASE_MODAL2_PATH_.size(); ++i) {
            status += model_param->train_meta_modal2_[i].load(Params.GA_ALG_BASE_MODAL2_PATH_[i], Params.is_norm_modal2_[i]);
            num = std::max(num, model_param->train_meta_modal2_[i].num);
        }

        printf("[PARAM] vector num: %u\n", num);
        for (int i = 0; i < Params.GA_ALG_BASE_MODAL1_PATH_.size(); ++i) {
            assert(model_param->train_meta_modal1_[i].num == num);
            printf("[PATH] modal 1[%d] vector path: %s\n", i, Params.GA_ALG_BASE_MODAL1_PATH_[i]);
            printf("[PARAM] modal 1[%d] vector dim: %u\n", i, model_param->train_meta_modal1_[i].dim);
        }
        for (int i = 0; i < Params.GA_ALG_BASE_MODAL2_PATH_.size(); ++i) {
            assert(model_param->train_meta_modal2_[i].num == num);
            printf("[PATH] modal 2[%d] vector path: %s\n", i, Params.GA_ALG_BASE_MODAL2_PATH_[i]);
            printf("[PARAM] modal 2[%d] vector dim: %u\n", i, model_param->train_meta_modal2_[i].dim);
        }

        return CStatus();
    }

protected:
    unsigned out_degree_ = 0;        // out-degree of initial graph
    IDType cur_num_ = 0;           // data id being processed
};

#endif //INDEXING_AND_SEARCH_C1_INITIALIZATION_BASIC_H

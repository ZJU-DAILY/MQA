//
// Created by MaouSanta on 2024/2/18.
//

#ifndef INDEXING_AND_SEARCH_C6_SEED_BASIC_H
#define INDEXING_AND_SEARCH_C6_SEED_BASIC_H

#include "../components_basic.h"
#include "../../utils/utils.h"
#include "../../elements/elements.h"

class C6SeedBasic : public ComponentsBasic {
protected:
    CStatus init() override {
        auto *s_param = CGRAPH_GET_GPARAM(NPGSearchParam, GA_ALG_NPG_SEARCH_PARAM_KEY)
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY)
        if (nullptr == model_ || nullptr == s_param) {
            CGRAPH_RETURN_ERROR_STATUS("C6SeedBasic get param failed")
        }

        CStatus status;
        IDType query_num = 0;
        model_->search_meta_modal_.resize(Params.GA_ALG_QUERY_MODAL_PATH_.size());
        for (int i = 0; i < Params.GA_ALG_QUERY_MODAL_PATH_.size(); ++i) {
            status += model_->search_meta_modal_[i].load(Params.GA_ALG_QUERY_MODAL_PATH_[i]);
            query_num = std::max(query_num, model_->search_meta_modal_[i].num);
        }
        for (int i = 0; i < Params.GA_ALG_QUERY_MODAL_PATH_.size(); ++i) {
            assert(model_->search_meta_modal_[i].num == query_num);
            printf("[PATH] modal [%d] query vector path: %s\n", i, model_->search_meta_modal_[i].file_path.c_str());
        }

        printf("[PARAM] query vector num: %u\n", query_num);
        IDType num = 0;
        model_->train_meta_modal_.resize(Params.GA_ALG_BASE_MODAL_PATH_.size());
        for (int i = 0; i < Params.GA_ALG_BASE_MODAL_PATH_.size(); ++i) {
            status += model_->train_meta_modal_[i].load(Params.GA_ALG_BASE_MODAL_PATH_[i]);
            num = std::max(num, model_->train_meta_modal_[i].num);
        }

        for (int i = 0; i < Params.GA_ALG_BASE_MODAL_PATH_.size(); ++i) {
            assert(model_->train_meta_modal_[i].num == num);
            assert(model_->train_meta_modal_[i].dim == model_->search_meta_modal_[i].dim);
            printf("[PARAM] modal [%d] query vector dim: %u\n", i, model_->train_meta_modal_[i].dim);
        }

        if (Params.is_delete_id_) {
            status += model_->delete_meta_.load(Params.GA_ALG_DELETE_ID_PATH_, 0);
            delete_num_each_query_ = model_->delete_meta_.dim;
        }

        if (!status.isOK()) {
            CGRAPH_RETURN_ERROR_STATUS("C6SeedBasic load param failed")
        }

        return CStatus();
    }

protected:
    unsigned search_L_; // candidate pool size for search
    unsigned delete_num_each_query_ = 0;
};


#endif //INDEXING_AND_SEARCH_C6_SEED_BASIC_H

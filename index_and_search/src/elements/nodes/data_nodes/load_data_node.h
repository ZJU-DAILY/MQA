//
// Created by MaouSanta on 2024/5/22.
//

#ifndef INDEXING_AND_SEARCH_LOAD_DATA_NODE_H
#define INDEXING_AND_SEARCH_LOAD_DATA_NODE_H


#include <fstream>
#include <cassert>

#include "../../elements_define.h"
#include "../../../../CGraph/src/CGraph.h"

class LoadDataNode : public CGraph::GNode {
public:
    CStatus init() override {
        auto m_param = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY)
        CGRAPH_ASSERT_NOT_NULL(m_param)
        CStatus status;
        IDType num = 0;
        m_param->train_meta_modal_.resize(Params.GA_ALG_BASE_MODAL_PATH_.size());
        for (int i = 0; i < Params.GA_ALG_BASE_MODAL_PATH_.size(); ++i) {
            status += m_param->train_meta_modal_[i].load(Params.GA_ALG_BASE_MODAL_PATH_[i]);
            num = std::max(num, m_param->train_meta_modal_[i].num);
        }

        printf("[PARAM] vector num: %u\n", num);
        for (int i = 0; i < Params.GA_ALG_BASE_MODAL_PATH_.size(); ++i) {
            assert(m_param->train_meta_modal_[i].num == num);
            printf("[PATH] modal [%d] vector path: %s\n", i, Params.GA_ALG_BASE_MODAL_PATH_[i]);
            printf("[PARAM] modal [%d] vector dim: %u\n", i, m_param->train_meta_modal_[i].dim);
        }
        return status;
    }

    CStatus run() override {
        CGRAPH_EMPTY_FUNCTION
    }
};

#endif //INDEXING_AND_SEARCH_LOAD_DATA_NODE_H

//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_TRAIN_REGION_H
#define INDEXING_AND_SEARCH_TRAIN_REGION_H

#include "../../../CGraph/src/CGraph.h"
#include "../elements_define.h"

class TrainRegion : public CGraph::GRegion {
public:
    CBool isHold() override {
        auto *m_param = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
        if (nullptr == m_param) {
            CGRAPH_THROW_EXCEPTION("TrainRegion get param exception")
        }

        m_param->cur_id_++;
        IDType num = m_param->search_meta_modal1_.empty() ? m_param->search_meta_modal2_[0].num
                                                          : m_param->search_meta_modal1_[0].num;
        return m_param->cur_id_ < num;
    }

    CStatus crashed(const CException& ex) override {
        return CStatus(ex.what());
    }
};

#endif //INDEXING_AND_SEARCH_TRAIN_REGION_H

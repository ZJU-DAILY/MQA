//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_SEARCH_CLUSTER_MODAL2_H
#define INDEXING_AND_SEARCH_SEARCH_CLUSTER_MODAL2_H

#include "../../../CGraph/src/CGraph.h"
#include "../elements_define.h"
/*
class SearchClusterModal2 : public CGraph::GCluster {
public:
    CBool isHold() override {
        auto *m_param = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
        auto *s_param = CGRAPH_GET_GPARAM(NPGSearchParam, GA_ALG_NPG_SEARCH_PARAM_KEY);
        if (nullptr == m_param || nullptr == s_param) {
            CGRAPH_THROW_EXCEPTION("SearchClusterModal2 isHold get param failed")
        }

        s_param->modal2_query_id++;
        return s_param->modal2_query_id < m_param->search_meta_modal2_.num;
    }


    CStatus crashed(const CException& ex) override {
        return CStatus(ex.what());
    }
};
*/
#endif //INDEXING_AND_SEARCH_SEARCH_CLUSTER_MODAL2_H

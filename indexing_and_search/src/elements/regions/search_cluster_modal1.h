//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_SEARCH_CLUSTER_MODAL1_H
#define INDEXING_AND_SEARCH_SEARCH_CLUSTER_MODAL1_H

#include "../../../CGraph/src/CGraph.h"
#include "../elements_define.h"
/*
class SearchClusterModal1 : public CGraph::GCluster {
public:
    CBool isHold() override {
        auto *m_param = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
        auto *s_param = CGRAPH_GET_GPARAM(NPGSearchParam, GA_ALG_NPG_SEARCH_PARAM_KEY);
        if (nullptr == m_param || nullptr == s_param) {
            CGRAPH_THROW_EXCEPTION("SearchClusterModal1 isHold get param failed")
        }

        s_param->modal1_query_id++;
        return s_param->modal1_query_id < m_param->search_meta_modal1_.num;
    }


    CStatus crashed(const CException& ex) override {
        return CStatus(ex.what());
    }
};
*/
#endif //INDEXING_AND_SEARCH_SEARCH_CLUSTER_MODAL1_H

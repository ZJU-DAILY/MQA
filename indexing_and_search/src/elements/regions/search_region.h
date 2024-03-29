//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_SEARCH_REGION_H
#define INDEXING_AND_SEARCH_SEARCH_REGION_H

#include "../../../CGraph/src/CGraph.h"
#include "../elements_define.h"

class SearchRegion : public CGraph::GCluster {
public:
    CBool isHold() override {
        auto *m_param = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
        auto *s_param = CGRAPH_GET_GPARAM(NPGSearchParam, GA_ALG_NPG_SEARCH_PARAM_KEY);
        if (nullptr == m_param || nullptr == s_param) {
            /**
             * throw exception, CGraph can catch this exception automic
             */
            CGRAPH_THROW_EXCEPTION("SearchRegion isHold get param failed")
        }

        s_param->query_id++;
        IDType num = m_param->search_meta_modal1_.empty() ? m_param->search_meta_modal2_[0].num
                                                          : m_param->search_meta_modal1_[0].num;
        return s_param->query_id < num;
    }


    CStatus crashed(const CException &ex) override {
        /**
         * this function can help you catch exception,
         * and you can transfer your exception info into other error code & error info
         */
        return CStatus(ex.what());
    }
};

#endif //INDEXING_AND_SEARCH_SEARCH_REGION_H

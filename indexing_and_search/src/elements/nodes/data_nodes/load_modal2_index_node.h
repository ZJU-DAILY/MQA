//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_LOAD_MODAL2_INDEX_NODE_H
#define INDEXING_AND_SEARCH_LOAD_MODAL2_INDEX_NODE_H

#include <fstream>

#include "../../elements_define.h"
#include "../../../../CGraph/src/CGraph.h"

/*
class LoadModal2IndexNode : public CGraph::GNode {
public:
    CStatus init() override {
        auto m_param = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
        CGRAPH_ASSERT_NOT_NULL(m_param)

        std::ifstream f_in(Params.GA_ALG_MODAL2_INDEX_PATH_, std::ios::binary);
        if (!f_in.is_open()) {
            CGRAPH_RETURN_ERROR_STATUS("load graph error!");
        }

        m_param->graph_m2_.emplace_back();
        while (!f_in.eof()) {
            unsigned GK = 0;
            f_in.read((char *) &GK, sizeof(unsigned));
            if (f_in.eof()) break;
            std::vector<IDType> tmp(GK);
            f_in.read((char *) tmp.data(), GK * sizeof(IDType));
            m_param->graph_m2_.back().emplace_back(tmp);
        }
        f_in.close();
        return CStatus();
    }

    CStatus run() override {
        CGRAPH_EMPTY_FUNCTION
    }
};
 */

#endif //INDEXING_AND_SEARCH_LOAD_MODAL2_INDEX_NODE_H

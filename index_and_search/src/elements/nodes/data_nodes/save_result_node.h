//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_SAVE_RESULT_NODE_H
#define INDEXING_AND_SEARCH_SAVE_RESULT_NODE_H

#include <fstream>
#include <random>

#include "../../../../CGraph/src/CGraph.h"
#include "../../elements_define.h"

class SaveResultNode : public CGraph::GNode {
public:
    CStatus run() override {
        CGRAPH_EMPTY_FUNCTION
    }

    CStatus destroy() override {
        auto *s_param = CGRAPH_GET_GPARAM(AlgParamBasic, GA_ALG_PARAM_BASIC_KEY);
        CGRAPH_ASSERT_NOT_NULL(s_param)

        std::ofstream f_out(Params.GA_ALG_RESULT_PATH_, std::ios::out);
        for (auto& result: s_param->results_) {
            int GK = (int) result.size();
            for (int j = 0; j < GK; ++j) {
                // id, dist
                f_out << result[j].second << ',' << result[j].first;
                f_out << ";\n"[j == GK - 1];
            }
        }
        f_out.close();
        return CStatus();
    }
};

#endif //INDEXING_AND_SEARCH_SAVE_RESULT_NODE_H

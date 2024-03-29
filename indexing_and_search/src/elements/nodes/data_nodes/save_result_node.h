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

//        std::mt19937 gen(42);
//        std::uniform_real_distribution<double> dis(-0.02, 0.02);

        std::ofstream f_out(Params.GA_ALG_RESULT_PATH_, std::ios::out);
        for (int i = 0; i < s_param->results.size(); ++i) {
            auto &result = s_param->results[i];
            auto &dist = s_param->results_dist[i];
            int GK = (int) result.size();
            for (int j = 0; j < GK; ++j) {
                f_out << result[j] << ',' << dist[j];
//                f_out << result[j] << "," << std::min(1.0, dist[j] + dis(gen));
                j == (GK - 1) ? f_out << "\n" : f_out << ";";
            }
        }
        f_out.close();
        return CStatus();
    }
};

#endif //INDEXING_AND_SEARCH_SAVE_RESULT_NODE_H

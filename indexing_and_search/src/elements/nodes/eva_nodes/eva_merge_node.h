//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_EVA_MERGE_NODE_H
#define INDEXING_AND_SEARCH_EVA_MERGE_NODE_H

#include "../../elements_define.h"
#include <unordered_set>

class EvaMergeNode : public CGraph::GNode {
public:
    CStatus init() override {
        auto m_param = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY)
        CGRAPH_ASSERT_NOT_NULL(m_param)
        query_num_ = m_param->search_meta_modal1_.empty() ? m_param->search_meta_modal2_[0].num
                                                          : m_param->search_meta_modal1_[0].num;

        return CStatus();
    }

    CStatus run() override {
        auto *s_param = CGRAPH_GET_GPARAM(AlgParamBasic, GA_ALG_PARAM_BASIC_KEY);
        if (nullptr == s_param) {
            CGRAPH_RETURN_ERROR_STATUS("EvaMergeNode run get param failed")
        }

        top_k_ = s_param->top_k;
        s_param->results.resize(top_k_);
        s_param->results_dist.resize(top_k_);
        // iterate for all queries
        for (IDType i = 0; i < query_num_; i++) {
            std::priority_queue<std::pair<std::pair<unsigned, DistResType1>, IDType>> res_set;
//            // accumulate all objects' IP appeared in the results_modal
            std::map<IDType, std::pair<unsigned, DistResType1>> ma;
//            // iterate all object in results_modal, fetch the top-k IP holders
            for (IDType modal = 0; modal < s_param->results_modal1.size(); ++modal) {
                for (IDType j = 0; j < s_param->results_modal1[0][i].size(); ++j) {
                    auto& id = s_param->results_modal1[modal][i][j];
                    auto& dist = s_param->results_dist1[modal][i][j];
                    auto& weight = Params.w1_[modal];
                    if (ma.find(id) == ma.end()) {
                        ma[id] = std::make_pair(1, dist * weight);
                    } else {
                        auto& tmp = ma[id];
                        tmp.first ++;
                        tmp.second += dist * weight;
                    }
                }
            }
            for (auto [key, value]: ma) {
                // dist, id
                res_set.emplace(value, key);
            }
            for (IDType j = 0; j < top_k_ && !res_set.empty(); j++) {
                std::pair<std::pair<unsigned, DistResType1>, IDType> p = res_set.top();
                res_set.pop();
                s_param->results[i].emplace_back(p.second);
                s_param->results_dist[i].emplace_back(p.first.second);
            }
        }

        return CStatus();
    }

private:
    IDType query_num_;
    unsigned top_k_;
    unsigned candi_top_k_;
};

#endif //INDEXING_AND_SEARCH_EVA_MERGE_NODE_H

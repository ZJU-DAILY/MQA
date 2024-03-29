//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_SAVE_INDEX_NODE_H
#define INDEXING_AND_SEARCH_SAVE_INDEX_NODE_H

#include <fstream>

#include "../../../../CGraph/src/CGraph.h"
#include "../../elements_define.h"

class SaveIndexNode : public CGraph::GNode {
public:
    CStatus run() override {
        CGRAPH_EMPTY_FUNCTION
    }

    CStatus destroy() override {
        auto m_param = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
        CGRAPH_ASSERT_NOT_NULL(m_param)

        std::fstream f_out(Params.GA_ALG_INDEX_PATH_, std::ios::binary | std::ios::out);

        IDType num = m_param->train_meta_modal1_.empty() ? m_param->train_meta_modal2_[0].num
                                                         : m_param->train_meta_modal1_[0].num;

        for (IDType i = 0; i < num; i++) {
            auto GK = (unsigned) m_param->cut_graph_[i].size();
            std::vector<IDType> vec;
            vec.reserve(GK);
            for (unsigned j = 0; j < GK; j++) {
                vec.push_back(m_param->cut_graph_[i][j].id_);
            }
            f_out.write((char *) &GK, sizeof(unsigned));
            f_out.write((char *) vec.data(), GK * sizeof(IDType));
            std::vector<Neighbor>().swap(m_param->cut_graph_[i]);
        }
        std::vector<std::vector<Neighbor>>().swap(m_param->cut_graph_);
        f_out.close();
        return CStatus();
    }
};

#endif //INDEXING_AND_SEARCH_SAVE_INDEX_NODE_H

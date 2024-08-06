//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_LOAD_INDEX_NODE_H
#define INDEXING_AND_SEARCH_LOAD_INDEX_NODE_H

#include <fstream>

#include "../../elements_define.h"
#include "../../../../CGraph/src/CGraph.h"
#include "../../../components/components_basic.h"

class LoadIndexNode : public CGraph::GNode {
public:
    CStatus init() override {
        auto m_param = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY)
        CGRAPH_ASSERT_NOT_NULL(m_param)
        std::ifstream f_in(Params.GA_ALG_INDEX_PATH_, std::ios::binary);
        if (!f_in.is_open()) {
            CGRAPH_RETURN_ERROR_STATUS("load graph error!")
        }
        f_in.read((char *) &m_param->ep_, sizeof(unsigned));
        while (!f_in.eof()) {
            unsigned GK = 0;
            f_in.read((char *) &GK, sizeof(unsigned));
            if (f_in.eof()) break;
            std::vector<IDType> tmp(GK);
            f_in.read((char *) tmp.data(), GK * sizeof(IDType));
            m_param->graph_m_.push_back(tmp);
        }
        f_in.close();
        return CStatus();
    }

    CStatus run() override {
//        auto m_param = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY)
//        CGRAPH_ASSERT_NOT_NULL(m_param)
//        num_ = m_param->train_meta_modal_[0].num;
//        for (const auto &modal: m_param->train_meta_modal_) {
//            dim_.emplace_back(modal.dim);
//            data_modal_.emplace_back(modal.data);
//        }
//        std::cerr << m_param->graph_m_.size() << std::endl;
//        m_param->graph_n_.resize(m_param->graph_m_.size());
//        for (int i = 0; i < m_param->graph_m_.size(); ++i) {
//            for (int j = 0; j < m_param->graph_m_[i].size(); ++j) {
//                auto id = m_param->graph_m_[i][j];
//
//                DistResType dist = 0;
//                std::vector<float *> modal_a;
//                std::vector<float *> modal_b;
//                for (int tmp = 0; tmp < data_modal_.size(); ++tmp) {
//                    auto modal = data_modal_[tmp];
//                    modal_a.emplace_back(modal + (size_t) i * dim_[tmp]);
//                    modal_b.emplace_back(modal + (size_t) id * dim_[tmp]);
//                }
//                m_param->graph_n_[i].emplace_back(id, dist);
//            }
//        }
//        eval_quality(m_param->graph_n_, m_param->sample_points_, m_param->knn_set_);
//        return CStatus();
        CGRAPH_EMPTY_FUNCTION
    }

protected:
    std::vector<VecValType *> data_modal_ = {};
    size_t num_ = 0;
    std::vector<unsigned> dim_ = {};
    DistCalcType dist_op_;
};

#endif //INDEXING_AND_SEARCH_LOAD_INDEX_NODE_H

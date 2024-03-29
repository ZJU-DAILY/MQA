//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_C1_INITIALIZATION_KGRAPH_H
#define INDEXING_AND_SEARCH_C1_INITIALIZATION_KGRAPH_H

#include "../c1_initialization_basic.h"

class C1InitializationKGraph : public C1InitializationBasic {
public:
    DAnnFuncType prepareParam() override {
        auto *t_param = CGRAPH_GET_GPARAM(NPGTrainParam, GA_ALG_NPG_TRAIN_PARAM_KEY)
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY)
        if (nullptr == model_ || nullptr == t_param) {
            return DAnnFuncType::ANN_PREPARE_ERROR;
        }

        if (!model_->train_meta_modal1_.empty()) {
            num_ = model_->train_meta_modal1_[0].num;
        } else {
            num_ = model_->train_meta_modal2_[0].num;
        }

        for (const auto& modal1: model_->train_meta_modal1_) {
            dim1_.emplace_back(modal1.dim);
            data_modal1_.emplace_back(modal1.data);
        }
        for (const auto& modal2: model_->train_meta_modal2_) {
            dim2_.emplace_back(modal2.dim);
            data_modal2_.emplace_back(modal2.data);
        }
        out_degree_ = t_param->k_init_graph;
        model_->graph_n_.reserve(num_);
        return DAnnFuncType::ANN_TRAIN;
    }

    CStatus train() override {
        graph_neigh_.clear();
        graph_neigh_.reserve(out_degree_);
        std::vector<IDType> neighbor_id(out_degree_);
        GenRandomID(neighbor_id.data(), num_, out_degree_);
        for (const IDType &id: neighbor_id) {
            if (id != cur_num_) {
                DistResType dist = 0;

                std::vector<float*> modal1_id;
                std::vector<float*> modal1_cur;
                for (int i = 0; i < data_modal1_.size(); ++i) {
                    auto modal1 = data_modal1_[i];
                    modal1_id.emplace_back(modal1 + (size_t)id * dim1_[i]);
                    modal1_cur.emplace_back(modal1 + (size_t)cur_num_ * dim1_[i]);
                }
                std::vector<int*> modal2_id;
                std::vector<int*> modal2_cur;
                for (int i = 0; i < data_modal2_.size(); ++i) {
                    auto modal2 = data_modal2_[i];
                    modal2_id.emplace_back(modal2 + (size_t)id * dim2_[i]);
                    modal2_cur.emplace_back(modal2 + (size_t)cur_num_ * dim2_[i]);
                }
                dist_op_.calculate(modal1_id, modal1_cur, dim1_, dim1_,
                                   modal2_id, modal2_cur, dim2_, dim2_,
                                   dist);
                graph_neigh_.emplace_back(id, dist);
            }
        }

        return CStatus();
    }

    CStatus refreshParam() override {
        {
            CGRAPH_PARAM_WRITE_CODE_BLOCK(model_)
            model_->graph_n_.emplace_back(graph_neigh_);
        }
        return CStatus();
    }

    CBool isHold() override {
        cur_num_++;
        if (cur_num_ >= num_) {
            CGraph::CGRAPH_ECHO("kgraph init complete!");
        }
        return cur_num_ < num_;
    }

protected:
    std::vector<Neighbor> graph_neigh_;   // temp neighbor
};

#endif //INDEXING_AND_SEARCH_C1_INITIALIZATION_KGRAPH_H

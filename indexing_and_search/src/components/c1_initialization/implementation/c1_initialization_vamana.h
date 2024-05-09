//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_C1_INITIALIZATION_VAMANA_H
#define INDEXING_AND_SEARCH_C1_INITIALIZATION_VAMANA_H

#include "../c1_initialization_basic.h"

class C1InitializationVamana : public C1InitializationBasic {
public:
    DAnnFuncType prepareParam() override {
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
        auto* t_param = CGRAPH_GET_GPARAM(NPGTrainParam, GA_ALG_NPG_TRAIN_PARAM_KEY);
        if (nullptr == t_param || nullptr == model_) {
            return DAnnFuncType::ANN_PREPARE_ERROR;
        }

        num_ = model_->train_meta_modal1_.empty() ? model_->train_meta_modal2_[0].num : model_->train_meta_modal1_[0].num;
        for (const auto& modal1: model_->train_meta_modal1_) {
            dim1_.emplace_back(modal1.dim);
            data_modal1_.emplace_back(modal1.data);
        }
        for (const auto& modal2: model_->train_meta_modal2_) {
            dim2_.emplace_back(modal2.dim);
            data_modal2_.emplace_back(modal2.data);
        }
        out_degree_ = t_param->k_init_graph;
//        model_->graph_n_.reserve(num_);
        model_->graph_n_.resize(num_);
        return DAnnFuncType::ANN_TRAIN;
    }


    CStatus train() override {
        graph_neigh_.clear();
//        graph_neigh_.reserve(out_degree_);
        graph_neigh_.resize(out_degree_);
        std::vector<IDType> neighbor_id(out_degree_);
        GenRandomID(neighbor_id.data(), num_, out_degree_);
        for (const IDType &id: neighbor_id) {
            if (id != cur_num_) {
                graph_neigh_.emplace_back(id);
            }
        }

        return CStatus();
    }


    CStatus refreshParam() override {
        {
            CGRAPH_PARAM_WRITE_CODE_BLOCK(model_)
            model_->graph_m_.emplace_back(graph_neigh_);
        }
        return CStatus();
    }

    CBool isHold() override {
        cur_num_++;
        if (cur_num_ >= num_) {
            CGraph::CGRAPH_ECHO("vamana init complete!");
        }
        return cur_num_ < num_;
    }

protected:
    std::vector<IDType> graph_neigh_;   // temp neighbor
};

#endif //INDEXING_AND_SEARCH_C1_INITIALIZATION_VAMANA_H

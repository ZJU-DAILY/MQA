//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_C1_INITIALIZATION_KGRAPH_OPENMP_H
#define INDEXING_AND_SEARCH_C1_INITIALIZATION_KGRAPH_OPENMP_H

#if GA_USE_OPENMP

#include <omp.h>

#endif

#include "../c1_initialization_basic.h"

class C1InitializationKGraphOpenMP : public C1InitializationBasic {
public:
    DAnnFuncType prepareParam() override {
        auto *t_param = CGRAPH_GET_GPARAM(NPGTrainParam, GA_ALG_NPG_TRAIN_PARAM_KEY);
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
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
        model_->graph_n_.resize(num_);
        return DAnnFuncType::ANN_TRAIN;
    }

    CStatus train() override {

#pragma omp parallel for num_threads(Params.thread_num_) schedule(dynamic) default(none)

        for (IDType i = 0; i < num_; i++) {
            model_->graph_n_[i].reserve(out_degree_);
            std::vector<IDType> neighbor_id(out_degree_);
            GenRandomID(neighbor_id.data(), num_, out_degree_);
            for (const IDType &id: neighbor_id) {
                if (id != cur_num_) {
                    DistResType dist = 0;
                    dist_op_.calculate(data_modal1_ + (size_t)(id * dim1_),
                                       data_modal1_ + (size_t)cur_num_ * dim1_,
                                       dim1_, dim1_,
                                       data_modal2_ + (size_t)(id * dim2_),
                                       data_modal2_ + (size_t)cur_num_ * dim2_,
                                       dim2_, dim2_, dist);
                    model_->graph_n_[i].emplace_back(id, dist);
                }
            }
        }

#if GA_USE_OPENMP
        CGraph::CGRAPH_ECHO("kgraph openmp init complete!");
#else
        CGraph::CGRAPH_ECHO("kgraph no openmp init complete!");
#endif
        return CStatus();
    }
};

#endif //INDEXING_AND_SEARCH_C1_INITIALIZATION_KGRAPH_OPENMP_H

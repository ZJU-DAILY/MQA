//
// Created by MaouSanta on 2024/2/18.
//

#ifndef INDEXING_AND_SEARCH_C3_NEIGHBOR_NSG_H
#define INDEXING_AND_SEARCH_C3_NEIGHBOR_NSG_H

#include "../c3_neighbor_basic.h"

class C3NeighborNSG : public C3NeighborBasic {
public:
    DAnnFuncType prepareParam() override {
        auto *t_param = CGRAPH_GET_GPARAM(NPGTrainParam, GA_ALG_NPG_TRAIN_PARAM_KEY)
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
        if (nullptr == model_ || nullptr == t_param) {
            return DAnnFuncType::ANN_PREPARE_ERROR;
        }

        num_ = 0;
        for (const auto& modal1: model_->train_meta_modal1_) {
            num_ = std::max(num_, (size_t)modal1.num);
            dim1_.emplace_back(modal1.dim);
            data_modal1_.emplace_back(modal1.data);
        }
        for (const auto& modal2: model_->train_meta_modal2_) {
            num_ = std::max(num_, (size_t)modal2.num);
            dim2_.emplace_back(modal2.dim);
            data_modal2_.emplace_back(modal2.data);
        }

        C_ = t_param->C_neighbor;
        R_ = t_param->R_neighbor;

        model_->cut_graph_.resize(num_);

        return DAnnFuncType::ANN_TRAIN;
    }

    CStatus train() override {
        unsigned start = 0;
        std::sort(model_->pool_.begin(), model_->pool_.end());
        result_.clear();
        if (model_->pool_[start].id_ == cur_id_) start++;
        result_.push_back(model_->pool_[start]);

        while (result_.size() < R_ && (++start) < model_->pool_.size() && start < C_) {
            auto &p = model_->pool_[start];
            unsigned occlude = false;
            for (auto &t: result_) {
                if (p.id_ == t.id_) {
                    occlude = true;
                    break;
                }
                DistResType djk = 0;
                dist_op_.calc(data_modal1_, data_modal2_, dim1_, dim2_, t.id_, p.id_, djk);
                if (djk < p.distance_) {
                    occlude = true;
                    break;
                }
            }
            if (!occlude) result_.push_back(p);
        }
        return CStatus();
    }


    CStatus refreshParam() override {
        {
            CGRAPH_PARAM_WRITE_CODE_BLOCK(model_)
            model_->cut_graph_.push_back(result_);
        }
        return CStatus();
    }
};

#endif //INDEXING_AND_SEARCH_C3_NEIGHBOR_NSG_H

//
// Created by MaouSanta on 2024/2/18.
//

#ifndef INDEXING_AND_SEARCH_C2_CANDIDATE_NSSG_H
#define INDEXING_AND_SEARCH_C2_CANDIDATE_NSSG_H

#include "../c2_candidate_basic.h"

class C2CandidateNSSG : public C2CandidateBasic {
public:
    DAnnFuncType prepareParam() override {
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
        auto *t_param = CGRAPH_GET_GPARAM(NPGTrainParam, GA_ALG_NPG_TRAIN_PARAM_KEY)
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
        cur_id_ = model_->cur_id_;

        L_ = t_param->L_candidate;
        return DAnnFuncType::ANN_TRAIN;
    }

    CStatus train() override {
        model_->pool_.clear();    // model_ cannot be nullptr, because it is checked in prepareParam()
        model_->pool_.reserve(L_);
        std::vector<unsigned> flags(num_, 0);
        flags[cur_id_] = true;

        for (unsigned j = 0; j < model_->graph_n_[cur_id_].size(); j++) {
            if (flags[j]) continue;
            flags[j] = true;
            IDType nid = model_->graph_n_[cur_id_][j].id_;
            float ndist = model_->graph_n_[cur_id_][j].distance_;
            model_->pool_.emplace_back(nid, ndist);
        }

        for (unsigned j = 0; j < model_->graph_n_[cur_id_].size(); j++) {
            IDType nid = model_->graph_n_[cur_id_][j].id_;
            for (auto &nn : model_->graph_n_[nid]) {
                IDType nnid = nn.id_;    // nnid is the id of neighbor's neighbor
                if (flags[nnid]) continue;
                flags[nnid] = true;
                DistResType dist = 0;
                dist_op_.calc(data_modal1_, data_modal2_, dim1_, dim2_, nnid, cur_id_, dist);
                model_->pool_.emplace_back(nnid, dist);
                if (model_->pool_.size() >= L_) break;
            }
            if (model_->pool_.size() >= L_) break;
        }
        return CStatus();
    }
};

#endif //INDEXING_AND_SEARCH_C2_CANDIDATE_NSSG_H

//
// Created by MaouSanta on 2024/5/22.
//

#ifndef INDEXING_AND_SEARCH_EVA_GENERATE_SAMPLE_NODE_H
#define INDEXING_AND_SEARCH_EVA_GENERATE_SAMPLE_NODE_H

#include "../../../components/components_basic.h"

class EvaGenerateSampleNode : public CGraph::GNode {
public:
    CStatus init() override {
        CGRAPH_EMPTY_FUNCTION
    }

    CStatus run() override {
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY)
        CGRAPH_ASSERT_NOT_NULL(model_)
        num_ = model_->train_meta_modal_[0].num;
        for (const auto &modal: model_->train_meta_modal_) {
            dim_.emplace_back(modal.dim);
            data_modal_.emplace_back(modal.data);
        }

        model_->sample_points_.resize(sample_num_);
        model_->knn_set_.resize(sample_num_);

        auto &s = model_->sample_points_;
        auto &g = model_->knn_set_;
        GenRandomID(s.data(), num_, s.size());
#pragma omp parallel for num_threads(Params.thread_num_) schedule(dynamic) shared(s, g) default(none)
        for (unsigned i = 0; i < s.size(); i++) {
            std::vector<Neighbor> cur;
            cur.reserve(num_);
            for (unsigned j = 0; j < num_; j++) {
                DistResType dist = 0;

                std::vector<float *> modal_a;
                std::vector<float *> modal_b;
                for (int tmp = 0; tmp < data_modal_.size(); ++tmp) {
                    auto modal = data_modal_[tmp];
                    modal_a.emplace_back(modal + (size_t) s[i] * dim_[tmp]);
                    modal_b.emplace_back(modal + (size_t) j * dim_[tmp]);
                }
                dist_op_.calculate(modal_a, modal_b, dim_, dim_, dist);
                cur.emplace_back(j, dist);
            }

            std::partial_sort(cur.begin(), cur.begin() + sample_num_, cur.end());
            for (unsigned j = 0; j < sample_num_; j++) {
                g[i].emplace_back(cur[j].id_);
            }
        }
        return CStatus();
    }

protected:
    unsigned sample_num_ = 100;
    AnnsModelParam *model_ = nullptr;
    std::vector<VecValType *> data_modal_ = {};
    size_t num_ = 0;
    std::vector<unsigned> dim_ = {};
    DistCalcType dist_op_;
};

#endif //INDEXING_AND_SEARCH_EVA_GENERATE_SAMPLE_NODE_H

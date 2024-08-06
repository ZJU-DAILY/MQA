//
// Created by MaouSanta on 2024/5/21.
//

#ifndef INDEXING_AND_SEARCH_C4_PREPROCESSING_CENTROID_H
#define INDEXING_AND_SEARCH_C4_PREPROCESSING_CENTROID_H

#include "../c4_preprocessing_basic.h"

class C4PreprocessingCentroid : public C4PreprocessingBasic {
public:
    DAnnFuncType prepareParam() override {
        auto *t_param = CGRAPH_GET_GPARAM(NPGTrainParam, GA_ALG_NPG_TRAIN_PARAM_KEY)
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY)
        if (nullptr == model_ || nullptr == t_param) {
            return DAnnFuncType::ANN_PREPARE_ERROR;
        }
        num_ = model_->train_meta_modal_[0].num;
        for (const auto &modal: model_->train_meta_modal_) {
            dim_.emplace_back(modal.dim);
            data_modal_.emplace_back(modal.data);
        }
        return DAnnFuncType::ANN_TRAIN;
    }

    // fetch the centroid of the target modal as enter point
    CStatus train() override {
        CStatus status;
        std::vector<float *> center(data_modal_.size());
        for (auto& item: center) {
            item = new float[num_ * dim_[0]];
        }
        for (unsigned i = 0; i < num_; ++i) {
            for (unsigned j = 0; j < dim_[0]; ++j) {
                for (unsigned tmp = 0; tmp < data_modal_.size(); ++tmp) {
                    const auto &modal = data_modal_[tmp];
                    center[tmp][j] += modal[i * dim_[0] + j];
                }
            }
        }

        for (unsigned j = 0; j < dim_[0]; ++j) {
            for (unsigned tmp = 0; tmp < data_modal_.size(); ++tmp) {
                center[tmp][j] /= num_;
            }
        }

        std::vector<NeighborFlag> sp;
        std::vector<Neighbor> pool;
        get_neighbors(center, sp, pool);
        model_->ep_ = sp[0].id_;

        return CStatus();
    }

protected:
    CStatus get_neighbors(std::vector<float *>& query_data, std::vector<NeighborFlag> &sp, std::vector<Neighbor> &full_set) {
        std::random_device rd;
        std::mt19937 rng(rd());
        std::vector<unsigned> init_ids;
        std::vector<bool> flags(num_, false);

        model_->ep_ = rng() % num_;
        for (unsigned i = 0; i < model_->graph_n_[model_->ep_].size() && i <= L_refine; ++i) {
            init_ids.emplace_back(model_->graph_n_[model_->ep_][i].id_);
            flags[init_ids[i]] = true;
        }

        while (init_ids.size() <= L_refine) {
            unsigned id = rng() % num_;
            while (flags[id]) id = rng() % num_;
            init_ids.emplace_back(id);
            flags[id] = true;
        }

        for (auto id: init_ids) {
            float dist = 0;
            std::vector<float *> modal_a;
            for (int tmp = 0; tmp < data_modal_.size(); ++tmp) {
                const auto &modal = data_modal_[tmp];
                modal_a.emplace_back(modal + (size_t) id * dim_[tmp]);
            }
            dist_op_.calculate(modal_a, query_data, dim_, dim_, dist);
            sp.emplace_back(id, dist, true);
        }
        std::sort(sp.begin(), sp.end());

        // search K-NN
        unsigned k = 0;
        while (k < L_refine) {
            unsigned nk = L_refine;

            if (sp[k].flag_) {
                sp[k].flag_ = false;

                for (const auto& [id, _]: model_->graph_n_[sp[k].id_]) {
                    if (flags[id]) continue;
                    flags[id] = true;

                    DistResType dist = 0;
                    std::vector<float *> modal_a;
                    for (int tmp = 0; tmp < data_modal_.size(); ++tmp) {
                        const auto &modal = data_modal_[tmp];
                        modal_a.emplace_back(modal + (size_t) id * dim_[tmp]);
                    }
                    dist_op_.calculate(modal_a, query_data, dim_, dim_, dist);

                    full_set.emplace_back(id, dist);
                    if (dist >= sp.back().distance_) continue;
                    NeighborFlag nn(id, dist, true);
                    int r = InsertIntoPool(sp.data(), L_refine, nn);

                    if (r < nk) nk = r;
                }
            }
            nk <= k ? (k = nk) : (++k);
        }
        return CStatus();
    }

protected:
    unsigned L_refine = 200;
};

#endif //INDEXING_AND_SEARCH_C4_PREPROCESSING_CENTROID_H

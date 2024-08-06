//
// Created by MaouSanta on 2024/5/22.
//

#ifndef INDEXING_AND_SEARCH_C2_CANDIDATE_GREEDY_H
#define INDEXING_AND_SEARCH_C2_CANDIDATE_GREEDY_H

#include "../c2_candidate_basic.h"

class C2CandidateGreedy : public C2CandidateBasic {
public:
    DAnnFuncType prepareParam() override {
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
        auto *t_param = CGRAPH_GET_GPARAM(NPGTrainParam, GA_ALG_NPG_TRAIN_PARAM_KEY)
        if (nullptr == model_ || nullptr == t_param) {
            return DAnnFuncType::ANN_PREPARE_ERROR;
        }

        num_ = model_->train_meta_modal_[0].num;
        for (const auto &modal: model_->train_meta_modal_) {
            dim_.emplace_back(modal.dim);
            data_modal_.emplace_back(modal.data);
        }

        L_ = t_param->L_candidate;
        return DAnnFuncType::ANN_TRAIN;
    }

    CStatus train() override {

        graph_pool_.resize(num_);
#pragma omp parallel for num_threads(Params.thread_num_) schedule(dynamic) default(none)
        for (IDType i = 0; i < num_; i++) {
            std::vector<NeighborFlag> sp;
            std::vector<Neighbor> pool;
            get_neighbors(i, sp, pool);
            std::sort(pool.begin(), pool.end());

            graph_pool_[i].assign(pool.begin(), pool.begin() + L_);
        }
        model_->graph_n_ = graph_pool_;
        eval_quality(model_->graph_n_, model_->sample_points_, model_->knn_set_);
        return CStatus();
    }

    void get_neighbors(unsigned query_id, std::vector<NeighborFlag> &sp, std::vector<Neighbor> &full_set) {
        unsigned L = L_;
        sp.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        std::vector<bool> flags(num_, false);

        L = 0;
        for (unsigned i = 0; i < model_->graph_n_[query_id].size() && i < L_; ++i) {
            init_ids[L++] = model_->graph_n_[query_id][i].id_;
            flags[init_ids[i]] = true;
        }

        std::random_device rd;
        std::mt19937 rng(rd());
        while (L < L_) {
            unsigned id = rng() % num_;
            while (flags[id] || id == query_id) id = rng() % num_;
            init_ids[L++] = id;
            flags[id] = true;
        }

        L = 0;
        for (auto id: init_ids) {
            float dist = 0;
            std::vector<float *> modal_a;
            std::vector<float *> modal_b;
            for (int tmp = 0; tmp < data_modal_.size(); ++tmp) {
                const auto &modal = data_modal_[tmp];
                modal_a.emplace_back(modal + (size_t) id * dim_[tmp]);
                modal_b.emplace_back(modal + (size_t) query_id * dim_[tmp]);
            }
            dist_op_.calculate(modal_a, modal_b, dim_, dim_, dist);
            sp[L++] = NeighborFlag(id, dist, true);
        }
        std::sort(sp.begin(), sp.begin() + L);

        // search K-NN
        unsigned k = 0;
        while (k < L) {
            unsigned nk = L;

            if (sp[k].flag_) {
                sp[k].flag_ = false;

                for (auto [id, _]: model_->graph_n_[sp[k].id_]) {
                    if (flags[id]) continue;
                    flags[id] = true;

                    DistResType dist = 0;
                    std::vector<float *> modal_a;
                    std::vector<float *> modal_b;
                    for (int tmp = 0; tmp < data_modal_.size(); ++tmp) {
                        const auto &modal = data_modal_[tmp];
                        modal_a.emplace_back(modal + (size_t) id * dim_[tmp]);
                        modal_b.emplace_back(modal + (size_t) query_id * dim_[tmp]);
                    }
                    dist_op_.calculate(modal_a, modal_b, dim_, dim_, dist);

                    full_set.emplace_back(id, dist);
                    if (dist >= sp[L - 1].distance_) continue;
                    NeighborFlag nn(id, dist, true);
                    int r = InsertIntoPool(sp.data(), L, nn);

                    if (L + 1 < sp.size()) ++L;
                    if (r < nk) nk = r;
                }
            }
            nk <= k ? (k = nk) : (++k);
        }
    }

    CStatus refreshParam() override {
        return CStatus();
    }

protected:
    std::vector<std::vector<Neighbor>> graph_pool_;
};

#endif //INDEXING_AND_SEARCH_C2_CANDIDATE_GREEDY_H

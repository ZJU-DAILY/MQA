//
// Created by MaouSanta on 2024/5/21.
//

#ifndef INDEXING_AND_SEARCH_C5_CONNECTIVITY_NSG_H
#define INDEXING_AND_SEARCH_C5_CONNECTIVITY_NSG_H

#include <stack>
#include "../c5_connectivity_basic.h"

class C5ConnectivityDFS : public C5ConnectivityBasic {
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

    CStatus train() override {
        tree_grow();
        eval_quality(model_->graph_n_, model_->sample_points_, model_->knn_set_);
        return CStatus();
    }


protected:
    void tree_grow() {
        unsigned root = model_->ep_;
        unsigned unlinked_cnt = 0;
        int tt = 0;
        std::vector<bool> flag(num_, false);
        while (unlinked_cnt < num_) {
            std::cerr << "begin DFS: " << (++tt) << std::endl;
            DFS(flag, root, unlinked_cnt);
            std::cerr << "root, cnt: " << root << ", " << unlinked_cnt << std::endl;
            if (unlinked_cnt >= num_) break;
            find_root(flag, root);
            std::cerr << "new root" << root << std::endl;
        }
    }

    // use stack to simulate recursion or may suffer lacking of memory
    void DFS(std::vector<bool>& flag, unsigned root, unsigned &cnt) {
        unsigned tmp = root;
        std::stack<unsigned> stk;
        stk.emplace(root);
        if (!flag[root]) cnt++;
        flag[root] = true;
        while (!stk.empty()) {
            unsigned next = num_ + 1;
            for (const auto& [id, dist]: model_->graph_n_[tmp]) {
                if (!flag[id]) {
                    next = id;
                    break;
                }
            }
            if (next == num_ + 1) {
                stk.pop();
                if (stk.empty()) break;
                tmp = stk.top();
                continue;
            }
            tmp = next;
            flag[tmp] = true;
            stk.push(tmp);
            cnt++;
        }
    }

    void find_root(std::vector<bool>& flag, unsigned &root) {
        std::cerr << "begin find root" << std::endl;
        unsigned root_candi = num_;
        // find an unlinked point
        // can be optimized by bitset and lowbit
        for (unsigned i = 0; i < num_; ++i) {
            if (!flag[i]) {
                root_candi = i;
                break;
            }
        }
        if (root_candi == num_) return;

        std::cerr << "begin search KNN" << std::endl;
        // search nearest neighbors of root_candi
        std::vector<NeighborFlag> sp;
        std::vector<Neighbor> pool;
        get_neighbors(root_candi, sp, pool);
        std::sort(pool.begin(), pool.end());

        // check if one of its neighbors has been found
        bool found = false;
        for (auto [id, dist]: pool) {
            if (flag[id]) {
                root = id;
                found = true;
                break;
            }
        }

        // if no one has been found, then randomly choose one
        // or just use ep. though it will make a huge point
        if (!found) {
//            while (true) {
//                std::random_device rd;
//                std::mt19937 rng(rd());
//                unsigned rid = rng() % num_;
//                if (flag[rid]) {
//                    root = rid;
//                    break;
//                }
//            }
            root = model_->ep_;
        }

        // connect the root_candi and the found one
        DistResType dist = 0;
        std::vector<float *> modal_a;
        std::vector<float *> modal_b;
        for (int tmp = 0; tmp < data_modal_.size(); ++tmp) {
            auto &modal = data_modal_[tmp];
            modal_a.emplace_back(modal + (size_t) root_candi * dim_[tmp]);
            modal_b.emplace_back(modal + (size_t) root * dim_[tmp]);
        }
        dist_op_.calculate(modal_a, modal_b, dim_, dim_, dist);

        model_->graph_n_[root].emplace_back(root_candi, dist);
    }

    // return the K-NN answer
    void get_neighbors(unsigned query_id, std::vector<NeighborFlag> &sp, std::vector<Neighbor> &full_set) {
        unsigned L = L_refine;
        sp.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        std::vector<bool> flags(num_, false);

        L = 0;
        for (unsigned i = 0; i < model_->graph_n_[query_id].size() && i < L_refine; ++i) {
            init_ids[L++] = model_->graph_n_[query_id][i].id_;
            flags[init_ids[i]] = true;
        }

        std::random_device rd;
        std::mt19937 rng(rd());
        while (L < L_refine) {
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
    unsigned L_refine = 100;
};

#endif //INDEXING_AND_SEARCH_C5_CONNECTIVITY_NSG_H

#pragma clang diagnostic pop
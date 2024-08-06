//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_C1_INITIALIZATION_NNDESCENT_H
#define INDEXING_AND_SEARCH_C1_INITIALIZATION_NNDESCENT_H

#if GA_USE_OPENMP

#include <omp.h>

#endif

#include <algorithm>
#include <mutex>
#include "../c1_initialization_basic.h"

static const unsigned NN_NEW = 0;     // new graph neighbors
static const unsigned NN_OLD = 1;     // old graph neighbors
static const unsigned RNN_NEW = 2;    // reverse new graph neighbors
static const unsigned RNN_OLD = 3;    // reverse old graph neighbors
static const unsigned MAX_NN_TYPE_SIZE = 4;

class C1InitializationNNDescent : public C1InitializationBasic {
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
        out_degree_ = t_param->k_init_graph;
        nn_size_ = t_param->nn_size;
        rnn_size_ = t_param->rnn_size;
        pool_size_ = t_param->pool_size;
        iter_ = t_param->iter;
        graph_quality_threshold_ = t_param->graph_quality_threshold;
        graph_pool_.resize(num_);
        graph_nn_[NN_NEW].resize(num_);
        graph_nn_[NN_OLD].resize(num_);
        graph_nn_[RNN_NEW].resize(num_);
        graph_nn_[RNN_OLD].resize(num_);
        for (int i = 0; i < num_; i++) {
            lock_mutex_.push_back(new std::mutex);
        }
        return DAnnFuncType::ANN_TRAIN;
    }

    CStatus train() override {
        CStatus status = init_neighbor();
        status += nn_descent();             // update points' neighbors by nn-descent
        if (status.isOK()) {
#if GA_USE_OPENMP
            printf("[OPENMP] nndescent openmp init complete!\n");
#else
            printf("[OPENMP] nndescent no openmp init complete!\n");
#endif
        }

        // one can also do pruning to cut the neighbor under R_neighbor
        model_->graph_n_.resize(num_);
        for (int i = 0; i < num_; ++i) {
            // reset the graph_n_
            std::vector<Neighbor>().swap(model_->graph_n_[i]);
            for (auto& item: graph_pool_[i]) {
                model_->graph_n_[i].emplace_back(item);
            }
            // graph_pool_ uses greater<>, hence we need to sort
            std::sort(model_->graph_n_[i].begin(), model_->graph_n_[i].end());
        }
        return status;
    }

    CStatus refreshParam() override {
        std::vector<std::set<NeighborFlag, std::greater<>>>().swap(graph_pool_);
        std::vector<std::vector<IDType >>().swap(graph_nn_[NN_NEW]);
        std::vector<std::vector<IDType >>().swap(graph_nn_[NN_OLD]);
        std::vector<std::vector<IDType >>().swap(graph_nn_[RNN_NEW]);
        std::vector<std::vector<IDType >>().swap(graph_nn_[RNN_OLD]);
        std::vector<std::mutex *>().swap(lock_mutex_);
        return CStatus();
    }

protected:
    /**
     * initialize a random graph
     * @return
     */
    CStatus init_neighbor() {
#pragma omp parallel for num_threads(Params.thread_num_) schedule(dynamic) default(none)
        for (size_t i = 0; i < num_; i++) {
            graph_nn_[NN_NEW][i].resize(nn_size_ * 2);
            GenRandomID(graph_nn_[NN_NEW][i].data(), num_, graph_nn_[NN_NEW][i].size());
            std::vector<IDType> cur(nn_size_ + 1);
            GenRandomID(cur.data(), num_, cur.size());
            for (unsigned j = 0; j < nn_size_; j++) {
                DistResType dist = 0;
                IDType id = cur[j];
                if (id == i) continue;

                std::vector<float *> modal_a;
                std::vector<float *> modal_b;
                for (int tmp = 0; tmp < data_modal_.size(); ++tmp) {
                    const auto& modal = data_modal_[tmp];
                    modal_a.emplace_back(modal + (size_t) i * dim_[tmp]);
                    modal_b.emplace_back(modal + (size_t) id * dim_[tmp]);
                }
                dist_op_.calculate(modal_a, modal_b, dim_, dim_, dist);
                graph_pool_[i].emplace(id, dist, true);
            }
        }
        return CStatus();
    }

    /**
     * insert neigh_id into pro_id's neighbor pool
     * @param pro_id
     * @param neigh_id
     * @param dist
     * @return
     */
    CStatus insert(IDType pro_id, IDType neigh_id, DistResType dist) {
        std::lock_guard<std::mutex> LockGuard(*lock_mutex_[pro_id]);
        graph_pool_[pro_id].emplace(neigh_id, dist, true);
        if (graph_pool_[pro_id].size() > pool_size_) {
            graph_pool_[pro_id].erase(graph_pool_[pro_id].begin());
        }
        return CStatus();
    }

    /**
     * Bi-directional insert. insert b and a into a's and b's neighbor pool, respectively
     * @param a
     * @param b
     * @return
     */
    CStatus bi_insert(IDType a, IDType b) {
        DistResType dist = 0;
        std::vector<float *> modal_a;
        std::vector<float *> modal_b;
        for (int tmp = 0; tmp < data_modal_.size(); ++tmp) {
            const auto& modal = data_modal_[tmp];
            modal_a.emplace_back(modal + (size_t) a * dim_[tmp]);
            modal_b.emplace_back(modal + (size_t) b * dim_[tmp]);
        }
        dist_op_.calculate(modal_a, modal_b, dim_, dim_, dist);

        insert(a, b, dist);
        insert(b, a, dist);
        return CStatus();
    }

    /**
     * obtain a new id j from nn_new or nn_old,
     * satisfying cur_id and j have not been inserted each other
     * @param pro_id
     * @param cur_id
     * @param nn_type: NN_NEW and NN_OLD
     * @return
     */
    CStatus mutual_insert(IDType pro_id, IDType cur_id, unsigned nn_type) {
        for (IDType const j: graph_nn_[nn_type][pro_id]) {
            if ((NN_NEW == nn_type && cur_id < j)
                || (NN_OLD == nn_type && cur_id != j)) {
                bi_insert(cur_id, j);
            }
        }
        return CStatus();
    }

    CStatus join_neighbor() {
#pragma omp parallel for num_threads(Params.thread_num_) schedule(dynamic) default(none)
        for (size_t n = 0; n < num_; n++) {
            for (unsigned const i: graph_nn_[NN_NEW][n]) {
                mutual_insert(n, i, NN_NEW);
                mutual_insert(n, i, NN_OLD);
            }
        }
        return CStatus();
    }

    /**
     *
     * @param pro_id
     * @param neigh_id
     * @param rnn_type : RNN_NEW and RNN_OLD
     * @return
     */
    CStatus generate_reverse_neighbor(IDType pro_id, IDType neigh_id, unsigned rnn_type) {
        if (graph_nn_[rnn_type][neigh_id].size() < rnn_size_) {
            graph_nn_[rnn_type][neigh_id].emplace_back(pro_id);
        } else {
            std::random_device rd;
            std::mt19937 rng(rd());
            unsigned int pos = rng() % rnn_size_;
            graph_nn_[rnn_type][neigh_id][pos] = pro_id;
        }

        return CStatus();
    }

    CStatus shuffle_reverse_neighbor(std::vector<IDType> rnn) const {
        if (rnn_size_ && rnn.size() > rnn_size_) {
            auto seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(rnn.begin(), rnn.end(), std::default_random_engine(seed));
            rnn.resize(rnn_size_);
        }
        return CStatus();
    }

    /**
     * generate pro_id's nn_new, nn_old, rnn_new, and rnn_old,
     * according to pro_id's neighbor neigh_id in pool
     * @param pro_id
     * @param neigh_id
     * @return
     */
    CStatus generate_neighbor(IDType pro_id, NeighborFlag& neigh) {
        unsigned nn_type;
        unsigned rnn_type;
        if (neigh.flag_) {
            nn_type = NN_NEW;
            rnn_type = RNN_NEW;
            neigh.flag_ = false;
        } else {
            nn_type = NN_OLD;
            rnn_type = RNN_OLD;
        }
        graph_nn_[nn_type][pro_id].emplace_back(neigh.id_);

        if (neigh.distance_ > graph_pool_[neigh.id_].begin()->distance_) {
            std::lock_guard<std::mutex> LockGuard(*lock_mutex_[neigh.id_]);
            generate_reverse_neighbor(pro_id, neigh.id_, rnn_type);
        }

        return CStatus();
    }

    /**
     * insert pro_id's reverse neighbor (rnn_type) into nn_type neighbor
     * @param pro_id
     * @param nn_type: NN_NEW and NN_OLD
     * @param rnn_type: RNN_NEW and RNN_OLD
     * @return
     */
    CStatus insert_reverse_neighbor(IDType pro_id, unsigned nn_type, unsigned rnn_type) {
        auto &nn = graph_nn_[nn_type][pro_id];
        auto &rnn = graph_nn_[rnn_type][pro_id];
        CStatus status = shuffle_reverse_neighbor(rnn);
        nn.insert(nn.end(), rnn.begin(), rnn.end());
        std::vector<IDType>().swap(rnn);
        return status;
    }

    CStatus update_neighbor() {
#pragma omp parallel for num_threads(Params.thread_num_) schedule(dynamic) default(none)
        for (IDType i = 0; i < num_; i++) {
            std::vector<IDType>().swap(graph_nn_[NN_NEW][i]);
            std::vector<IDType>().swap(graph_nn_[NN_OLD][i]);
        }

#pragma omp parallel for num_threads(Params.thread_num_) schedule(dynamic) default(none)
        for (IDType i = 0; i < num_; i++) {
            // the neigh.flag may change, but one can not simply modify
            // "auto" to "auto&" to fixed due to the type is std::set
            std::set<NeighborFlag, std::greater<>> graph_pool_tmp_;
            for (auto neigh: graph_pool_[i]) {
                generate_neighbor(i, neigh);
                graph_pool_tmp_.emplace(neigh);
            }
            graph_pool_[i] = graph_pool_tmp_;

            std::lock_guard<std::mutex> LockGuard(*lock_mutex_[i]);
            insert_reverse_neighbor(i, NN_NEW, RNN_NEW);
            insert_reverse_neighbor(i, NN_OLD, RNN_OLD);
        }

        return CStatus();
    }

    /**
     * iteratively optimize vertices' neighbors
     * @return
     */
    CStatus nn_descent() {
        CStatus status;
        printf("[NNDESCENT] begin nn-descent\n");
        for (unsigned it = 0; it < iter_; it++) {
            clock_t t1, t2, t3;
            t1 = clock();
            status += join_neighbor();    // neighbors join each other
            t2 = clock();
            status += update_neighbor();    // update candidate neighbors for neighbors join
            t3 = clock();
            printf("[NNDESCENT] iter: %d, join_neighbor: %lf(s)\n", it, (double) (t2 - t1) / CLOCKS_PER_SEC);
            printf("[NNDESCENT] iter: %d, update_neighbor: %lf(s)\n", it, (double) (t3 - t2) / CLOCKS_PER_SEC);
            float rc = eval_quality(graph_pool_, model_->sample_points_, model_->knn_set_);    // evaluate graph quality for this iteration
            printf("[NNDESCENT] iter: %d, graph quality: %f\n", it, rc);
            if (rc >= graph_quality_threshold_)
                break;
        }

        return status;
    }

protected:
    unsigned iter_ = 3;
    float graph_quality_threshold_ = 0.8;
    unsigned nn_size_ = 10;
    unsigned rnn_size_ = 5;
    unsigned pool_size_ = 20;
    std::vector<std::mutex *> lock_mutex_;

    std::vector<std::set<NeighborFlag, std::greater<>>> graph_pool_;
    std::vector<std::vector<IDType>> graph_nn_[MAX_NN_TYPE_SIZE]; // new, old, reverse new, and reverse old graph neighbors
};

#endif //INDEXING_AND_SEARCH_C1_INITIALIZATION_NNDESCENT_H

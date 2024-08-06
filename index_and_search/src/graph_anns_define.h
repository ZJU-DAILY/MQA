//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_GRAPH_ANNS_DEFINE_H
#define INDEXING_AND_SEARCH_GRAPH_ANNS_DEFINE_H


#include <utility>

#include "../CGraph/src/CGraph.h"

struct ParamConfig {
    explicit ParamConfig() = default;

    void set_train_param(std::vector<char *> modal_base_path, char *index_path,
                         unsigned r = 30, unsigned c = 200, unsigned l = 100, unsigned k_init_graph = 100,
                         unsigned nn_size = 50, unsigned rnn_size = 25, unsigned pool_size = 200,
                         unsigned iter = 8, unsigned sample_num = 100, float graph_quality_threshold = 0.8) {
        GA_ALG_BASE_MODAL_PATH_ = std::move(modal_base_path);
        GA_ALG_INDEX_PATH_ = index_path;
        L_candidate_ = l;
        R_neighbor_ = r;
        C_neighbor_ = c;
        k_init_graph_ = k_init_graph;
        nn_size_ = nn_size;
        rnn_size_ = rnn_size;
        pool_size_ = pool_size;
        iter_ = iter;
        sample_num_ = sample_num;
        graph_quality_threshold_ = graph_quality_threshold;
    }

    void set_search_param(std::vector<char *> modal_base_path,
                          std::vector<char *> modal_query_path,
                          char *res_path, unsigned top_k = 1,
                          char *index_path = nullptr,
                          char *gt_path = {}, unsigned gt_k = 1, unsigned L_search = 200) {
        GA_ALG_BASE_MODAL_PATH_ = std::move(modal_base_path);
        GA_ALG_QUERY_MODAL_PATH_ = std::move(modal_query_path);
        GA_ALG_GROUND_TRUTH_PATH_ = gt_path;
        GA_ALG_INDEX_PATH_ = index_path;
        GA_ALG_RESULT_PATH_ = res_path;
        top_k_ = top_k;
        gt_k_ = gt_k;
        L_search_ = L_search;
    }

    void set_data_param(const std::vector<float> &w) {
        w_ = w;
    }

    void set_general_param(unsigned thread_num = 1,
                           unsigned is_delete_id = 0,
                           unsigned is_multi_res_equal = 0) {
        thread_num_ = thread_num;
        is_multi_res_equal_ = is_multi_res_equal;
        is_delete_id_ = is_delete_id;
    }

    void set_delete_id_path(char *delete_id_path) {
        GA_ALG_DELETE_ID_PATH_ = delete_id_path;
    }

public:
    std::vector<char *> GA_ALG_BASE_MODAL_PATH_{};    // base vector data for modal #1
    std::vector<char *> GA_ALG_QUERY_MODAL_PATH_{};
    char *GA_ALG_GROUND_TRUTH_PATH_{};
    char *GA_ALG_INDEX_PATH_{};
    std::vector<char *> GA_ALG_MODAL_INDEX_PATH_{};
    char *GA_ALG_RESULT_PATH_{};
    char *GA_ALG_DELETE_ID_PATH_{};

    unsigned L_candidate_{};
    unsigned R_neighbor_{};
    unsigned C_neighbor_{};
    unsigned k_init_graph_{};
    unsigned nn_size_{};
    unsigned rnn_size_{};
    unsigned pool_size_{};
    unsigned iter_{};
    unsigned sample_num_{};
    float graph_quality_threshold_{};

    unsigned top_k_{};
    unsigned gt_k_{};
    unsigned L_search_{};

    unsigned thread_num_{};
    unsigned is_multi_res_equal_{};
    unsigned is_delete_id_{};
    unsigned candi_top_k_{};

    std::vector<float> w_{};
};

ParamConfig Params{};

#endif //INDEXING_AND_SEARCH_GRAPH_ANNS_DEFINE_H

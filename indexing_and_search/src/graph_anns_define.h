//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_GRAPH_ANNS_DEFINE_H
#define INDEXING_AND_SEARCH_GRAPH_ANNS_DEFINE_H


#include <utility>

#include "../CGraph/src/CGraph.h"

struct ParamConfig {
    explicit ParamConfig() = default;

    void set_train_param(std::vector<char *> modal1_base_path, std::vector<char *> modal2_base_path, char *index_path,
                         unsigned r = 30, unsigned c = 200, unsigned l = 100, unsigned k_init_graph = 100,
                         unsigned nn_size = 50, unsigned rnn_size = 25, unsigned pool_size = 200,
                         unsigned iter = 8, unsigned sample_num = 100, float graph_quality_threshold = 0.8) {
        GA_ALG_BASE_MODAL1_PATH_ = std::move(modal1_base_path);
        GA_ALG_BASE_MODAL2_PATH_ = std::move(modal2_base_path);
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

    void set_search_param(std::vector<char *> modal1_base_path, std::vector<char *> modal2_base_path,
                          std::vector<char *> modal1_query_path, std::vector<char *> modal2_query_path,
                          char *res_path, unsigned top_k = 1,
                          char *index_path = nullptr,
                          char *gt_path = {}, unsigned gt_k = 1, unsigned L_search = 200) {
        GA_ALG_BASE_MODAL1_PATH_ = std::move(modal1_base_path);
        GA_ALG_BASE_MODAL2_PATH_ = std::move(modal2_base_path);
        GA_ALG_QUERY_MODAL1_PATH_ = std::move(modal1_query_path);
        GA_ALG_QUERY_MODAL2_PATH_ = std::move(modal2_query_path);
        GA_ALG_GROUND_TRUTH_PATH_ = gt_path;
        GA_ALG_INDEX_PATH_ = index_path;
        GA_ALG_RESULT_PATH_ = res_path;
        top_k_ = top_k;
        gt_k_ = gt_k;
        L_search_ = L_search;
    }

    void set_data_param(const std::vector<float> &w1, const std::vector<float> &w2) {
        w1_ = w1;
        w2_ = w2;
    }

    void set_general_param(unsigned thread_num = 1,
                           std::vector<unsigned> is_norm_modal1 = std::vector<unsigned>{},
                           std::vector<unsigned> is_norm_modal2 = std::vector<unsigned>{},
                           unsigned is_delete_id = 0, unsigned is_skip = 0, unsigned skip_num = 0,
                           unsigned is_multi_res_equal = 0) {
        thread_num_ = thread_num;
        is_norm_modal1_ = std::move(is_norm_modal1);
        is_norm_modal2_ = std::move(is_norm_modal2);
        is_skip_ = is_skip;
        skip_num_ = skip_num;
        is_multi_res_equal_ = is_multi_res_equal;
        is_delete_id_ = is_delete_id;
    }

    void set_modal_index_path(std::vector<char *> modal1_index_path, std::vector<char *> modal2_index_path) {
        GA_ALG_MODAL1_INDEX_PATH_ = std::move(modal1_index_path);
        GA_ALG_MODAL2_INDEX_PATH_ = std::move(modal2_index_path);
    }

    void set_delete_id_path(char *delete_id_path) {
        GA_ALG_DELETE_ID_PATH_ = delete_id_path;
    }

    void set_candidate_top_k(unsigned candi_top_k) {
        candi_top_k_ = candi_top_k;
    }

public:
    std::vector<char *> GA_ALG_BASE_MODAL1_PATH_{};    // base vector data for modal #1
    std::vector<char *> GA_ALG_BASE_MODAL2_PATH_{};    // base vector data for modal #2
    std::vector<char *> GA_ALG_QUERY_MODAL1_PATH_{};
    std::vector<char *> GA_ALG_QUERY_MODAL2_PATH_{};
    char *GA_ALG_GROUND_TRUTH_PATH_{};
    char *GA_ALG_INDEX_PATH_{};
    std::vector<char *> GA_ALG_MODAL1_INDEX_PATH_{};
    std::vector<char *> GA_ALG_MODAL2_INDEX_PATH_{};
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
    std::vector<unsigned> is_norm_modal1_{};
    std::vector<unsigned> is_norm_modal2_{};
    unsigned is_skip_{};
    unsigned skip_num_{};
    unsigned is_multi_res_equal_{};
    unsigned is_delete_id_{};
    unsigned candi_top_k_{};

    std::vector<float> w1_{};
    std::vector<float> w2_{};
};

ParamConfig Params{};

#endif //INDEXING_AND_SEARCH_GRAPH_ANNS_DEFINE_H

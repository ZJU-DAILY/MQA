//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_EVA_MODAL1_BRUTE_FORCE_NODE_H
#define INDEXING_AND_SEARCH_EVA_MODAL1_BRUTE_FORCE_NODE_H

#if GA_USE_OPENMP

#include <omp.h>

#endif

#include <cassert>
#include "../../elements_define.h"

class EvaModal1BruteForceNode : public CGraph::GNode {
public:
    CStatus init() override {
        if (Params.GA_ALG_BASE_MODAL1_PATH_.empty()) {
            CGRAPH_EMPTY_FUNCTION
        }

        auto *model_param = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
        if (!model_param) {
            CGRAPH_RETURN_ERROR_STATUS("EvaModal1BruteForceNode get model param failed")
        }

        CStatus status;
        model_param->train_meta_modal1_.resize(Params.GA_ALG_BASE_MODAL1_PATH_.size());
        for (size_t i = 0; i < Params.GA_ALG_BASE_MODAL1_PATH_.size(); ++i) {
            status += model_param->train_meta_modal1_[i].load(Params.GA_ALG_BASE_MODAL1_PATH_[i], Params.is_norm_modal1_[i]);
            dim1_.emplace_back(model_param->train_meta_modal1_[i].dim);
            printf("[PATH] modal 1 [%zu] vector path: %s\n", i, Params.GA_ALG_BASE_MODAL1_PATH_[i]);
            printf("[PARAM] modal 1 [%zu] vector num: %d\n", i, model_param->train_meta_modal1_[i].num);
            printf("[PARAM] modal 1 [%zu] vector dim: %d\n", i, model_param->train_meta_modal1_[i].dim);
        }
        auto *s_param = CGRAPH_GET_GPARAM(AlgParamBasic, GA_ALG_PARAM_BASIC_KEY)
        if (!s_param) {
            CGRAPH_RETURN_ERROR_STATUS("EvaModal1BruteForceNode get search param failed")
        }

        model_param->search_meta_modal1_.resize(Params.GA_ALG_QUERY_MODAL1_PATH_.size());
        for (int i = 0; i < Params.GA_ALG_QUERY_MODAL1_PATH_.size(); ++i) {
            status += model_param->search_meta_modal1_[i].load(Params.GA_ALG_QUERY_MODAL1_PATH_[i], Params.is_norm_modal1_[i]);
            assert(model_param->search_meta_modal1_[i].dim == model_param->train_meta_modal1_[i].dim);
            printf("[PATH] modal 1 [%d] vector path: %s\n", i, model_param->search_meta_modal1_[i].file_path.c_str());
            printf("[PARAM] modal 1 [%d] vector num: %d\n", i, model_param->search_meta_modal1_[i].num);
            printf("[PARAM] modal 1 [%d] vector dim: %d\n", i, model_param->search_meta_modal1_[i].dim);
        }

        if (Params.is_delete_id_) {
            status += model_param->delete_meta_.load(Params.GA_ALG_DELETE_ID_PATH_, 0);
            delete_num_each_query_ = model_param->delete_meta_.dim;
        }
        if (!status.isOK()) {
            return CStatus("EvaModal1BruteForceNode init load param failed");
        }

        num_ = model_param->train_meta_modal1_[0].num;
        query_num_ = model_param->search_meta_modal1_[0].num;
        return CStatus();
    }

    CStatus run() override {
        if (Params.GA_ALG_BASE_MODAL1_PATH_.empty()) {
            CGRAPH_EMPTY_FUNCTION
        }

        auto *m_param = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY)
        auto *s_param = CGRAPH_GET_GPARAM(AlgParamBasic, GA_ALG_PARAM_BASIC_KEY);
        if (nullptr == m_param || nullptr == s_param) {
            CGRAPH_RETURN_ERROR_STATUS("EvaModal1BruteForceNode run get param failed")
        }
        s_param->results_modal1.resize(m_param->search_meta_modal1_.size());
        s_param->results_dist1.resize(m_param->search_meta_modal1_.size());
        for (auto& item: s_param->results_modal1) {
            item.resize(query_num_);
        }
        for (auto& item: s_param->results_dist1) {
            item.resize(query_num_);
        }
        unsigned top_k = Params.candi_top_k_;

        for (int modal = 0; modal < m_param->search_meta_modal1_.size(); ++modal) {
#pragma omp parallel for num_threads(Params.thread_num_) schedule(dynamic) \
                         shared(m_param, s_param, top_k) default(none)
            for (IDType i = 0; i < query_num_; i++) {
                std::priority_queue<std::pair<DistResType, IDType> > dist_id;
                for (IDType j = 0; j < num_; j++) {
                    bool is_delete = false;
                    if (delete_num_each_query_) {
                        for (IDType k = 0; k < delete_num_each_query_; k++) {
                            if (j == m_param->delete_meta_.data[i * delete_num_each_query_ + k]) {
                                is_delete = true;
                                break;
                            }
                        }
                    }
                    if (is_delete) continue;
                    DistResType dist = 0;
                    dist_op_.calc_vec1(m_param->search_meta_modal1_[modal].data + i * dim1_[modal],
                                       m_param->train_meta_modal1_[modal].data + j * dim1_[modal],
                                       dim1_[modal], dim1_[modal], dist);
                    dist_id.emplace(dist, j);
                }
                for (IDType j = 0; j < top_k; j++) {
                    std::pair<DistResType, IDType> p = dist_id.top();
                    dist_id.pop();
                    s_param->results_modal1[modal][i].emplace_back(p.second);
                    s_param->results_dist1[modal][i].emplace_back(p.first);
                }
            }
        }

#if GA_USE_OPENMP
        printf("[OPENMP] brute force openmp init complete!\n");
#else
        CGraph::CGRAPH_ECHO("brute force no openmp init complete!");
#endif

        return CStatus();
    }

private:
    std::vector<unsigned> dim1_;
    unsigned delete_num_each_query_ = 0;
    unsigned query_num_ = 0;
    unsigned num_ = 0;
    BiDistanceCalculator<> dist_op_;
};

#endif //INDEXING_AND_SEARCH_EVA_MODAL1_BRUTE_FORCE_NODE_H

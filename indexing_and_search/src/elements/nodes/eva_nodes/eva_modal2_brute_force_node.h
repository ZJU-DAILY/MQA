//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_EVA_MODAL2_BRUTE_FORCE_NODE_H
#define INDEXING_AND_SEARCH_EVA_MODAL2_BRUTE_FORCE_NODE_H

#if GA_USE_OPENMP

#include <omp.h>

#endif

#include <cassert>
#include "../../elements_define.h"

class EvaModal2BruteForceNode : public CGraph::GNode {
public:
    CStatus init() override {
        if (Params.GA_ALG_BASE_MODAL2_PATH_.empty()) {
            CGRAPH_EMPTY_FUNCTION
        }

        auto *model_param = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
        if (!model_param) {
            CGRAPH_RETURN_ERROR_STATUS("EvaModal2BruteForceNode get model param failed")
        }

        CStatus status;
        model_param->train_meta_modal2_.resize(Params.GA_ALG_BASE_MODAL2_PATH_.size());
        for (int i = 0; i < Params.GA_ALG_BASE_MODAL2_PATH_.size(); ++i) {
            status += model_param->train_meta_modal2_[i].load(Params.GA_ALG_BASE_MODAL2_PATH_[i], Params.is_norm_modal2_[i]);

            dim2_.emplace_back(model_param->train_meta_modal2_[i].dim);
            printf("[PATH] modal 2 [%d] vector path: %s\n", i, Params.GA_ALG_BASE_MODAL2_PATH_[i]);
            printf("[PARAM] modal 2 [%d] vector num: %d\n", i, model_param->train_meta_modal2_[i].num);
            printf("[PARAM] modal 2 [%d] vector dim: %d\n", i, model_param->train_meta_modal2_[i].dim);
        }

        auto *s_param = CGRAPH_GET_GPARAM(AlgParamBasic, GA_ALG_PARAM_BASIC_KEY)
        if (!s_param) {
            CGRAPH_RETURN_ERROR_STATUS("EvaModal2BruteForceNode get search param failed")
        }

        model_param->search_meta_modal2_.resize(Params.GA_ALG_QUERY_MODAL2_PATH_.size());
        for (int i = 0; i < Params.GA_ALG_QUERY_MODAL2_PATH_.size(); ++i) {
            status += model_param->search_meta_modal2_[i].load(Params.GA_ALG_QUERY_MODAL2_PATH_[i], Params.is_norm_modal2_[i]);

            assert(model_param->search_meta_modal2_[i].dim == model_param->train_meta_modal2_[i].dim);
            printf("[PATH] modal 2 [%d] vector path: %s\n", i, model_param->search_meta_modal2_[i].file_path.c_str());
            printf("[PARAM] modal 2 [%d] vector num: %d\n", i, model_param->search_meta_modal2_[i].num);
            printf("[PARAM] modal 2 [%d] vector dim: %d\n", i, model_param->search_meta_modal2_[i].dim);
        }

        if (Params.is_delete_id_) {
            status += model_param->delete_meta_.load(Params.GA_ALG_DELETE_ID_PATH_, 0);
            delete_num_each_query_ = model_param->delete_meta_.dim;
        }
        if (!status.isOK()) {
            return CStatus("EvaModal2BruteForceNode init load param failed");
        }

        num_ = model_param->train_meta_modal2_[0].num;
        query_num_ = model_param->search_meta_modal2_[0].num;
        return CStatus();
    }

    CStatus run() override {
        if (Params.GA_ALG_BASE_MODAL2_PATH_.empty()) {
            CGRAPH_EMPTY_FUNCTION
        }

        auto *m_param = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY)
        auto *s_param = CGRAPH_GET_GPARAM(AlgParamBasic, GA_ALG_PARAM_BASIC_KEY);
        if (nullptr == m_param || nullptr == s_param) {
            CGRAPH_RETURN_ERROR_STATUS("EvaModal2BruteForceNode run get param failed")
        }

        s_param->results_modal2.resize(query_num_);
        unsigned top_k = Params.candi_top_k_;

        for (int modal = 0; modal < m_param->search_meta_modal2_.size(); ++modal) {
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
                    dist_op_.calc_vec2(m_param->search_meta_modal2_[modal].data + (i * dim2_[modal]),
                                       m_param->train_meta_modal2_[modal].data + j * dim2_[modal],
                                       dim2_[modal], dim2_[modal], dist);
                    dist_id.emplace(dist, j);
                }
                for (IDType j = 0; j < top_k; j++) {
                    std::pair<DistResType, IDType> p = dist_id.top();
                    dist_id.pop();
                    s_param->results_modal2[modal][i].emplace_back(p.second);
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
    std::vector<unsigned> dim2_;
    unsigned delete_num_each_query_ = 0;
    unsigned query_num_ = 0;
    unsigned num_ = 0;
    BiDistanceCalculator<> dist_op_;
};

#endif //INDEXING_AND_SEARCH_EVA_MODAL2_BRUTE_FORCE_NODE_H

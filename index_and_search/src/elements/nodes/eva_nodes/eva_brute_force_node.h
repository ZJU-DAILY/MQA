//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_EVA_BRUTE_FORCE_NODE_H
#define INDEXING_AND_SEARCH_EVA_BRUTE_FORCE_NODE_H

#if GA_USE_OPENMP

#include <omp.h>

#endif

#include <cassert>
#include "../../elements_define.h"

class EvaBruteForceNode : public CGraph::GNode {
public:
    CStatus init() override {
        auto *model_param = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY)
        if (!model_param) {
            CGRAPH_RETURN_ERROR_STATUS("EvaBruteForceNode get model param failed")
        }

        // input base modal
        CStatus status;
        model_param->train_meta_modal_.resize(Params.GA_ALG_BASE_MODAL_PATH_.size());
        for (int i = 0; i < Params.GA_ALG_BASE_MODAL_PATH_.size(); ++i) {
            status += model_param->train_meta_modal_[i].load(Params.GA_ALG_BASE_MODAL_PATH_[i]);
            num_ = std::max(num_, model_param->train_meta_modal_[i].num);
            dim_.emplace_back(model_param->train_meta_modal_[i].dim);
        }
        // check base modal
        printf("[PARAM] vector num: %u\n", num_);
        for (int i = 0; i < Params.GA_ALG_BASE_MODAL_PATH_.size(); ++i) {
            assert(model_param->train_meta_modal_[i].num == num_);
            printf("[PATH] modal [%d] vector path: %s\n", i, Params.GA_ALG_BASE_MODAL_PATH_[i]);
            printf("[PARAM] modal [%d] vector dim: %u\n", i, model_param->train_meta_modal_[i].dim);
        }
        // set result params
        auto *s_param = CGRAPH_GET_GPARAM(AlgParamBasic, GA_ALG_PARAM_BASIC_KEY)
        if (!s_param) {
            CGRAPH_RETURN_ERROR_STATUS("EvaBruteForceNode get search param failed")
        }
        // input query modal
        model_param->search_meta_modal_.resize(Params.GA_ALG_QUERY_MODAL_PATH_.size());
        for (int i = 0; i < Params.GA_ALG_QUERY_MODAL_PATH_.size(); ++i) {
            status += model_param->search_meta_modal_[i].load(Params.GA_ALG_QUERY_MODAL_PATH_[i]);
            query_num_ = std::max(query_num_, model_param->search_meta_modal_[i].num);
            printf("[PATH] modal [%d] query vector path: %s\n", i,
                   model_param->search_meta_modal_[i].file_path.c_str());
        }
        // input delete id
        if (Params.is_delete_id_) {
            status += model_param->delete_meta_.load(Params.GA_ALG_DELETE_ID_PATH_, 0);
            delete_num_each_query_ = model_param->delete_meta_.dim;
        }
        // check query modal
        for (int i = 0; i < Params.GA_ALG_QUERY_MODAL_PATH_.size(); ++i) {
            assert(query_num_ == model_param->search_meta_modal_[i].num);
            assert(model_param->train_meta_modal_[i].dim == model_param->search_meta_modal_[i].dim);
            printf("[PARAM] modal [%d] query vector dim: %u\n", i, model_param->search_meta_modal_[i].dim);
        }
        if (!status.isOK()) {
            return CStatus("EvaBruteForceNode init load param failed");
        }
        printf("[PARAM] query vector num: %u\n", query_num_);

        return CStatus();
    }

    CStatus run() override {
        auto *m_param = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY)
        auto *s_param = CGRAPH_GET_GPARAM(AlgParamBasic, GA_ALG_PARAM_BASIC_KEY)
        if (nullptr == m_param || nullptr == s_param) {
            CGRAPH_RETURN_ERROR_STATUS("EvaBruteForceNode run get param failed")
        }

        s_param->results_.resize(query_num_);
        unsigned top_k = s_param->top_k;
#pragma omp parallel for num_threads(Params.thread_num_) schedule(dynamic) \
                         shared(m_param, s_param, top_k) default(none)
        for (IDType i = 0; i < query_num_; i++) {
            std::set<std::pair<DistResType, IDType>> dist_id;
            for (IDType j = 0; j < num_; j++) {
                bool is_delete = false;
                if (delete_num_each_query_) {
                    for (IDType k = 0; k < delete_num_each_query_; k++) {
                        if (j == m_param->delete_meta_.data[k]) {
                            is_delete = true;
                            break;
                        }
                    }
                }
                if (is_delete) continue;
                DistResType dist = 0;
                std::vector<float *> modal_id;
                std::vector<float *> modal_cur;
                for (int k = 0; k < m_param->search_meta_modal_.size(); ++k) {
                    modal_id.emplace_back(m_param->search_meta_modal_[k].data + (size_t) i * dim_[k]);
                    modal_cur.emplace_back(m_param->train_meta_modal_[k].data + (size_t) j * dim_[k]);
                }
                dist_op_.calculate(modal_id, modal_cur, dim_, dim_, dist);

                dist_id.emplace(dist, j);
            }

            IDType j = 0;
            for (auto item: dist_id) {
                s_param->results_[i].emplace_back(item);
                if (++j == top_k) break;
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
    std::vector<unsigned> dim_;
    unsigned delete_num_each_query_ = 0;
    unsigned query_num_ = 0;
    unsigned num_ = 0;
    BiDistanceCalculator<> dist_op_;
};

#endif //INDEXING_AND_SEARCH_EVA_BRUTE_FORCE_NODE_H

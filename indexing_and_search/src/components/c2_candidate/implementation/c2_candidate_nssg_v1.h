//
// Created by MaouSanta on 2024/2/18.
//

#ifndef INDEXING_AND_SEARCH_C2_CANDIDATE_NSSG_V1_H
#define INDEXING_AND_SEARCH_C2_CANDIDATE_NSSG_V1_H

#include "../c2_candidate_basic.h"

#if GA_USE_OPENMP

#include <omp.h>

#endif

class C2CandidateNSSGV1 : public C2CandidateNSSG {
public:
    CStatus train() override {
        model_->pool_m_.resize(num_);

#pragma omp parallel for num_threads(Params.thread_num_) schedule(dynamic) default(none)
        for (IDType i = 0; i < num_; i++) {
            std::vector<bool> flags(num_, false);
            flags[i] = true;
            for (unsigned j = 0; j < model_->graph_n_[i].size(); j++) {
                if (flags[j]) continue;
                flags[j] = true;
                IDType nid = model_->graph_n_[i][j].id_;
                float ndist = model_->graph_n_[i][j].distance_;
                model_->pool_m_[i].emplace_back(nid, ndist);
            }
//            std::vector<Neighbor>().swap(model_->graph_n_[i]);

            for (unsigned j = 0; j < model_->graph_n_[i].size(); j++) {
                IDType nid = model_->graph_n_[i][j].id_;
                for (unsigned nn = 0; nn < model_->graph_n_[nid].size(); nn++) {
                    IDType nnid = model_->graph_n_[nid][nn].id_;
                    if (flags[nnid]) continue;
                    flags[nnid] = true;
                    DistResType dist = 0;
                    dist_op_.calc(data_modal1_, data_modal2_, dim1_, dim2_, i, nnid, dist);
                    model_->pool_m_[i].emplace_back(nnid, dist);
                    if (model_->pool_m_[i].size() >= L_) break;
                }
                if (model_->pool_m_[i].size() >= L_) break;
            }
        }

        for (IDType i = 0; i < num_; i++) {
//            if (model_->graph_n_[i].empty()) continue;
//            std::cerr << model_->graph_n_[i].size() << " ";
            std::vector<Neighbor>().swap(model_->graph_n_[i]);
        }
        std::vector<std::vector<Neighbor>>().swap(model_->graph_n_);
        std::cout << "C2 end" << std::endl;
        return CStatus();
    }
};

#endif //INDEXING_AND_SEARCH_C2_CANDIDATE_NSSG_V1_H

//
// Created by MaouSanta on 2024/2/18.
//

#ifndef INDEXING_AND_SEARCH_C3_NEIGHBOR_NSG_V1_H
#define INDEXING_AND_SEARCH_C3_NEIGHBOR_NSG_V1_H

#include "../c3_neighbor_basic.h"

#if GA_USE_OPENMP

#include <omp.h>

#endif

class C3NeighborNSGV1 : public C3NeighborNSG {
public:
    CStatus train() override {
        auto t_param = CGRAPH_GET_GPARAM(NPGTrainParam, GA_ALG_NPG_TRAIN_PARAM_KEY)
        CGRAPH_ASSERT_NOT_NULL(t_param)
        constexpr double kPi = 3.14159265358979323846264;
//        const double pi = acos(-1.0);
        float angle = 60;
        float threshold = std::cos(angle / 180 * kPi);

#pragma omp parallel for num_threads(Params.thread_num_) schedule(dynamic) shared(threshold) default(none)
        for (unsigned i = 0; i < num_; i++) {
            unsigned start = 0;
            std::sort(model_->pool_m_[i].begin(), model_->pool_m_[i].end());
            std::vector<Neighbor> result;
            if (model_->pool_m_[i][start].id_ == i) start++;
            result.push_back(model_->pool_m_[i][start]);

            while (result.size() < R_
                   && (++start) < model_->pool_m_[i].size() && start < C_) {
                auto &p = model_->pool_m_[i][start];
                bool occlude = false;
                for (const auto &res: result) {
                    if (p.id_ == res.id_) {
                        occlude = true;
                        break;
                    }
                    DistResType djk = 0;
                    dist_op_.calc(data_modal1_, data_modal2_, dim1_, dim2_, res.id_, p.id_, djk);
                    float cos_ij = (p.distance_ + res.distance_ - djk) / 2 / sqrt(p.distance_ * res.distance_);
                    if (cos_ij > threshold) {
                        occlude = true;
                        break;
                    }
//                    if (djk < p.distance_) {
//                        occlude = true;
//                        break;
//                    }
                }
                if (!occlude) result.push_back(p);
            }

            model_->cut_graph_[i] = result;
            std::vector<Neighbor>().swap(model_->pool_m_[i]);
        }
        std::vector<std::vector<Neighbor>>().swap(model_->pool_m_);
        return CStatus();
    }
};

#endif //INDEXING_AND_SEARCH_C3_NEIGHBOR_NSG_V1_H

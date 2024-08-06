//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_DISTANCES_INCLUDE_H
#define INDEXING_AND_SEARCH_DISTANCES_INCLUDE_H

#include <utility>

#include "euclidean_distance.h"
#include "hamming_distance.h"
#include "attribute_distance.h"
#include "manhattan_distance.h"
#include "inner_product.h"
#include "attribute_similarity.h"
#include "euclidean_distance_eigen.h"

using VecValType = float;
using DistResType = float;

// distances calculation type
using DistInnerProduct = CGraph::UDistanceCalculator<VecValType, DistResType, InnerProduct<VecValType, DistResType>>;

template<typename DistCalcType = DistInnerProduct,
        typename TVec = VecValType,    // vector type of modal
        typename TRes = DistResType>    // fusion distance type
struct BiDistanceCalculator {
    DistCalcType dist_op_;
    std::vector<float> weight_ = Params.w_;

    explicit BiDistanceCalculator() = default;

    CStatus calculate(const std::vector<TVec *> &vec_1, const std::vector<TVec *> &vec_2,
                      const std::vector<unsigned> &dim_1, const std::vector<unsigned> &dim_2,
                      DistResType &result) {
        CStatus status;
        result = 0.f;
        for (int i = 0; i < vec_1.size(); ++i) {
            if (weight_[i] == 0) continue;
            DistResType res = 0.f;
            status += dist_op_.calculate(vec_1[i], vec_2[i], dim_1[i], dim_2[i], res);
            result += status.isOK() ? res * weight_[i] : 0.f;
        }
        return status;
    }
};

#endif //INDEXING_AND_SEARCH_DISTANCES_INCLUDE_H

//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_EUCLIDEAN_DISTANCE_EIGEN_H
#define INDEXING_AND_SEARCH_EUCLIDEAN_DISTANCE_EIGEN_H

#include "euclidean_distance.h"

#if GA_USE_EIGEN3

#include <Eigen/Core>

template<typename TVec = float, typename TRes = TVec, const bool needSqrt = false>
class EuclideanDistanceEigen : public EuclideanDistance<TVec, TRes, needSqrt> {
    using EigenDynamicMatrix = Eigen::Map<Eigen::Matrix<TVec, 1, Eigen::Dynamic> >;

public:
    CStatus calc(const TVec *a, const TVec *b, CSize dim1, CSize dim2, TRes &res, CVoidPtr ext) override {
        EigenDynamicMatrix v1((TVec *)a, dim1);
        EigenDynamicMatrix v2((TVec *)b, dim2);

        res = static_cast<TRes>(needSqrt ? (v1-v2).norm() : (v1-v2).squaredNorm());    // calc euclidean distance with sqrt
        return CStatus();
    }
};

#else

/** degrade to basic version if input eigen failed */
template<typename TVec = float, typename TRes = TVec, const bool needSqrt = false>
using EuclideanDistanceEigen = EuclideanDistance<TVec, TRes, needSqrt>;

#endif

#endif //INDEXING_AND_SEARCH_EUCLIDEAN_DISTANCE_EIGEN_H

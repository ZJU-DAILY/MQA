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

using VecValType1 = float;   // vector value type
using VecValType2 = int;   // vector value type
using DistResType1 = float;
using DistResType2 = float;
using DistResType = float;  // distances value type

// distances calculation type
using DistInnerProduct = CGraph::UDistanceCalculator<VecValType1, DistResType1, InnerProduct<VecValType1, DistResType1> >;
using DistAttributeSimilarity = CGraph::UDistanceCalculator<VecValType2, DistResType2, AttributeSimilarity<VecValType2, DistResType2> >;
using DistManhattanDistance = CGraph::UDistanceCalculator<VecValType2, DistResType2, ManhattanDistance<VecValType2, DistResType2> >;

template<typename DistCalcType1 = DistInnerProduct,
        typename DistCalcType2 = DistAttributeSimilarity,
        typename TVec1 = VecValType1,    // vector type of modal1
        typename TVec2 = VecValType2,    // vector type of modal2
        typename TRes1 = DistResType1,
        typename TRes2 = DistResType2,
        typename TRes = DistResType>    // fusion distance type
struct BiDistanceCalculator {
    DistCalcType1 dist_op1_;
    DistCalcType2 dist_op2_;

    explicit BiDistanceCalculator() = default;

    explicit BiDistanceCalculator(const std::vector<float>& w1, const std::vector<float>& w2) {
        set_weight(w1, w2);
    }

    std::vector<float> weight_1_ = Params.w1_;
    std::vector<float> weight_2_ = Params.w2_;

    void set_weight(const std::vector<float>& a, const std::vector<float>& b) {
        weight_1_ = a;
        weight_2_ = b;
    }

    // float * float
    CStatus calc_vec1(const TVec1 *vec_11, const TVec1 *vec_21, unsigned dim_11, unsigned dim_21,
                      DistResType &result) {
        CStatus status;
        DistResType1 res_1;

        status += dist_op1_.calculate(vec_11, vec_21, dim_11, dim_21, res_1);
        result = status.isOK() ? res_1 : 0.f;

        return status;
    }

    // int * int
    CStatus calc_vec2(const TVec2 *vec_12, const TVec2 *vec_22, unsigned dim_12, unsigned dim_22,
                      DistResType &result) {
        CStatus status;
        DistResType2 res_2;

        status += dist_op2_.calculate(vec_12, vec_22, dim_12, dim_22, res_2);
        result = status.isOK() ? res_2 : 0.f;
        return status;
    }

    CStatus calc(std::vector<VecValType1 *> data_modal1_, std::vector<VecValType2 *> data_modal2_,
                 std::vector<unsigned> dim1_, std::vector<unsigned> dim2_,
                 size_t a, size_t b, DistResType &result) {
        std::vector<float*> modal1_a;
        std::vector<float*> modal1_b;
        for (int it = 0; it < data_modal1_.size(); ++it) {
            auto modal1 = data_modal1_[it];
            modal1_a.emplace_back(modal1 + a * dim1_[it]);
            modal1_b.emplace_back(modal1 + b * dim1_[it]);
        }
        std::vector<int*> modal2_a;
        std::vector<int*> modal2_b;
        for (int it = 0; it < data_modal2_.size(); ++it) {
            auto modal2 = data_modal2_[it];
            modal2_a.emplace_back(modal2 + (size_t)a * dim2_[it]);
            modal2_b.emplace_back(modal2 + (size_t)b * dim2_[it]);
        }
        CStatus status = calculate(modal1_a, modal1_b, dim1_, dim1_,
                                   modal2_a, modal2_b, dim2_, dim2_,
                                   result);
        if (!status.isOK()) {
            result = 0;
        }
        return status;
    }

    CStatus calculate(const std::vector<TVec1 *> vec_11, const std::vector<TVec1 *> vec_21, const std::vector<unsigned>& dim_11, const std::vector<unsigned>& dim_21,
                      const std::vector<TVec2 *> vec_12, const std::vector<TVec2 *> vec_22, const std::vector<unsigned>& dim_12, const std::vector<unsigned>& dim_22,
                      DistResType &result) {
        CStatus status;
        std::vector<DistResType1> res_1;
        std::vector<DistResType2> res_2;

        for (int i = 0; i < vec_11.size(); ++i) {
            if (weight_1_[i] == 0) {
                res_1.emplace_back(0);
                continue;
            }
            DistResType1 res = 0.f;
            status += dist_op1_.calculate(vec_11[i], vec_21[i], dim_11[i], dim_21[i], res);
            res_1.emplace_back(res * weight_1_[i]);
        }
        for (int i = 0; i < vec_12.size(); ++i) {
            if (weight_2_[i] == 0) {
                res_2.emplace_back(0);
                continue;
            }
            DistResType2 res = 0.f;
            status += dist_op2_.calculate(vec_12[i], vec_22[i], dim_12[i], dim_22[i], res);
            res_2.emplace_back(res * weight_2_[i]);
        }

        result = 0.f;
        if (status.isOK()) {
            for (auto res: res_1) {
                result += res;
            }
            for (auto res: res_2) {
                result += res;
            }
        }
        return status;
    }
};

#endif //INDEXING_AND_SEARCH_DISTANCES_INCLUDE_H

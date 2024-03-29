//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_COMPONENTS_BASIC_H
#define INDEXING_AND_SEARCH_COMPONENTS_BASIC_H

#include "../../CGraph/src/CGraph.h"
#include "../utils/utils.h"

using DistCalcType = BiDistanceCalculator<DistInnerProduct, DistAttributeSimilarity>;

class ComponentsBasic : public CGraph::DAnnNode {
protected:
    AnnsModelParam *model_ = nullptr;          // ann model ptr
    std::vector<VecValType1 *> data_modal1_ = {};               // vector data
    std::vector<VecValType2 *> data_modal2_ = {};
    size_t num_ = 0;                         // number of vector
    std::vector<unsigned> dim1_ = {};        // dimensionality of vector for modal1
    std::vector<unsigned> dim2_ = {};        // dimensionality of vector for modal2
    DistCalcType dist_op_;    // distance calculation type
};

#endif //INDEXING_AND_SEARCH_COMPONENTS_BASIC_H

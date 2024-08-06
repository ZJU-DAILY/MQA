//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_COMPONENTS_BASIC_H
#define INDEXING_AND_SEARCH_COMPONENTS_BASIC_H

#include "../../CGraph/src/CGraph.h"
#include "../utils/utils.h"

using DistCalcType = BiDistanceCalculator<DistInnerProduct>;

class ComponentsBasic : public CGraph::DAnnNode {
protected:
    AnnsModelParam *model_ = nullptr;           // ann model ptr
    std::vector<VecValType *> data_modal_ = {}; // vector data
    size_t num_ = 0;                            // number of vector
    std::vector<unsigned> dim_ = {};            // dimensionality of vector for modal
    DistCalcType dist_op_;                      // distance calculation type
};

#endif //INDEXING_AND_SEARCH_COMPONENTS_BASIC_H

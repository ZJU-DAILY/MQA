//
// Created by MaouSanta on 2024/2/18.
//

#ifndef INDEXING_AND_SEARCH_C3_NEIGHBOR_BASIC_H
#define INDEXING_AND_SEARCH_C3_NEIGHBOR_BASIC_H

#include <cmath>

#include "../components_basic.h"
#include "../../utils/utils.h"

class C3NeighborBasic : public ComponentsBasic {
protected:
    unsigned C_ = 0;
    unsigned R_ = 0;
    std::vector<Neighbor> result_;
};

#endif //INDEXING_AND_SEARCH_C3_NEIGHBOR_BASIC_H

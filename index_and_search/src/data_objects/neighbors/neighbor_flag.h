//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_NEIGHBOR_FLAG_H
#define INDEXING_AND_SEARCH_NEIGHBOR_FLAG_H

#include "neighbor_basic.h"

struct NeighborFlag : public Neighbor {
public:
    explicit NeighborFlag() : Neighbor(), flag_(false) {}
    explicit NeighborFlag(unsigned id, DistResType distance, bool f) : Neighbor(id, distance), flag_(f) {}

public:
    bool flag_;
};

#endif //INDEXING_AND_SEARCH_NEIGHBOR_FLAG_H

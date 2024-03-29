//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_NEIGHBOR_BASIC_H
#define INDEXING_AND_SEARCH_NEIGHBOR_BASIC_H

#include "../distances/distances_include.h"
#include "../data_objects_define.h"

struct Neighbor {
public:
    explicit Neighbor() : id_(-1), distance_(0) {}
    explicit Neighbor(unsigned id, DistResType distance) : id_{id}, distance_{distance} {}

    inline bool operator<(const Neighbor &other) const {
        return distance_ < other.distance_;
    }

    inline bool operator>(const Neighbor &other) const {
        return distance_ > other.distance_;
    }

public:
    IDType id_;
    DistResType distance_;
};

#endif //INDEXING_AND_SEARCH_NEIGHBOR_BASIC_H

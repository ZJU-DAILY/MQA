//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_INSERT_INTO_POOL_H
#define INDEXING_AND_SEARCH_INSERT_INTO_POOL_H


#include "../../data_objects/neighbors/neighbors_include.h"
#include <cstring>

/**
 * no repeated insertion by binary search
 * @param addr
 * @param K
 * @param nn
 * @return
 */
int InsertIntoPool(NeighborFlag *addr, unsigned K, const NeighborFlag& nn) {
    // find the location to insert
    int left = 0, right = K - 1;
    if (addr[left].distance_ > nn.distance_) {
        memmove((char *) &addr[left + 1], &addr[left], K * sizeof(NeighborFlag));
        addr[left] = nn;
        return left;
    }
    if (addr[right].distance_ < nn.distance_) {
        addr[K] = nn;
        return K;
    }
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (addr[mid].distance_ > nn.distance_) right = mid;
        else left = mid;
    }
    //check equal ID

    while (left > 0) {
        if (addr[left].distance_ < nn.distance_) break;
        if (addr[left].id_ == nn.id_) return K + 1;
        left--;
    }
    if (addr[left].id_ == nn.id_ || addr[right].id_ == nn.id_) return K + 1;
    memmove((char *) &addr[right + 1], &addr[right], (K - right) * sizeof(NeighborFlag));
    addr[right] = nn;
    return right;
}

#endif //INDEXING_AND_SEARCH_INSERT_INTO_POOL_H

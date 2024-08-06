//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_GEN_RANDOM_ID_H
#define INDEXING_AND_SEARCH_GEN_RANDOM_ID_H

#include <random>

/**
 * generate random id number
 * @param id
 * @param num 是总个数
 * @param size 生成的n个数，范围在 [0,num) 之间。不重复
 */

template<typename TID = IDType, typename TSize = unsigned>
void GenRandomID(TID *id, size_t num, TSize size) {
    std::random_device rd;
    std::mt19937 rng(rd());

    std::vector<TID> pool(num);
    for (unsigned i = 0; i < num; ++i) pool[i] = i;
    std::shuffle(pool.begin(), pool.end(), rng);

    for (unsigned i = 0; i < size; ++i) {
        id[i] = pool[i];
    }
}

#endif //INDEXING_AND_SEARCH_GEN_RANDOM_ID_H

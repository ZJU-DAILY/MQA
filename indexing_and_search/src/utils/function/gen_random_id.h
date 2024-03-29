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
    std::mt19937 rng(rand());
    for (TSize i = 0; i < size; ++i) {
        id[i] = rng() % (num - size);
    }
    std::sort(id, id + size);

    for (TSize i = 1; i < size; ++i) {
        if (id[i] <= id[i - 1]) {
            id[i] = id[i - 1] + 1;
        }
    }
    TID off = rng() % num;
    for (TSize i = 0; i < size; ++i) {
        id[i] = (id[i] + off) % num;
    }
}

template<typename TID = IDType, typename TSize = unsigned>
void GenRandomID(std::vector<TID>& id, size_t num, TSize size) {
    std::random_device rnd;
    std::mt19937 rng(rnd());
    std::vector<IDType> numbers(num);
    std::iota(numbers.begin(), numbers.end(), 0);
    std::shuffle(numbers.begin(), numbers.end(), rng);
    numbers.resize(size);
    id = numbers;
}

#endif //INDEXING_AND_SEARCH_GEN_RANDOM_ID_H

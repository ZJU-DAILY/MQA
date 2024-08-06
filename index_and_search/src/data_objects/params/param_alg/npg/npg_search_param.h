//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_NPG_SEARCH_PARAM_H
#define INDEXING_AND_SEARCH_NPG_SEARCH_PARAM_H

#include "../alg_param_basic.h"

struct NPGSearchParam : public BasicParam {
    unsigned search_L = Params.L_search_;
    unsigned query_id = 0;
    std::vector<unsigned> modal_query_id = {};

    std::vector<NeighborFlag> sp;   // start point
    std::set<Neighbor> ss;          // search set

    CVoid reset() {
        ss.clear();
        sp.clear();
    }
};

#endif //INDEXING_AND_SEARCH_NPG_SEARCH_PARAM_H

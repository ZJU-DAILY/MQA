//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_NPG_SEARCH_PARAM_H
#define INDEXING_AND_SEARCH_NPG_SEARCH_PARAM_H

#include "../alg_param_basic.h"

struct NPGSearchParam : public BasicParam {
    unsigned search_L = Params.L_search_;
    unsigned query_id = 0;
    std::vector<unsigned> modal1_query_id = {};
    std::vector<unsigned> modal2_query_id = {};

    std::vector<NeighborFlag> sp;
    std::vector<std::vector<NeighborFlag> > sp_modal1;
    std::vector<std::vector<NeighborFlag> > sp_modal2;

    CVoid reset() {
        sp.clear();
        sp_modal1.clear();
        sp_modal2.clear();
    }
};

#endif //INDEXING_AND_SEARCH_NPG_SEARCH_PARAM_H

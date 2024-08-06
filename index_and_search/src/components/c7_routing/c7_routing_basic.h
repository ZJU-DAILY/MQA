//
// Created by MaouSanta on 2024/2/18.
//

#ifndef INDEXING_AND_SEARCH_C7_ROUTING_BASIC_H
#define INDEXING_AND_SEARCH_C7_ROUTING_BASIC_H

#include "../components_basic.h"
#include "../../utils/utils.h"

class C7RoutingBasic : public ComponentsBasic {
protected:
    unsigned search_L_;                             // candidate pool size for search
    unsigned K_;                                    // top-k for search
    std::vector<VecValType *> query_modal_;         // query data
    unsigned query_id_;                             // current query id
    std::vector<std::pair<VecValType, IDType>> res_;// current query result
    unsigned delete_num_each_query_ = 0;
};

#endif //INDEXING_AND_SEARCH_C7_ROUTING_BASIC_H

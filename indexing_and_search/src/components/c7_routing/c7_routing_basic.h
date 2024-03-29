//
// Created by MaouSanta on 2024/2/18.
//

#ifndef INDEXING_AND_SEARCH_C7_ROUTING_BASIC_H
#define INDEXING_AND_SEARCH_C7_ROUTING_BASIC_H

#include "../components_basic.h"
#include "../../utils/utils.h"

class C7RoutingBasic : public ComponentsBasic {
protected:
    unsigned search_L_; // candidate pool size for search
    unsigned K_;    // top-k for search
    std::vector<VecValType1 *> query_modal1_;    // query data
    std::vector<VecValType2 *> query_modal2_;    // query data
    unsigned query_id_; // current query id
    std::vector<unsigned> query_id_modal1_; // current modal1 query id
    std::vector<unsigned> query_id_modal2_; // current modal2 query id
//    std::vector<IDType> res_; // current query result
    std::vector<IDType> res_;
    std::vector<VecValType1> res_dist_;
    std::vector<std::vector<IDType> > res_modal1_; // current modal1 query result
    std::vector<std::vector<IDType> > res_modal2_; // current modal2 query result
    unsigned delete_num_each_query_ = 0;
};

#endif //INDEXING_AND_SEARCH_C7_ROUTING_BASIC_H

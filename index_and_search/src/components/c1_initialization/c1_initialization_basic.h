//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_C1_INITIALIZATION_BASIC_H
#define INDEXING_AND_SEARCH_C1_INITIALIZATION_BASIC_H

#include <string>
#include <cassert>

#include "../components_basic.h"
#include "../../utils/utils.h"

class C1InitializationBasic : public ComponentsBasic {
protected:
    unsigned out_degree_ = 0;        // out-degree of initial graph
};

#endif //INDEXING_AND_SEARCH_C1_INITIALIZATION_BASIC_H

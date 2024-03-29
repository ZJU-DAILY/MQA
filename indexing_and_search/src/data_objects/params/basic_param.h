//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_BASIC_PARAM_H
#define INDEXING_AND_SEARCH_BASIC_PARAM_H

#include <string>
#include <fstream>

#include "../../utils/utils.h"
#include "params_define.h"
#include "../../../CGraph/src/CGraph.h"
#include "../../graph_anns_define.h"

struct BasicParam : public CGraph::GParam {
    CVoid reset() {
    }
};

#endif //INDEXING_AND_SEARCH_BASIC_PARAM_H

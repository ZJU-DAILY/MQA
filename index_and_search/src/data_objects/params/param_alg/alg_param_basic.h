//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_ALG_PARAM_BASIC_H
#define INDEXING_AND_SEARCH_ALG_PARAM_BASIC_H

#include "../basic_param.h"

struct AlgParamBasic : public BasicParam {
    unsigned top_k = Params.top_k_;

    std::vector<std::vector<std::pair<float, IDType>>> results_;

    CVoid reset() {
    }
};

#endif //INDEXING_AND_SEARCH_ALG_PARAM_BASIC_H

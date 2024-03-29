//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_ALG_PARAM_BASIC_H
#define INDEXING_AND_SEARCH_ALG_PARAM_BASIC_H

#include "../basic_param.h"

struct AlgParamBasic : public BasicParam {
    unsigned top_k = Params.top_k_;

    std::vector<std::vector<float>> results_dist;
    std::vector<std::vector<std::vector<float> > > results_dist1;
    std::vector<std::vector<std::vector<float> > > results_dist2;
    std::vector<std::vector<IDType> > results;
    std::vector<std::vector<std::vector<IDType> > > results_modal1;
    std::vector<std::vector<std::vector<IDType> > > results_modal2;

    CVoid reset() {
    }
};

#endif //INDEXING_AND_SEARCH_ALG_PARAM_BASIC_H

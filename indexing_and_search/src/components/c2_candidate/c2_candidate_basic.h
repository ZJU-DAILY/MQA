//
// Created by MaouSanta on 2024/2/18.
//

#ifndef INDEXING_AND_SEARCH_C2_CANDIDATE_BASIC_H
#define INDEXING_AND_SEARCH_C2_CANDIDATE_BASIC_H

#include "../components_basic.h"
#include "../../utils/utils.h"

class C2CandidateBasic : public ComponentsBasic {
protected:
    IDType cur_id_ = 0;  // data id being processed
    unsigned L_ = 0;
};

#endif //INDEXING_AND_SEARCH_C2_CANDIDATE_BASIC_H

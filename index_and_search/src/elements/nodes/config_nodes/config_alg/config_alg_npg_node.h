//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_CONFIG_ALG_NPG_NODE_H
#define INDEXING_AND_SEARCH_CONFIG_ALG_NPG_NODE_H

#include "../config_basic.h"
#include "../../../../data_objects/data_objects.h"

class ConfigAlgNPGNode : public ConfigBasic {
public:
    CStatus init() override {
        CGRAPH_EMPTY_FUNCTION
    }

    CStatus run() override {
        auto *t_param = CGRAPH_GET_GPARAM(NPGTrainParam, GA_ALG_NPG_TRAIN_PARAM_KEY)
        CGRAPH_ASSERT_NOT_NULL(t_param)

        t_param->L_candidate = Params.L_candidate_;
        t_param->R_neighbor = Params.R_neighbor_;
        t_param->C_neighbor = Params.C_neighbor_;
        t_param->k_init_graph = Params.k_init_graph_;

        return CStatus();
    }
};

#endif //INDEXING_AND_SEARCH_CONFIG_ALG_NPG_NODE_H

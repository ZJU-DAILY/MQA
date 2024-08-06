//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_NPG_TRAIN_PARAM_H
#define INDEXING_AND_SEARCH_NPG_TRAIN_PARAM_H

#include "../../basic_param.h"

struct NPGTrainParam : public BasicParam {
    unsigned L_candidate = Params.L_candidate_;    // size of candidate set for neighbor selection
    unsigned R_neighbor = Params.R_neighbor_;      // size of neighbor set
    unsigned C_neighbor = Params.C_neighbor_;      // number of visited candidate neighbors when neighbor selection
    unsigned k_init_graph = Params.k_init_graph_;  // number of neighbors of initial graph
    unsigned nn_size = Params.nn_size_;            // size of candidate neighbors during nn-descent
    unsigned rnn_size = Params.rnn_size_;          // size of reverse candidate neighbors during nn-descent
    unsigned pool_size = Params.pool_size_;        // size of neighbor pool during nn-descent
    unsigned iter = Params.iter_;                  // number of nn-descent iteration
    unsigned sample_num = Params.sample_num_;      // number of sample data when evaluating graph quality for each iteration
    float graph_quality_threshold = Params.graph_quality_threshold_;         // graph quality threshold

    CVoid reset() {

    }
};

#endif //INDEXING_AND_SEARCH_NPG_TRAIN_PARAM_H

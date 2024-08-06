//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_ANNS_MODEL_PARAM_H
#define INDEXING_AND_SEARCH_ANNS_MODEL_PARAM_H

#include "../../../CGraph/src/CGraph.h"
#include "../neighbors/neighbors_include.h"
#include "../meta/meta_include.h"

struct AnnsModelParam : public CGraph::GParam {
    unsigned ep_;
    std::vector<std::vector<Neighbor>> graph_n_;
    std::vector<std::vector<IDType>> graph_m_;

    std::vector<MetaData<VecValType>> train_meta_modal_;
    std::vector<MetaData<VecValType>> search_meta_modal_;

    std::vector<IDType> sample_points_;
    std::vector<std::vector<IDType>> knn_set_;
    MetaVector<IDType> eva_meta_;
    MetaData<IDType> delete_meta_;

    CVoid reset() {
    }
};

#endif //INDEXING_AND_SEARCH_ANNS_MODEL_PARAM_H

//
// Created by MaouSanta on 2024/5/22.
//

#ifndef INDEXING_AND_SEARCH_EVAL_RECALL_H
#define INDEXING_AND_SEARCH_EVAL_RECALL_H

template<typename T>
float eval_quality(
        const std::vector<T> &graph_pool_,
        const std::vector<IDType> &ctrl_points,
        const std::vector<std::vector<IDType>> &knn_set) {
    float mean_acc = 0;
    unsigned ctrl_points_size = ctrl_points.size();

#pragma omp parallel for num_threads(Params.thread_num_) schedule(dynamic) \
                         shared(ctrl_points_size, ctrl_points, knn_set, mean_acc) default(none)

    for (unsigned i = 0; i < ctrl_points_size; i++) {
        unsigned acc = 0;
        for (auto &j: graph_pool_[ctrl_points[i]]) {
            for (IDType k: knn_set[i]) {
                if (j.id_ == k) {
                    acc++;
                    break;
                }
            }
        }
#pragma omp atomic
        mean_acc += ((float) acc / (float) knn_set[i].size());
    }

    float rc = mean_acc / (float) ctrl_points_size;
    printf("[EVA] graph quality: %f\n", rc);
    return rc;
}
//
//float eval_quality(
//        const std::vector<std::set<Neighbor>> &graph_pool_,
//        const std::vector<IDType> &ctrl_points,
//        const std::vector<std::vector<IDType>> &knn_set) {
//    float mean_acc = 0;
//    unsigned ctrl_points_size = ctrl_points.size();
//
//#pragma omp parallel for num_threads(Params.thread_num_) schedule(dynamic) \
//                         shared(ctrl_points_size, ctrl_points, knn_set, mean_acc) default(none)
//
//    for (unsigned i = 0; i < ctrl_points_size; i++) {
//        unsigned acc = 0;
//        for (auto &j: graph_pool_[ctrl_points[i]]) {
//            for (IDType k: knn_set[i]) {
//                if (j.id_ == k) {
//                    acc++;
//                    break;
//                }
//            }
//        }
//#pragma omp atomic
//        mean_acc += ((float) acc / (float) knn_set[i].size());
//    }
//
//    float rc = mean_acc / (float) ctrl_points_size;
//    printf("[EVA] graph quality: %f\n", rc);
//    return rc;
//}

#endif //INDEXING_AND_SEARCH_EVAL_RECALL_H

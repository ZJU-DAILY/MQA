//
// Created by MaouSanta on 2024/2/22.
//

#include "src/graph_anns.h"
//#include "src/components/c1_initialization/implementation/c1_initialization_nndescent.h"
//#include "src/components/c1_initialization/implementation/c1_initialization_vamana.h"

using namespace CGraph;

const float eps = 1e-3;

int main(int argc, char **argv) {
    time_t tt = time(nullptr);
    tm *t = localtime(&tt);
    std::cout << std::endl;
    std::cout << t->tm_year + 1900 << "-" << t->tm_mon + 1 << "-" << t->tm_mday
              << "-" << t->tm_hour << "-" << t->tm_min << "-" << t->tm_sec << std::endl;
    std::cout << std::endl;

    auto get_param = [&](unsigned type, int aux = 0) {
        static unsigned argv_count = 0;
        char *ret = argv[argv_count++];
        switch (type) {
            case 0:
                std::cout << "[RUN] Execution object: ";
                break;
            case 1:
                std::cout << "[PARAM] Modal number: ";
                break;
            case 2:
                std::cout << "[PARAM] Modal[" << aux << "] base path: ";
                break;
            case 3:
                std::cout << "[PARAM] Modal[" << aux << "] query path: ";
                break;
            case 4:
                std::cout << "[PARAM] Modal[" << aux << "] weight: ";
                break;
            case 5:
                std::cout << "[PARAM] result number: ";
                break;
            case 6:
                std::cout << "[PARAM] Save Result path: ";
                break;
            case 7:
                std::cout << "[PARAM] R_neighbor: ";
                break; // neighbor size
            case 8:
                std::cout << "[PARAM] C_neighbor: ";
                break; // candidate size
            case 9:
                std::cout << "[PARAM] delete_id_path: ";
                break;
            case 21:
                std::cout << "[PARAM] index_path: ";
                break;
            case 22:
                std::cout << "[PARAM] index_method: ";
                break;
            default:
                assert(false);
        }
        std::cout << ret << std::endl;
        return ret;
    };

    std::vector<char *> modal_base_path;
    std::vector<float> modal_weight;
    std::vector<unsigned> is_norm_modal;

    get_param(0);
    unsigned modal1_num = strtoul(get_param(1), nullptr, 10);
    for (int i = 0; i < modal1_num; ++i) {
        modal_base_path.emplace_back(get_param(2, i));
        modal_weight.emplace_back(strtof(get_param(4, i), nullptr));
    }
    unsigned thread_num = 4;
    char *index_method = get_param(22);
    unsigned R_neighbor = strtoul(get_param(7), nullptr, 10);
    unsigned C_neighbor = strtoul(get_param(8), nullptr, 10);
    char *index_path = get_param(21);

    Params.set_general_param(thread_num);
    Params.set_data_param(modal_weight);
    Params.set_train_param(modal_base_path, index_path, R_neighbor, C_neighbor);

    GPipelinePtr pipeline = GPipelineFactory::create();
    pipeline->createGParam<NPGTrainParam>(GA_ALG_NPG_TRAIN_PARAM_KEY);
    pipeline->createGParam<NPGSearchParam>(GA_ALG_NPG_SEARCH_PARAM_KEY);
    pipeline->createGParam<AlgParamBasic>(GA_ALG_PARAM_BASIC_KEY);
    pipeline->createGParam<AnnsModelParam>(GA_ALG_MODEL_PARAM_KEY);
    GElementPtr a, b, c, d, e, f = nullptr;
    GElementPtr c1, c2, c3, c4, c5, region;

    CStatus status = pipeline->registerGElement<ConfigAlgNPGNode>(&a, {}, "config_npg");
    status += pipeline->registerGElement<ConfigModelNode>(&b, {a}, "config_model");
    status += pipeline->registerGElement<LoadDataNode>(&c, {b}, "load_data");
    status += pipeline->registerGElement<EvaGenerateSampleNode>(&d, {c}, "eva_generate");
//    status += pipeline->registerGElement<LoadIndexNode>(&e, {d}, "load_index");

    if (strcmp(index_method, "NSG") == 0) {
        c1 = pipeline->createGNode<C1InitializationNNDescent>(GNodeInfo("c1_nndescent"));
        c2 = pipeline->createGNode<C2CandidateGreedy>(GNodeInfo("c2_greedy"));
        c3 = pipeline->createGNode<C3NeighborNSG>(GNodeInfo("c3_nsg"));
        c4 = pipeline->createGNode<C4PreprocessingCentroid>(GNodeInfo("c4_centroid"));
        c5 = pipeline->createGNode<C5ConnectivityDFS>(GNodeInfo("c5_dfs"));
        region = pipeline->createGGroup<GCluster>({c1, c2, c3, c4, c5});
    } else if (strcmp(index_method, "NSSG") == 0) {
        c1 = pipeline->createGNode<C1InitializationNNDescent>(GNodeInfo("c1_nndescent"));
        c2 = pipeline->createGNode<C2CandidateFetch>(GNodeInfo("c2_fetch"));
        c3 = pipeline->createGNode<C3NeighborNSSG>(GNodeInfo("c3_nssg"));
        c4 = pipeline->createGNode<C4PreprocessingCentroid>(GNodeInfo("c4_random"));
        c5 = pipeline->createGNode<C5ConnectivityDFS>(GNodeInfo("c5_dfs"));
        region = pipeline->createGGroup<GCluster>({c1, c2, c3, c4, c5});
    } else if (strcmp(index_method, "Vamana") == 0) {
        c1 = pipeline->createGNode<C1InitializationRandom>(GNodeInfo("c1_random"));
        c2 = pipeline->createGNode<C2CandidateGreedy>(GNodeInfo("c2_greedy"));
        c3 = pipeline->createGNode<C3NeighborVamana>(GNodeInfo("c3_vamana"));
        c4 = pipeline->createGNode<C4PreprocessingCentroid>(GNodeInfo("c4_centroid"));
        c5 = pipeline->createGNode<C5ConnectivityDFS>(GNodeInfo("c5_dfs"));
        region = pipeline->createGGroup<GCluster>({c1, c2, c3, c4, c5});
    } else if (strcmp(index_method, "KGraph") == 0) {
        c1 = pipeline->createGNode<C1InitializationNNDescent>(GNodeInfo("c1_nndescent"));
        region = pipeline->createGGroup<GCluster>({c1});
    }

    status += pipeline->registerGElement<GCluster>(&region, {d}, "build");
    status += pipeline->registerGElement<SaveIndexNode>(&f, {region}, "save_index");

    pipeline->addGAspect<TimerAspect>()->addGAspect<TraceAspect>();
    status += pipeline->process();
    if (!status.isOK()) {
        CGRAPH_ECHO("process graph error, error info is [%s]", status.getInfo().c_str());
        exit(status.getCode());
    }
    GPipelineFactory::remove(pipeline);
    return 0;
}
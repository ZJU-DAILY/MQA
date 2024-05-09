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
            case 11:
                std::cout << "[PARAM] Modal(int) number: ";
                break;
            case 12:
                std::cout << "[PARAM] Modal(int)[" << aux << "] base path: ";
                break;
            case 13:
                std::cout << "[PARAM] Modal(int)[" << aux << "] query path: ";
                break;
            case 14:
                std::cout << "[PARAM] Modal[" << aux << "] weight: ";
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

    std::vector<char *> modal1_base_path;
    std::vector<char *> modal2_base_path;
    std::vector<float> modal1_weight;
    std::vector<float> modal2_weight;
    std::vector<unsigned> is_norm_modal1;
    std::vector<unsigned> is_norm_modal2;

    get_param(0);
    unsigned modal1_num = strtoul(get_param(1), nullptr, 10);
    for (int i = 0; i < modal1_num; ++i) {
        modal1_base_path.emplace_back(get_param(2, i));
        modal1_weight.emplace_back(strtof(get_param(4, i), nullptr));
        is_norm_modal1.emplace_back(std::fabs(modal1_weight[i]) > eps);
    }
    unsigned thread_num = 1;
    char *index_method = get_param(22);
    unsigned R_neighbor = strtoul(get_param(7), nullptr, 10);
    unsigned C_neighbor = strtoul(get_param(8), nullptr, 10);
    char *index_path = get_param(21);

    Params.set_general_param(thread_num, is_norm_modal1, is_norm_modal2);
    Params.set_data_param(modal1_weight, modal2_weight);
    Params.set_train_param(modal1_base_path, modal2_base_path, index_path, R_neighbor, C_neighbor);

    GPipelinePtr pipeline = GPipelineFactory::create();
    pipeline->createGParam<NPGTrainParam>(GA_ALG_NPG_TRAIN_PARAM_KEY);
    pipeline->createGParam<NPGSearchParam>(GA_ALG_NPG_SEARCH_PARAM_KEY);
    pipeline->createGParam<AlgParamBasic>(GA_ALG_PARAM_BASIC_KEY);
    pipeline->createGParam<AnnsModelParam>(GA_ALG_MODEL_PARAM_KEY);
    GElementPtr a, b, c, d, e, cde_region, f = nullptr;

    CStatus status = pipeline->registerGElement<ConfigAlgNPGNode>(&a, {}, "config_npg");
    status += pipeline->registerGElement<ConfigModelNode>(&b, {a}, "config_model");

    if (strcmp(index_method, "NSG") == 0) {
        c = pipeline->createGNode<C1InitializationNNDescent>(GNodeInfo("c1_nssg"));
        d = pipeline->createGNode<C2CandidateNSSGV1>(GNodeInfo("c2_nssg"));
        e = pipeline->createGNode<C3NeighborNSGV1>(GNodeInfo("c3_nsg"));
        cde_region = pipeline->createGGroup<GCluster>({c, d, e});
    }
//    }
//    else if (strcmp(index_method, "DiskANN") == 0) {
//        c = pipeline->createGNode<C1InitializationVamana>(GNodeInfo("c1_vamana"));
//        d = pipeline->createGNode<C2CandidateNSSGV1>(GNodeInfo("c2_nssg"));
//        e = pipeline->createGNode<C3NeighborNSGV1>(GNodeInfo("c3_nsg"));
//        cde_region = pipeline->createGGroup<GCluster>({c, d, e});
//    }
//    else if (strcmp(index_method, "KGraph") == 0) {
//        c = pipeline->createGNode<C1InitializationKGraph>(GNodeInfo("c1_kgraph"));
//        d = pipeline->createGNode<C2CandidateNSSGV1>(GNodeInfo("c2_nssg"));
//        e = pipeline->createGNode<C3NeighborNSGV1>(GNodeInfo("c3_nsg"));
//        cde_region = pipeline->createGGroup<GCluster>({c, d, e});
//    }
//    else if (strcmp(index_method, "HNSW") == 0) {
//        c = pipeline->createGNode<HNSWindex>(GNodeInfo("c123_HNSW"));
//        cde_region = pipeline->createGGroup<GCluster>({c});
//    }
//    else if (strcmp(index_method, "IVFPQ") == 0) {
//        c = pipeline->createGNode<C123IVFPQ>(GNodeInfo("c1_ivfpq"));
//        cde_region = pipeline->createGGroup<GCluster>({c});
//    }
    status += pipeline->registerGElement<GCluster>(&cde_region, {b}, "build");
    status += pipeline->registerGElement<SaveIndexNode>(&f, {cde_region}, "save_index");

    pipeline->addGAspect<TimerAspect>()->addGAspect<TraceAspect>();
    status += pipeline->process();
    if (!status.isOK()) {
        CGRAPH_ECHO("process graph error, error info is [%s]", status.getInfo().c_str());
        exit(status.getCode());
    }
    GPipelineFactory::remove(pipeline);
    return 0;
}
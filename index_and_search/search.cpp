//
// Created by MaouSanta on 2024/5/7.
//
#include <cstdlib>
#include "src/graph_anns.h"

using namespace CGraph;

const float eps = 1e-3;

int main(int argc, char **argv) {
    time_t tt = time(nullptr);
    tm* t=localtime(&tt);
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
    std::vector<char *> modal_query_path;
    std::vector<float> modal_weight;

    get_param(0);
    unsigned modal_num = strtoul(get_param(1), nullptr, 10);
    for (int i = 0; i < modal_num; ++i) {
        modal_base_path.emplace_back(get_param(2, i));
        modal_query_path.emplace_back(get_param(3, i));
        modal_weight.emplace_back(strtof(get_param(4, i), nullptr));
    }
    unsigned thread_num = 4;
    unsigned top_k = strtoul(get_param(5), nullptr, 10);
    char *res_path = get_param(6);
    char *delete_id_path = get_param(9);
    char *index_method = get_param(22);
    char *index_path = nullptr;
    if (strcmp(index_method, "Flat") != 0) {
        index_path = get_param(21);
    }
    Params.set_search_param(modal_base_path, modal_query_path, res_path, top_k, index_path);
    Params.set_general_param(thread_num, 1);
    Params.set_data_param(modal_weight);
    Params.set_delete_id_path(delete_id_path);

    GPipelinePtr pipeline = GPipelineFactory::create();
    pipeline->createGParam<NPGTrainParam>(GA_ALG_NPG_TRAIN_PARAM_KEY);
    pipeline->createGParam<NPGSearchParam>(GA_ALG_NPG_SEARCH_PARAM_KEY);
    pipeline->createGParam<AlgParamBasic>(GA_ALG_PARAM_BASIC_KEY);
    pipeline->createGParam<AnnsModelParam>(GA_ALG_MODEL_PARAM_KEY);

    GElementPtr a, b, f, g, h, i, p, gh_region = nullptr;
    // build
    CStatus status = pipeline->registerGElement<ConfigAlgNPGNode>(&a, {}, "config_npg");
    status += pipeline->registerGElement<ConfigModelNode>(&b, {a}, "config_model");

    if (strcmp(index_method, "Flat") == 0) {
        status += pipeline->registerGElement<EvaBruteForceNode>(&f, {a}, "eva_brute_force");
        status += pipeline->registerGElement<SaveResultNode>(&g, {f}, "save_result");
        f->addGAspect<TimerAspect>()->addGAspect<TraceAspect>();
    } else if (strcmp(index_method, "KGraph") == 0 || strcmp(index_method, "NSSG") == 0) {
        status += pipeline->registerGElement<LoadIndexNode>(&f, {a}, "load_index");
        g = pipeline->createGNode<C6SeedRandom>(GNodeInfo("c6_random"));
        h = pipeline->createGNode<C7RoutingGreedy>(GNodeInfo("c7_greedy"));
        gh_region = pipeline->createGGroup<SearchRegion>({g, h});
        status += pipeline->registerGElement<SearchRegion>(&gh_region, {f}, "search");
        status += pipeline->registerGElement<SaveResultNode>(&p, {gh_region}, "save_result");
        gh_region->addGAspect<TimerAspect>()->addGAspect<TraceAspect>();
    } else if (strcmp(index_method, "NSG") == 0 || strcmp(index_method, "Vamana") == 0) {
        status += pipeline->registerGElement<LoadIndexNode>(&f, {a}, "load_index");
        g = pipeline->createGNode<C6SeedCentroid>(GNodeInfo("c6_centroid"));
        h = pipeline->createGNode<C7RoutingGreedy>(GNodeInfo("c7_greedy"));
        gh_region = pipeline->createGGroup<SearchRegion>({g, h});
        status += pipeline->registerGElement<SearchRegion>(&gh_region, {f}, "search");
        status += pipeline->registerGElement<SaveResultNode>(&p, {gh_region}, "save_result");
        gh_region->addGAspect<TimerAspect>()->addGAspect<TraceAspect>();
    }

    status += pipeline->process();
    if (!status.isOK()) {
        CGRAPH_ECHO("process graph error, error info is [%s]", status.getInfo().c_str());
        exit(status.getCode());
    }
    GPipelineFactory::remove(pipeline);
    return 0;
}
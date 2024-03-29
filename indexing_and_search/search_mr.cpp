//
// Created by MaouSanta on 2024/3/16.
//
#include "src/graph_anns.h"
#include "src/components/c1_initialization/implementation/c1_initialization_nndescent.h"

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
            default:
                assert(false);
        }
        std::cout << ret << std::endl;
        return ret;
    };

    std::vector<char *> modal1_base_path;
    std::vector<char *> modal2_base_path;
    std::vector<char *> modal1_query_path;
    std::vector<char *> modal2_query_path;
    std::vector<float> modal1_weight;
    std::vector<float> modal2_weight;
    std::vector<unsigned> is_norm_modal1;
    std::vector<unsigned> is_norm_modal2;

    get_param(0);
    unsigned modal1_num = strtoul(get_param(1), nullptr, 10);
    for (int i = 0; i < modal1_num; ++i) {
        modal1_base_path.emplace_back(get_param(2, i));
        modal1_query_path.emplace_back(get_param(3, i));
        modal1_weight.emplace_back(strtof(get_param(4, i), nullptr));
        is_norm_modal1.emplace_back(std::fabs(modal1_weight[i]) > eps);
    }
    unsigned thread_num = 1;
    unsigned top_k = strtoul(get_param(5), nullptr, 10);
    char *res_path = get_param(6);
    char *delete_id_path = get_param(9);
    Params.set_search_param(modal1_base_path, modal2_base_path, modal1_query_path, modal2_query_path, res_path, top_k);
    Params.set_general_param(thread_num, is_norm_modal1, is_norm_modal2, 1);
    Params.set_candidate_top_k(10); // number of result's candidates
    Params.set_data_param(modal1_weight, modal2_weight);
    Params.set_delete_id_path(delete_id_path);

    GPipelinePtr pipeline = GPipelineFactory::create();
    pipeline->createGParam<AnnsModelParam>(GA_ALG_MODEL_PARAM_KEY);
    pipeline->createGParam<AlgParamBasic>(GA_ALG_PARAM_BASIC_KEY);

    GElementPtr a, b, c, d, f, g;
    CStatus status = pipeline->registerGElement<ConfigAlgBruteForceNode>(&a, {}, "config_brute_force");
    status += pipeline->registerGElement<ConfigModelNode>(&b, {a}, "config_model");
    status += pipeline->registerGElement<EvaModal1BruteForceNode>(&f, {a, b}, "eva_modal1_brute_force");
    status += pipeline->registerGElement<EvaModal2BruteForceNode>(&c, {a, b}, "eva_modal2_brute_force");
    status += pipeline->registerGElement<EvaMergeNode>(&d, {c, f}, "eva_merge");
    status += pipeline->registerGElement<SaveResultNode>(&g, {d}, "save_result");

    f->addGAspect<TimerAspect>()->addGAspect<TraceAspect>();
    c->addGAspect<TimerAspect>()->addGAspect<TraceAspect>();
    d->addGAspect<TimerAspect>()->addGAspect<TraceAspect>();
    status += pipeline->process();
    if (!status.isOK()) {
        CGRAPH_ECHO("process graph error, error info is [%s]", status.getInfo().c_str());
        exit(status.getCode());
    }
    GPipelineFactory::remove(pipeline);
    return 0;
}
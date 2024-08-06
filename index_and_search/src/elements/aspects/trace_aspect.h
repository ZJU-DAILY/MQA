//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_TRACE_ASPECT_H
#define INDEXING_AND_SEARCH_TRACE_ASPECT_H

#include "../elements_define.h"

class TraceAspect : public CGraph::GAspect {
public:
    CStatus beginInit() override {
        printf("[EXEC] %s init begin ...\n", this->getName().c_str());
        return CStatus();
    }

    CVoid finishInit(const CStatus& curStatus) override {
        printf("[EXEC] %s init finished, error code is %d ...\n", this->getName().c_str(),
               curStatus.getCode());
    }

    CStatus beginRun() override {
        printf("[EXEC] %s run begin ...\n", this->getName().c_str());
        return CStatus();
    }

    CVoid finishRun(const CStatus& curStatus) override {
        if (!curStatus.isOK()) {
            printf("[EXEC] %s run finished, status is ok ...\n", this->getName().c_str());
        } else {
            printf("[EXEC] %s run finished, error code is %d ...\n", this->getName().c_str(),
                   curStatus.getCode());
        }
    }

    CStatus beginDestroy() override {
        printf("[EXEC] %s destroy begin ...\n", this->getName().c_str());
        return CStatus();
    }

    CVoid finishDestroy(const CStatus& curStatus) override {
        printf("[EXEC] %s destroy finished, error code is %d ...\n", this->getName().c_str(),
               curStatus.getCode());
    }
};

#endif //INDEXING_AND_SEARCH_TRACE_ASPECT_H

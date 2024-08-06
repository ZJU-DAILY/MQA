//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_TIMER_ASPECT_H
#define INDEXING_AND_SEARCH_TIMER_ASPECT_H

#include <chrono>
#include "../elements_define.h"

class TimerAspect : public CGraph::GAspect {
public:
    /**
     * record time cost of run function
     */
    CStatus beginRun() override {
        start_ts_ = std::chrono::high_resolution_clock::now();
        return CStatus();
    }

    CVoid finishRun(const CStatus& curStatus) override {
        std::chrono::duration<double, std::milli> span = std::chrono::high_resolution_clock::now() - start_ts_;
        printf("[TIME] %s time cost is : %0.2lf ms\n",
               this->getName().c_str(), span.count());
    }

private:
    std::chrono::high_resolution_clock::time_point start_ts_;
};

#endif //INDEXING_AND_SEARCH_TIMER_ASPECT_H

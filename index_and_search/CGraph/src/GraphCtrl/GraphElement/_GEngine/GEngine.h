/***************************
@Author: Chunel
@Contact: chunel@foxmail.com
@File: GEngine.h
@Time: 2022/12/11 16:27
@Desc: 
***************************/

#ifndef CGRAPH_GENGINE_H
#define CGRAPH_GENGINE_H

#include "GEngineObject.h"
#include "../GElementSorter.h"

CGRAPH_NAMESPACE_BEGIN

class GEngine : public GEngineObject {
protected:
    explicit GEngine() = default;

    /**
     * 将所有注册进入 pipeline的内容，进行分析和解构
     * @param elements
     * @return
     */
    virtual CStatus setup(const GSortedGElementPtrSet& elements) = 0;

    /**
     * 执行完毕后，确认运行是否正常
     * @return
     */
    virtual CStatus afterRunCheck() = 0;

    /**
     * 计算出来最终计算的index值
     * @param element
     * @return
     */
    CIndex calcIndex(GElementPtr element) const {
        /**
         * 如果没有设定绑定线程的话，就用默认调度策略
         * 否则的话，会走绑定的thread。
         * 如果设定的 binding_index_ >= thread 总数，会在 threadpool 层做统一判定
         */
        auto bindingIndex = element->getBindingIndex();
        return CGRAPH_DEFAULT_BINDING_INDEX == bindingIndex
               ? schedule_strategy_ : bindingIndex;
    }

    /**
     * 分析所有的可以设置 linkable 的数据
     * @param elements
     * @return
     */
    CVoid link(const GSortedGElementPtrSet& elements) {
        /**
         * 认定图可以连通的判定条件：
         * 1，当前元素仅有一个依赖
         * 2，当前元素依赖的节点，只有一个后继
         * 3，当前元素的依赖的后继，仍是当前节点
         * 4，前后元素绑定机制是一样的
         */
        for (GElementPtr element : elements) {
            element->linkable_ = false;    // 防止出现之前的留存逻辑。确保只有当前链接关系下，需要设置 linkable的，才会设置为 true
            if (1 == element->dependence_.size()
                && 1 == (*element->dependence_.begin())->run_before_.size()
                && (*(element->dependence_.begin()))->run_before_.find(element) != (*(element->dependence_.begin()))->run_before_.end()
                && element->getBindingIndex() == (*(element->dependence_.begin()))->getBindingIndex()) {
                element->linkable_ = true;
            }
        }
    }


protected:
    UThreadPoolPtr thread_pool_ { nullptr };                    // 内部执行的线程池
    int schedule_strategy_ = CGRAPH_DEFAULT_TASK_STRATEGY;      // 调度策略

    friend class GElementManager;
    friend class GPipeline;
    friend class GMutable;
};

using GEnginePtr = GEngine *;

CGRAPH_NAMESPACE_END

#endif //CGRAPH_GENGINE_H

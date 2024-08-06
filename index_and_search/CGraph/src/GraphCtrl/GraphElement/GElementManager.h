/***************************
@Author: Chunel
@Contact: chunel@foxmail.com
@File: GElementManager.h
@Time: 2021/6/2 10:33 下午
@Desc:
***************************/

#ifndef CGRAPH_GELEMENTMANAGER_H
#define CGRAPH_GELEMENTMANAGER_H

#include "GElement.h"
#include "GElementSorter.h"
#include "_GEngine/GEngineInclude.h"

CGRAPH_NAMESPACE_BEGIN

class GElementManager : public GElementObject,
                        public GraphManager<GElement> {
protected:
    explicit GElementManager() = default;
    ~GElementManager() override;
    CGRAPH_NO_ALLOWED_COPY(GElementManager);

    CStatus init() final;
    CStatus run() final;
    CStatus destroy() final;

    /**
     * 添加一个元素信息
     * @param element
     * @return
     */
    CStatus add(GElementPtr element) final;

    /**
     * 删除传入元素
     * @param element
     */
    CStatus remove(GElementPtr element) final;

    /**
     * 查找传入元素
     * @param element
     * @return
     */
    CBool find(GElementPtr element) const final;

    /**
     * 清空信息
     * @return
     */
    CStatus clear() final;

    /**
     * 设置调度策略
     * @param type
     * @return
     */
    GElementManager* setScheduleStrategy(int strategy);

    /**
     * 设置图引擎运行模式
     * @param engineType
     * @return
    */
    GElementManager* setEngineType(GEngineType engineType);

    /**
     * 构造执行引擎
     * @param strategy
     * @return
     */
    CStatus initEngine();

    /**
     * 设置线程池
     * @param ptr
     * @return
     */
    GElementManager* setThreadPool(UThreadPoolPtr ptr);

    /**
     * 获取最大的并发数
     * @param size
     * @return
     */
    CStatus calcMaxParaSize(CSize& size);

    /**
     * 查看是否可以串行执行
     * @return
     */
    CBool checkSerializable();

    /**
     * 加入数据，并且执行
     * @param elements
     * @return
     */
    CStatus process(const GSortedGElementPtrSet& elements);

private:
    GSortedGElementPtrSet manager_elements_;                    // 保存节点信息的内容
    GEnginePtr engine_ { nullptr };                             // 执行引擎
    GEngineType engine_type_ { GEngineType::DYNAMIC };          // 引擎执行方式
    UThreadPoolPtr thread_pool_ { nullptr };                    // 线程池
    CBool auto_check_enable_ = true;                            // 是否自动实现后校验逻辑

    friend class GPipeline;
    friend class GRegion;
    friend class GMutable;
    friend class UAllocator;
};

using GElementManagerPtr = GElementManager *;

CGRAPH_NAMESPACE_END

#endif //CGRAPH_GELEMENTMANAGER_H

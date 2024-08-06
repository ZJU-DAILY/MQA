/***************************
@Author: Chunel
@Contact: chunel@foxmail.com
@File: GRegion.h
@Time: 2021/6/1 10:14 下午
@Desc: 实现多个element，根据依赖关系执行的功能
***************************/


#ifndef CGRAPH_GREGION_H
#define CGRAPH_GREGION_H

#include "../GGroup.h"
#include "../../GElementManager.h"

CGRAPH_NAMESPACE_BEGIN

class GRegion : public GGroup {
public:
    /**
     * 设置EngineType信息
     * @param type
     * @return
     */
    GRegion* setGEngineType(GEngineType type);

protected:
    explicit GRegion();
    ~GRegion() override;

    CStatus init() final;
    CStatus run() final;
    CStatus destroy() final;

private:
    CStatus addElement(GElementPtr element) final;

    CVoid dump(std::ostream& oss) final;

    CBool isSerializable() const final;

    CStatus addManagers(GParamManagerPtr paramManager,
                        GEventManagerPtr eventManager) final;

private:
    GElementManagerPtr manager_ = nullptr;    // region 内部通过 manager来管理其中的 element 信息

    CGRAPH_NO_ALLOWED_COPY(GRegion)

    friend class GPipeline;
    friend class UAllocator;
};

using GRegionPtr = GRegion *;

CGRAPH_NAMESPACE_END

#endif //CGRAPH_GREGION_H

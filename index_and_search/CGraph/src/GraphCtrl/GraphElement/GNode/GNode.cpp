/***************************
@Author: Chunel
@Contact: chunel@foxmail.com
@File: GNode.cpp
@Time: 2022/3/28 23:14
@Desc: 
***************************/

#include "GNode.h"

CGRAPH_NAMESPACE_BEGIN

GNode::GNode() : GElement() {
    element_type_ = GElementType::NODE;
    session_ = URandom<>::generateSession(CGRAPH_STR_NODE);
}


GNodePtr GNode::setType(const GNodeType& type) {
    node_type_ = type;
    return this;
}


CStatus GNode::spawn(const UTaskGroup& tasks, CMSec ttl) {
    CGRAPH_FUNCTION_BEGIN
    CGRAPH_ASSERT_INIT(true)
    CGRAPH_ASSERT_NOT_NULL(thread_pool_)

    status = thread_pool_->submit(tasks, ttl);
    CGRAPH_FUNCTION_END
}

CGRAPH_NAMESPACE_END

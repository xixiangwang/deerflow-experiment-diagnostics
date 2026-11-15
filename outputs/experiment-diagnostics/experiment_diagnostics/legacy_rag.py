"""Adapted from the user's rag-project; no imports of its config or secrets.
Original data.py credits ByteTora/hello-rag and Sam5440/RAG_Python_Demo.
Only pure splitting/routing functions are migrated; embedding/reranking are not.
"""

import re

SEMANTIC_MAX_CHARS = 1000

_SENT_END = re.compile(r"(?<=[。！？；])")

def _split_by_sentence(paragraph: str) -> list:
    """把一段话按中文句号/问号/叹号/分号切开。"""
    sents = _SENT_END.split(paragraph)
    return [s.strip() for s in sents if s.strip()]

def chunk_semantic(text: str, source: str) -> list:
    """
    语义切分：三级降级策略。

    第 1 级：按空行（段落）切 → 每段是一个天然的语义单元
    第 2 级：段落超过 SEMANTIC_MAX_CHARS，按句子切，然后把句子攒到接近上限
    第 3 级：单句还是超长（比如一整段没有标点），才用硬切兜底

    这样切出来的每一块，都是"一整段话"或"几句话"，语义完整。
    """
    chunks = []

    # --- 第 1 级：按段落切 ---
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    for para in paragraphs:
        if len(para) <= SEMANTIC_MAX_CHARS:
            # 段落本身够短，直接作为一块
            chunks.append(para)
            continue

        # --- 第 2 级：段落太长，按句子攒 ---
        buffer = ""
        for sent in _split_by_sentence(para):
            if len(sent) > SEMANTIC_MAX_CHARS:
                # --- 第 3 级：单句超长，硬切兜底 ---
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                for i in range(0, len(sent), SEMANTIC_MAX_CHARS):
                    chunks.append(sent[i:i + SEMANTIC_MAX_CHARS])
                continue

            if len(buffer) + len(sent) <= SEMANTIC_MAX_CHARS:
                buffer += sent
            else:
                chunks.append(buffer)
                buffer = sent

        if buffer:
            chunks.append(buffer)

    # 打包成统一格式
    result = []
    for c in chunks:
        if len(c) < 20:
            continue
        result.append({
            "text": c,
            "source": source,
            "chunk_id": len(result),
            "mode": "semantic",
        })

    return result

_CHITCHAT_WORDS = [
    "你好", "您好", "谢谢", "感谢", "再见", "拜拜",
    "你是谁", "你叫什么", "你能做什么", "介绍一下你自己",
    "早上好", "晚上好", "在吗", "辛苦",
    # 2026-10-23：原来写的是"辛苦了"，漏掉"辛苦啦"，实测这条被判成要检索。
    # 改成"辛苦"前缀，覆盖 辛苦了 / 辛苦啦 / 辛苦你了。
]

_KNOWLEDGE_HINTS = [
    "怎么", "如何", "什么", "哪些", "多少", "为什么",
    "支持", "区别", "参数", "配置", "接口", "调用",
    "价格", "计费", "限制", "要求", "文档", "说明",
    "模型", "向量", "部署", "报错", "失败",
    # 2026-10-23 补：这几类词原来不在表里，
    # 导致「语义切分分几级」这种短问题既没命中疑问词、又没超过 10 字，
    # 被兜底规则判成"不检索"，直接漏检。
    "几级", "几层", "几步", "几种", "种类", "流程", "步骤",
    "原理", "机制", "策略", "维度", "优点", "缺点", "局限",
    "切分", "切块", "重排", "召回", "检索", "索引", "评测",
]

def route_by_rule(query: str) -> bool:
    """
    规则判断：返回 True 表示"需要检索"，False 表示"直接回答"。

    逻辑：先看是不是纯寒暄；再看有没有知识类疑问词；最后默认检索。

    ⚠️ 兜底一定要是 True。
        原来的写法是 `return len(q) > 10`，注释却写着"宁可多查，不可漏查"——
        **代码和注释是反的**：短问题反而被判成不检索。
        「语义切分分几级」只有 7 个字，正好被这条兜底漏掉。
        规则路由的原则就该是"宁可多查"：多查一次只是多花点时间，
        漏查一次就是直接答错。
    """
    q = query.strip()

    # 寒暄词命中且句子很短 → 认定为闲聊，不检索
    has_chitchat = any(w in q for w in _CHITCHAT_WORDS)
    if has_chitchat and len(q) <= 8:
        return False

    # 含有知识类疑问词/主题词 → 检索
    if any(w in q for w in _KNOWLEDGE_HINTS):
        return True

    # 兜底：默认检索（宁可多查，不可漏查）
    return True

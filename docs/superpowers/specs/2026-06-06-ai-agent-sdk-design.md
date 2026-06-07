# AI Agent SDK 设计规范

## 项目概述

**项目名称**: AI Agent SDK  
**目标**: 打造一个便捷、本地优先、渐进式复杂度的 AI 智能体开发 SDK

**核心理念**:
- 本地优先：核心功能无外部依赖
- 渐进式复杂度：简单任务简单用，复杂任务有能力深入
- 厂商无关：不绑定特定厂商，支持多模型切换
- 容错降级：自动重试、备用方案、错误恢复

---

## 目标用户

- AI 应用开发者：构建聊天机器人、助手等
- AI Agent 开发者：构建自主 Agent 系统
- 垂直领域开发者：企业知识库、批阅评审、文档生成等

---

## 架构设计

### 分层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         业务应用层                              │
│  KnowledgeQA │ Reviewer │ DocGen │ StructuredOutput │ Dashboard│
├─────────────────────────────────────────────────────────────────┤
│                         Agent 编排层                            │
│  SingleAgent │ Workflow │ MultiAgent │ HumanLoop               │
├─────────────────────────────────────────────────────────────────┤
│                         核心能力层                              │
│  LLMEngine │ VectorEngine │ MemoryEngine │ ToolEngine          │
├─────────────────────────────────────────────────────────────────┤
│                         数据处理层                              │
│  DocParser │ TextSplitter │ DataCleaner │ FormatConverter      │
├─────────────────────────────────────────────────────────────────┤
│                         安全合规层                              │
│  Permission │ SensitiveFilter │ AuditLog │ Compliance         │
├─────────────────────────────────────────────────────────────────┤
│                         运维监控层                              │
│  Performance │ CostTracker │ ErrorTracker │ Versioning        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 功能优先级

### P0 - 必须有（MVP）

| 模块 | 功能 | 描述 |
|------|------|------|
| **KnowledgeQA** | 知识库问答 | 本地向量库 + 文档解析 + 来源追溯 |
| **StructuredOutput** | 结构化输出 | JSON Schema 校验 + 自动修正 |
| **LocalLLM** | 本地模型 | Ollama 支持 + 自动检测 |
| **Streaming** | 流式输出 | 实时反馈 + 中断恢复 |

### P1 - 重要

| 模块 | 功能 | 描述 |
|------|------|------|
| **DocParser** | 文档解析 | PDF/Word/Excel/PPT/Markdown |
| **SessionMemory** | 会话记忆 | 上下文保持 + 历史对话 |
| **Reviewer** | 批阅评审 | 评分 + 批语 + 建议 |
| **DocGen** | 文档生成 | Markdown/HTML/PDF 报告 |

### P2 - 可选

| 模块 | 功能 | 描述 |
|------|------|------|
| **Workflow** | 工作流编排 | 步骤 + 分支 + 循环 |
| **Dashboard** | 可视化表盘 | 图表 + 仪表盘 + 报告导出 |
| **MultiAgent** | 多Agent协作 | 角色分工 + 协作通信 |
| **HumanLoop** | 人机协作 | 人工审核 + 确认点 |

---

## 使用示例

### P0 - 知识库问答

```python
from vertai import KnowledgeQA

# 10行代码完成企业知识库
qa = KnowledgeQA(
    docs_path="./企业文档",
    model="local",  # 本地模型
)

answer = qa.ask("公司的报销流程是什么？")
# 返回：
# {
#   "answer": "报销流程是...",
#   "sources": [{"file": "财务手册.pdf", "page": 15}],
#   "confidence": 0.92
# }
```

### P0 - 结构化输出

```python
from vertai import StructuredOutput

schema = {
    "name": "string",
    "amount": "number",
    "category": "enum[报销,采购,其他]",
}

result = StructuredOutput(schema).extract("张三报销500元会议费用")
# 返回：{"name": "张三", "amount": 500, "category": "报销"}
```

### P1 - 批阅评审

```python
from vertai import Reviewer

reviewer = Reviewer(
    criteria=["准确性", "完整性", "格式规范"],
    template="评分 + 批语 + 建议",
)

result = reviewer.evaluate(submission)
# 返回：
# {
#   "score": 85,
#   "criteria_scores": {"准确性": 90, "完整性": 80},
#   "comments": "整体完成度高...",
#   "suggestions": ["增加案例说明", "修正格式"]
# }
```

---

## 技术选型

### 本地向量库
- **ChromaDB**: 轻量级，纯 Python，适合小规模
- **FAISS**: 高性能，适合大规模
- **Qdrant**: 功能丰富，可本地部署

### 本地模型
- **Ollama**: 一键部署本地模型
- **llama.cpp**: 高性能推理
- **vLLM**: 生产级推理服务

### 文档解析
- **PyMuPDF**: PDF 解析
- **python-docx**: Word 解析
- **openpyxl**: Excel 解析

---

## 目录结构

```
ai-sdk/
├── ai_sdk/
│   ├── __init__.py
│   ├── core/
│   │   ├── llm.py          # 本地模型引擎
│   │   ├── vector.py       # 向量引擎
│   │   ├── memory.py       # 记忆引擎
│   │   └── tools.py        # 工具引擎
│   ├── data/
│   │   ├── parser.py       # 文档解析
│   │   ├── splitter.py     # 文本切分
│   │   └── cleaner.py      # 数据清洗
│   ├── output/
│   │   ├── structured.py   # 结构化输出
│   │   ├── docgen.py       # 文档生成
│   │   ├── dashboard.py    # 可视化
│   ├── scenarios/
│   │   ├── knowledge_qa.py # 知识问答
│   │   ├── reviewer.py     # 批阅评审
│   │   └── workflow.py     # 工作流
│   ├── security/
│   │   ├── permission.py
│   │   ├── filter.py
│   │   ├── audit.py
│   └── utils/
│       ├── retry.py        # 容错重试
│       ├── fallback.py     # 降级备用
│       └── monitor.py      # 监控追踪
├── tests/
├── docs/
├── examples/
├── pyproject.toml
└── README.md
```

---

## 质量要求

- 测试覆盖率 ≥ 80%
- 类型注解完整
- 文档字符串规范
- 无硬编码默认值
- 所有异常友好处理
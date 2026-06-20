"""快速开发示例：构建一个企业知识库问答系统 | Quick Start Example: Building an Enterprise Knowledge QA System

展示如何使用 vertai 快速开发包含垂直领域需要的所有功能的产品。
Demonstrates how to use vertai to quickly develop products with all features needed for vertical domains.

运行 | Run: python examples/quick_start_knowledge_base.py
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from vertai import (
    # LLM - 智能对话 | LLM - Intelligent Chat
    LLMEngine, LLMConfig, ModelProvider,

    # Vector - 知识库搜索 | Vector - Knowledge Base Search
    VectorEngine, Document, VectorConfig,

    # KnowledgeQA - 知识库问答 | KnowledgeQA - Knowledge Base Q&A
    KnowledgeQA, KnowledgeQAConfig,

    # Workflow - 自动化流程 | Workflow - Automation Process
    Workflow, WorkflowContext,

    # StructuredOutput - 数据提取 | StructuredOutput - Data Extraction
    StructuredOutput,

    # DocParser - 文档解析 | DocParser - Document Parsing
    DocParser,

    # DocGen - 报告生成 | DocGen - Report Generation
    DocGen,

    # LocalModelManager - 本地模型 | LocalModelManager - Local Models
    LocalModelManager, check_hardware_requirements,

    # SessionMemory - 会话记忆 | SessionMemory - Session Memory
    SessionMemory, SessionConfig,
)


def example_1_basic_llm_chat():
    """示例1: 基础 LLM 对话 | Example 1: Basic LLM Chat"""
    print("\n" + "="*60)
    print("示例1: 基础 LLM 对话 | Example 1: Basic LLM Chat")
    print("="*60)

    # 配置 LLM (支持 DeepSeek/Anthropic/OpenAI) | Configure LLM (supports DeepSeek/Anthropic/OpenAI)
    config = LLMConfig(
        provider=ModelProvider.DEEPSEEK,
        base_url="https://api.deepseek.com/anthropic",  # 必须指定! | Must specify!
        api_key="sk-xxx",  # 或设置环境变量 VERTAI_API_KEY | Or set environment variable VERTAI_API_KEY
        model="deepseek-chat",
    )

    # 快速使用 | Quick usage
    llm = LLMEngine(config)

    # 单次生成 | Single generation
    result = llm.generate("你好，请介绍一下你自己")
    print(f"回复: {result.content[:100]}... | Response: {result.content[:100]}...")

    # 流式输出 | Stream output
    print("\n流式输出 | Stream output: ")
    for chunk in llm.stream("讲一个短笑话 | Tell a short joke"):
        print(chunk, end="", flush=True)
    print()

    # 多轮对话 | Multi-turn conversation
    messages = [
        {"role": "user", "content": "我叫小明 | My name is Xiao Ming"},
        {"role": "assistant", "content": "你好小明！ | Hello Xiao Ming!"},
        {"role": "user", "content": "我叫什么名字？ | What's my name?"},
    ]
    result = llm.chat(messages)
    print(f"\n多轮对话: {result.content} | Multi-turn chat: {result.content}")


def example_2_knowledge_qa():
    """示例2: 知识库问答（RAG）| Example 2: Knowledge Base Q&A (RAG)"""
    print("\n" + "="*60)
    print("示例2: 知识库问答 | Example 2: Knowledge Base Q&A")
    print("="*60)

    # 知识库需要一个 EmbeddingProvider 来把文本转成向量。
    # 真实语义搜索请安装 vertai[embeddings] 并使用 LocalSentenceTransformerProvider。
    # 这里用一个简单的确定性函数演示（不具备语义相似性，仅演示 API）。
    #
    # A knowledge base needs an EmbeddingProvider to vectorize text. For real
    # semantic search install vertai[embeddings] and use
    # LocalSentenceTransformerProvider. Here we use a simple deterministic
    # function for demonstration (NOT semantically meaningful).
    from vertai.core.embedding import FunctionEmbeddingProvider

    def demo_embedding_fn(text: str) -> list[float]:
        # 演示用：基于字符的确定性向量（无语义）。| Demo: deterministic char-based vector (non-semantic).
        vec = [0.0] * 32
        for ch in text:
            vec[ord(ch) % 32] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec

    embedding_provider = FunctionEmbeddingProvider(demo_embedding_fn, dimension=32)

    # 创建知识库 (索引与检索完全本地，无需 LLM) | Create knowledge base
    # (indexing + retrieval are fully local, no LLM needed)
    qa = KnowledgeQA(embedding_provider=embedding_provider)

    # 添加文档 | Add documents
    docs = [
        Document(content="公司报销流程：员工需要在系统中提交报销申请，附上发票原件，经过部门经理审批后，财务部门会在3个工作日内处理。"),
        Document(content="请假制度：年假每年15天，病假需要医院证明，事假需要提前申请并获得批准。"),
        Document(content="加班规定：工作日加班需要提前申请，周末加班按双倍工资计算。"),
    ]
    qa.add_documents(docs)
    print(f"已索引 {qa.count_documents()} 个文档 | Indexed {qa.count_documents()} documents")

    # 搜索知识库 (向量检索，完全本地) | Search knowledge base (vector retrieval, fully local)
    print("\n知识库搜索 | Knowledge base search:")
    results = qa._get_retriever().retrieve("报销流程", top_k=2)
    for r in results[:2]:
        print(f"  - 相似度: {r.score:.2f} | Similarity: {r.score:.2f}")
        print(f"    内容: {r.document.content[:50]}... | Content: {r.document.content[:50]}...")

    # 问答 (需要 LLM 生成) | Q&A (requires an LLM provider for generation)
    # from vertai.core.provider import LLMConfig, create_provider
    # qa_with_llm = KnowledgeQA(
    #     embedding_provider=embedding_provider,
    #     provider=create_provider(LLMConfig()),  # 默认 Ollama | default Ollama
    # )
    # qa_with_llm.add_documents(docs)
    # answer = qa_with_llm.ask("报销需要什么材料？")
    # print(f"\n问答结果: {answer.answer} | Q&A result: {answer.answer}")


def example_3_structured_extraction():
    """示例3: 结构化数据提取 | Example 3: Structured Data Extraction"""
    print("\n" + "="*60)
    print("示例3: 结构化数据提取 | Example 3: Structured Data Extraction")
    print("="*60)

    # 本地模式 (正则提取，无需 LLM) | Local mode (regex extraction, no LLM needed)
    schema = {"name": "string", "amount": "number"}
    output = StructuredOutput(schema)

    result = output.extract("张三报销500元")
    print(f"本地提取: {result.data} | Local extraction: {result.data}")

    # LLM 模式 (语义提取，更准确) | LLM mode (semantic extraction, more accurate)
    # llm = LLMEngine(config)
    # output = StructuredOutput(schema, llm=llm)
    # result = output.extract("李四昨天消费了三百块买咖啡")
    # print(f"LLM提取: {result.data} | LLM extraction: {result.data}")


def example_4_workflow():
    """示例4: 工作流编排 | Example 4: Workflow Orchestration"""
    print("\n" + "="*60)
    print("示例4: 工作流编排 | Example 4: Workflow Orchestration")
    print("="*60)

    wf = Workflow()

    # 定义步骤 | Define steps
    wf.step("input", lambda ctx: ctx.set("data", 100))
    wf.step("process", lambda ctx: ctx.set("result", ctx.get("data") * 2))
    wf.step("output", lambda ctx: print(f"结果: {ctx.get('result')} | Result: {ctx.get('result')}"))

    # 运行 | Run
    wf.run()

    # 分支示例 | Branch example
    wf2 = Workflow()
    wf2.step("init", lambda ctx: ctx.set("value", 10))
    wf2.branch(
        condition=lambda ctx: ctx.get("value") > 5,
        yes_steps=[("yes", lambda ctx: print("值大于5 | Value greater than 5"))],
        no_steps=[("no", lambda ctx: print("值小于等于5 | Value less than or equal to 5"))],
    )
    wf2.run()

    # 并行示例 | Parallel example
    wf3 = Workflow()
    wf3.parallel(steps=[
        ("a", lambda ctx: ctx.set("a", 1)),
        ("b", lambda ctx: ctx.set("b", 2)),
        ("c", lambda ctx: ctx.set("c", 3)),
    ])
    wf3.step("sum", lambda ctx: print(f"并行结果: a={ctx.get('a')}, b={ctx.get('b')}, c={ctx.get('c')} | Parallel result: a={ctx.get('a')}, b={ctx.get('b')}, c={ctx.get('c')}"))
    wf3.run()


def example_5_document_processing():
    """示例5: 文档解析与生成 | Example 5: Document Parsing and Generation"""
    print("\n" + "="*60)
    print("示例5: 文档解析与生成 | Example 5: Document Parsing and Generation")
    print("="*60)

    # 解析文档 (支持 Markdown/PDF/Word/Excel) | Parse documents (supports Markdown/PDF/Word/Excel)
    parser = DocParser()

    # 需要实际文件路径 | Requires actual file path
    # result = parser.parse("report.pdf")
    # print(f"解析结果: {result['text'][:50]}... | Parse result: {result['text'][:50]}...")

    print("文档解析需要实际文件路径，支持 | Document parsing requires actual file path, supports: PDF, Word, Excel, PPT, Markdown")

    # 生成文档 | Generate document
    docgen = DocGen(template="report")
    report = docgen.generate({
        "title": "Q4 季度报告 | Q4 Quarterly Report",
        "author": "AI团队 | AI Team",
        "summary": "本季度完成了多个重要项目。 | Multiple important projects completed this quarter.",
    })
    print(f"\n生成的报告 | Generated report:\n{report[:200]}...")


def example_6_dashboard():
    """示例6: 数据可视化仪表盘 | Example 6: Data Visualization Dashboard

    Dashboard moved to the optional ``vertai[viz]`` extra. Install with
    ``pip install vertai[viz]`` to run this example.
    """
    print("\n" + "="*60)
    print("示例6: 数据可视化仪表盘 | Example 6: Data Visualization Dashboard")
    print("="*60)

    try:
        from vertai.viz.dashboard import Dashboard, ChartType  # noqa: F401
    except ImportError:
        print("Dashboard 需要可选依赖 | Dashboard requires the optional extra: pip install vertai[viz]")
        return

    # 创建仪表盘 | Create dashboard
    dashboard = Dashboard(title="业务监控 | Business Monitoring")

    # 添加指标 (threshold 是单个数值) | Add metrics (threshold is a single value)
    dashboard.add_metric("日活用户 | DAU", 12500, unit="人")
    dashboard.add_metric("转化率 | Conversion Rate", 15.8, format_spec=".1f", unit="%")
    dashboard.add_metric("响应时间 | Response Time", 120, unit="ms")

    # 添加图表 (注意参数顺序: title, data, chart_type) | Add charts (note parameter order: title, data, chart_type)
    dashboard.add_chart(
        title="用户增长 | User Growth",
        data=[100, 120, 150, 180, 220],
        chart_type=ChartType.LINE,
    )
    dashboard.add_chart(
        title="来源分布 | Source Distribution",
        data=[40, 35, 25],
        chart_type=ChartType.PIE,
        labels=["搜索引擎 | Search Engine", "社交媒体 | Social Media", "直接访问 | Direct"],
    )

    # 导出 HTML (可直接嵌入前端) | Export HTML (can be embedded in frontend)
    html = dashboard.show()
    print(f"仪表盘 HTML 已生成，长度 | Dashboard HTML generated, length: {len(html)} 字符 | characters")

    # 导出 JSON (可发送给前端) | Export JSON (can be sent to frontend)
    json_data = dashboard.to_json()
    print(f"仪表盘 JSON | Dashboard JSON: {json_data[:100]}...")


def example_7_session_memory():
    """示例7: 会话记忆 | Example 7: Session Memory"""
    print("\n" + "="*60)
    print("示例7: 会话记忆 | Example 7: Session Memory")
    print("="*60)

    # 创建会话 (session_id 直接传给 SessionMemory) | Create session (session_id passed directly to SessionMemory)
    memory = SessionMemory(session_id="user_123")

    # 记录对话 | Record conversation
    memory.add_message("user", "我想查询我的订单")
    memory.add_message("assistant", "好的，请提供订单号")
    memory.add_message("user", "订单号是 202401001")

    # 获取历史 | Get history
    history = memory.get_history()
    print(f"对话历史 | Conversation history: {len(history)} 条消息 | messages")

    # 保存到文件 | Save to file
    memory.save("session_user_123.json")
    print("会话已保存 | Session saved")


def example_8_local_models():
    """示例8: 本地小模型 | Example 8: Local Small Models"""
    print("\n" + "="*60)
    print("示例8: 本地小模型（语音转文字/向量嵌入）| Example 8: Local Small Models (Speech-to-Text/Vector Embedding)")
    print("="*60)

    # 检查硬件需求 | Check hardware requirements
    hw = check_hardware_requirements("whisper-small")
    print(f"硬件检查 | Hardware check:")
    print(f"  当前内存 | Current RAM: {hw.get('current_ram_gb', 'N/A')} GB")
    print(f"  需要内存 | Required RAM: {hw.get('required_ram_gb', 'N/A')} GB")
    print(f"  是否满足 | Satisfied: {hw.get('satisfied', 'N/A')}")

    # 列出可用模型 | List available models
    manager = LocalModelManager()
    models = manager.list_models()
    print(f"\n可用本地模型 | Available local models: {len(models)} 个 | units")

    for m in models[:3]:
        print(f"  - {m.name}: {m.network.download_size_mb}MB | Size")
        print(f"    类别 | Category: {m.category.value}")
        print(f"    语言 | Languages: {', '.join(m.languages[:3])}")


def example_9_full_application():
    """示例9: 完整应用 - 企业智能助手 | Example 9: Full Application - Enterprise Intelligent Assistant"""
    print("\n" + "="*60)
    print("示例9: 完整应用 - 企业智能助手 | Example 9: Full Application - Enterprise Intelligent Assistant")
    print("="*60)

    print("""
完整企业智能助手架构 | Complete Enterprise Intelligent Assistant Architecture:

┌─────────────────────────────────────────────────────────────┐
│                      前端 (React/Vue) | Frontend             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  对话UI  │  │ 知识库   │  │ 仪表盘   │  │ 报告页   │    │
│  │ Chat UI  │  │ Knowledge│  │ Dashboard│  │ Reports  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                           ↓ API 调用 | API calls
┌─────────────────────────────────────────────────────────────┐
│                    后端 (FastAPI/Flask) | Backend             │
│                                                              │
│  from vertai import *                                        │
│                                                              │
│  # 1. LLM 对话 | LLM Chat                                     │
│  llm = LLMEngine(config)                                     │
│  result = llm.chat(messages)                                 │
│                                                              │
│  # 2. 知识库问答 | Knowledge Base Q&A                          │
│  qa = KnowledgeQA("./docs")                                  │
│  answer = qa.ask(question)                                   │
│                                                              │
│  # 3. 数据提取 | Data Extraction                              │
│  data = StructuredOutput(schema, llm=llm).extract(text)     │
│                                                              │
│  # 4. 报告生成 | Report Generation                            │
│  report = DocGen("report").generate(data)                    │
│                                                              │
│  # 5. 仪表盘数据 | Dashboard Data                              │
│  dashboard = Dashboard().add_metric(...).to_json()          │
│                                                              │
│  # 6. 工作流自动化 | Workflow Automation                        │
│  wf = Workflow().step(...).run()                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                           ↓ 可选本地部署 | Optional local deployment
┌─────────────────────────────────────────────────────────────┐
│                    本地模型 (可选) | Local Models (Optional)   │
│                                                              │
│  # 语音转文字 | Speech to Text                                 │
│  whisper = LocalModelManager().load("whisper-small")        │
│  text = whisper.transcribe("meeting.mp3")                    │
│                                                              │
│  # 向量嵌入 | Vector Embedding                                 │
│  embedder = LocalModelManager().load("bge-small-zh-v1.5")   │
│  vector = embedder.embed("文本")                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘

代码量对比 | Code Comparison:

┌──────────────────────────────────────────────────────────────┐
│  传统开发 | Traditional         │  使用 vertai | Using vertai │
├──────────────────────────────────────────────────────────────┤
│  LLM集成: ~500行              │  LLM集成: ~10行            │
│  向量搜索: ~300行             │  向量搜索: ~5行            │
│  知识库RAG: ~800行            │  知识库RAG: ~5行           │
│  数据提取: ~200行             │  数据提取: ~3行            │
│  工作流: ~400行               │  工作流: ~10行             │
│  报告生成: ~150行             │  报告生成: ~3行            │
│  仪表盘: ~300行               │  仪表盘: ~10行             │
├──────────────────────────────────────────────────────────────┤
│  总计: ~2650行                │  总计: ~46行               │
│  开发周期: 2-4周              │  开发周期: 1-2天           │
└──────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    print("="*60)
    print("VertAI 快速开发示例 | VertAI Quick Start Example")
    print("展示如何用 SDK 快速构建垂直领域应用 | Demonstrates how to use SDK to quickly build vertical applications")
    print("="*60)

    # 运行所有示例 | Run all examples
    try:
        example_2_knowledge_qa()
    except Exception as e:
        print(f"示例2 错误 | Example 2 error: {e}")

    try:
        example_3_structured_extraction()
    except Exception as e:
        print(f"示例3 错误 | Example 3 error: {e}")

    try:
        example_4_workflow()
    except Exception as e:
        print(f"示例4 错误 | Example 4 error: {e}")

    try:
        example_5_document_processing()
    except Exception as e:
        print(f"示例5 错误 | Example 5 error: {e}")

    try:
        example_6_dashboard()
    except Exception as e:
        print(f"示例6 错误 | Example 6 error: {e}")

    try:
        example_7_session_memory()
    except Exception as e:
        print(f"示例7 错误 | Example 7 error: {e}")

    try:
        example_8_local_models()
    except Exception as e:
        print(f"示例8 错误 | Example 8 error: {e}")

    example_9_full_application()

    print("\n" + "="*60)
    print("所有示例完成! | All examples completed!")
    print("="*60)

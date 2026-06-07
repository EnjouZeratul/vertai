"""知识库问答系统使用示例 | Knowledge QA System Usage Example

演示如何使用 KnowledgeQA 模块进行企业知识库问答。
Demonstrates how to use the KnowledgeQA module for enterprise knowledge base Q&A.
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import tempfile
import json

from vertai import KnowledgeQA, KnowledgeQAConfig
from vertai.core.vector import Document


def demo_basic_usage():
    """基本使用示例 | Basic Usage Example"""
    print("=" * 60)
    print("基本使用示例 | Basic Usage Example")
    print("=" * 60)

    # 创建临时目录和文档 | Create temporary directory and documents
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)

        # 创建示例文档 | Create sample documents
        (docs_dir / "python_intro.txt").write_text(
            "Python是一种高级编程语言，由Guido van Rossum于1991年创建。"
            "Python以简洁、易读的语法著称，广泛应用于Web开发、数据科学、"
            "人工智能、自动化脚本等领域。",
            encoding="utf-8",
        )

        (docs_dir / "machine_learning.txt").write_text(
            "机器学习是人工智能的一个分支，它使计算机能够从数据中学习，"
            "而无需显式编程。常见的机器学习算法包括：线性回归、决策树、"
            "支持向量机、神经网络等。",
            encoding="utf-8",
        )

        # 初始化知识库 | Initialize knowledge base
        qa = KnowledgeQA(docs_dir)

        print(f"已索引 {qa.count_documents()} 个文档片段 | Indexed {qa.count_documents()} document chunks")

        # 提问 | Ask questions
        questions = [
            "Python是什么？",
            "机器学习有哪些算法？",
        ]

        for question in questions:
            print(f"\n问题: {question} | Question: {question}")
            result = qa.ask(question)
            print(f"答案: {result.answer} | Answer: {result.answer}")
            print(f"置信度: {result.confidence:.2%} | Confidence: {result.confidence:.2%}")

            if result.sources:
                print("来源 | Sources:")
                for src in result.sources:
                    print(f"  - {src.source} (相关度: {src.relevance_score:.2f} | Relevance: {src.relevance_score:.2f})")


def demo_json_documents():
    """JSON 文档加载示例 | JSON Document Loading Example"""
    print("\n" + "=" * 60)
    print("JSON 文档加载示例 | JSON Document Loading Example")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)

        # 创建 JSON 格式的知识库 | Create JSON format knowledge base
        faq_data = [
            {
                "content": "公司的年假政策：员工入职满一年后可享受5天年假，"
                "满三年后可享受10天年假。",
                "category": "人事政策",
                "tags": ["年假", "假期"],
            },
            {
                "content": "报销流程：员工需要在费用发生后30天内提交报销申请，"
                "附上正规发票，经部门经理审批后提交财务部门。",
                "category": "财务流程",
                "tags": ["报销", "财务"],
            },
        ]

        json_file = docs_dir / "faq.json"
        json_file.write_text(json.dumps(faq_data, ensure_ascii=False), encoding="utf-8")

        # 加载并使用 | Load and use
        qa = KnowledgeQA(docs_dir)

        result = qa.ask("年假有多少天？")
        print(f"问题: 年假有多少天？ | Question: How many days of annual leave?")
        print(f"答案: {result.answer} | Answer: {result.answer}")


def demo_manual_documents():
    """手动添加文档示例 | Manual Document Addition Example"""
    print("\n" + "=" * 60)
    print("手动添加文档示例 | Manual Document Addition Example")
    print("=" * 60)

    qa = KnowledgeQA()

    # 手动添加文档 | Manually add documents
    documents = [
        Document(
            content="产品A是一款面向企业的协作工具，支持实时编辑和版本控制。",
            metadata={"source": "产品手册", "product": "A"},
        ),
        Document(
            content="产品B是一款数据分析平台，提供可视化报表和预测分析功能。",
            metadata={"source": "产品手册", "product": "B"},
        ),
    ]

    qa.add_documents(documents)

    result = qa.ask("产品A有什么功能？")
    print(f"问题: 产品A有什么功能？ | Question: What features does Product A have?")
    print(f"答案: {result.answer} | Answer: {result.answer}")


def demo_custom_config():
    """自定义配置示例 | Custom Configuration Example"""
    print("\n" + "=" * 60)
    print("自定义配置示例 | Custom Configuration Example")
    print("=" * 60)

    config = KnowledgeQAConfig(
        chunk_size=256,       # 较小的分块 | Smaller chunks
        top_k=3,              # 返回前3个结果 | Return top 3 results
        min_confidence=0.5,   # 较高的置信度阈值 | Higher confidence threshold
    )

    qa = KnowledgeQA(config=config)

    qa.add_documents([
        Document(content="这是测试文档"),
    ])

    print(f"配置: chunk_size={config.chunk_size}, top_k={config.top_k} | Config: chunk_size={config.chunk_size}, top_k={config.top_k}")


def demo_source_tracing():
    """来源追溯示例 | Source Tracing Example"""
    print("\n" + "=" * 60)
    print("来源追溯示例 | Source Tracing Example")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)

        # 创建带详细信息的文档 | Create documents with detailed information
        (docs_dir / "policy.txt").write_text(
            "第三章 休假制度\n\n"
            "第十五条 年假规定：员工年假天数根据工龄确定。\n"
            "- 工龄1-3年：5天年假\n"
            "- 工龄3-5年：10天年假\n"
            "- 工龄5年以上：15天年假\n",
            encoding="utf-8",
        )

        qa = KnowledgeQA(docs_dir)

        result = qa.ask("工龄5年以上有多少天年假？")

        print(f"问题: 工龄5年以上有多少天年假？ | Question: How many days of annual leave for 5+ years of service?")
        print(f"答案: {result.answer} | Answer: {result.answer}")
        print(f"置信度: {result.confidence:.2%} | Confidence: {result.confidence:.2%}")

        print("\n答案来源详情 | Answer Source Details:")
        for i, source in enumerate(result.sources, 1):
            print(f"\n来源 {i} | Source {i}:")
            print(f"  文件: {source.source} | File: {source.source}")
            print(f"  相关度: {source.relevance_score:.2f} | Relevance: {source.relevance_score:.2f}")
            print(f"  内容片段: {source.content[:100]}... | Content Snippet: {source.content[:100]}...")


if __name__ == "__main__":
    demo_basic_usage()
    demo_json_documents()
    demo_manual_documents()
    demo_custom_config()
    demo_source_tracing()

    print("\n" + "=" * 60)
    print("所有示例运行完成 | All examples completed")
    print("=" * 60)

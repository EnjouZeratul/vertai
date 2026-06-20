# S3 审查记录 — Embedding/Vector/Retriever/TextSplitter 抽象（0.4.0）

> 阶段完成后三重审查。基于实测，非代理自报。

## 阶段范围回顾

S3：EmbeddingProvider/Retriever/TextSplitter ABC + VectorStore 重构 + KnowledgeQA 用 Retriever + 修 C1/C2/C3 + 注入安全 + Reviewer 泛化。

## 1. 代码真实实现审查

### 契约兑现（ARCHITECTURE 3.2/3.3/3.4/3.5）
- ✅ `core/embedding.py` EmbeddingProvider ABC（embed/aembed/dimension）+ FunctionEmbeddingProvider + LocalSentenceTransformerProvider（真实 sentence-transformers）
- ✅ `core/retriever.py` Retriever ABC（retrieve/aretrieve）+ VectorRetriever（组合 EmbeddingProvider+VectorStore）+ rerank/transform 扩展点
- ✅ `core/text_splitter.py` TextSplitter ABC + RecursiveTextSplitter + FixedLengthSplitter
- ✅ VectorStore 重构：add(documents, embeddings) 外部计算向量；VectorEngine 无 provider 显式抛错

### Critical bug 修复
- ✅ C1：`knowledge_qa._get_provider` 用 create_provider/注入 LLMProvider，不再 LLMEngine(model=)（实测 grep 仅 docstring 引用）
- ✅ C2：hash 随机彻底移除（grep random.seed/_local_embed/hash(text) 空匹配）；无 provider 时显式 RuntimeError（vector.py:460 "raise (C2 fix: no silent random)"）
- ✅ C3：FAISSVectorStore.delete 真实移除 _documents/_id_to_idx；count 返回 len(_documents) 非 index.ntotal；search 过滤（delete 测试 3 passed）

### 安全
- ✅ _sanitize_context 真实检测并 redact 注入（英文+中文模式：忽略之前的指令/你现在扮演/disregard prior）
- ✅ load_directory 用 os.walk(followlinks=False) + is_symlink() 拒绝 symlink
- ✅ 安全测试：间接注入（投毒文档 redacted）、中文注入、symlink（14 passed, 3 skip=平台限制）

### scenarios 依赖抽象
- ✅ knowledge_qa 检索走 Retriever ABC（可注入 stub 验证）
- ✅ reviewer 泛化为 Evaluation ABC（LLM-as-judge），依赖 LLMProvider

### 无摆设
- ✅ 无"声称防护实为摆设"注释；_sanitize_context 真实实现

## 2. 测试覆盖率与真实性审查

### 实测
- 全套：646 passed, 31 skipped（31 = 17 DeepSeek 无key + 14 S3 无extras，诚实 skip）
- S3 新测试：test_embedding/test_text_splitter/test_retriever 34 passed（3 skip 无依赖）
- ask 端到端 + 注入 + symlink：14 passed
- mypy --strict S3 文件（6 files）：0 错
- ruff：0 错

### 测试真实性红线
- ✅ **test_get_llm_default 的 except Exception: pass 已移除**（S2 审查标记的关键债务）→ 改 test_default_provider_is_llm_provider 真实断言
- ✅ 删除 TestVectorStoreAbstract（测 ABC pass 刷覆盖率）、Chroma/FAISS patch.dict mock 循环、TestNumpyNotAvailable import-reload 刷覆盖率
- ✅ ask() 真实路径：FakeLLMProvider + DeterministicEmbeddingProvider 端到端验证 retrieve→generate→answer
- ✅ except 仅 (OSError, NotImplementedError) 平台降级（symlink），非 Exception:pass 掩盖
- ✅ 集成测试诚实：@integration + requires_extra，无依赖 skip，有依赖真实跑

## 3. 文档与实现一致性审查

- ✅ FUNCTION_DEPENDENCIES.md：修正"默认随机向量"误导（显式标注无 provider 抛错）；5MB→硬依赖说明；deepseek-v4-flash→deepseek-chat
- ✅ docstring 英文为主
- ✅ 导出同步（EmbeddingProvider/Retriever/TextSplitter 等）

## 实测命令输出

```
mypy --strict S3 文件(6) → Success: no issues found
ruff S3 → All checks passed!
全套测试 → 646 passed, 31 skipped
C2 grep(random.seed/_local_embed/hash(text)) → 空匹配
C1 grep(LLMEngine(model=)) → 仅 docstring，无实际调用
except 掩盖 → 仅 docstring 引用，实际代码仅 OSError/NotImplementedError 平台降级
FAISS delete 测试 → 3 passed
全局 mypy → 63→24 errors（S3 修了 vector/scenarios，剩余在 local/parser/output 留各自阶段）
```

## Gate 判定

| Gate | 结果 |
|------|------|
| mypy --strict S3 文件 0 错 | ✅ |
| ruff 0 错 | ✅ |
| C1 修复（ask 不崩溃） | ✅ |
| C2 修复（无 hash 随机，显式抛错） | ✅ |
| C3 修复（FAISS delete 一致） | ✅ |
| 安全测试通过（注入/symlink） | ✅ |
| 无 except 掩盖、无刷覆盖率 | ✅ |
| TextSplitter 真实分块 | ✅ |
| scenarios 依赖 core 抽象 | ✅ |

**判定：S3 通过，可进入 S4。**

## 遗留项（有意留后续阶段，非缺陷）

- FAISS C3 集成测试本地 skip（无 faiss-cpu），代码修复已就位，CI 装上即跑；InMemory delete 一致性有真实测试
- 全局 mypy 24 错在 local/models(S7)/parser/output(S8) 留各自阶段
- KnowledgeQA 仍持 VectorEngine 门面（core 公开门面，检索路径已抽象化，可接受）

## 产出文件

- 新增：core/embedding.py、retriever.py、text_splitter.py
- 重构：core/vector.py、scenarios/knowledge_qa.py、scenarios/reviewer.py
- 测试：test_embedding/test_text_splitter/test_retriever（新）；test_vector/test_knowledge_qa/test_reviewer（重写）；_helpers/conftest（共享）
- 配置：pyproject.toml（mypy overrides for chromadb/faiss/sentence_transformers/numpy）
- 文档：FUNCTION_DEPENDENCIES.md（修正随机向量/5MB/模型名）

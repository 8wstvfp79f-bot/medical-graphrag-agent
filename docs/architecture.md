# GraphRAG Architecture

本项目已经完成 ETL、BGE/Milvus 向量检索、Neo4j 图谱检索、Hybrid Evidence Merge、BGE-Reranker、Context Budget、自然语言到受校验 Function Call 的 Agent 编排、FastAPI JSON/SSE 问答服务，以及 250 题 Vector-only/Hybrid 对照与 500 条本地 LLM Judge 评测。

```text
medical.json
├── documents.jsonl → chunks.jsonl → BGE → Milvus → text evidence
└── triples.csv → Neo4j typed graph → graph evidence
                                      ↓
用户问题 → 疾病实体/意图识别 → strict tool call（已完成）
                         ↓
          Vector / Graph / Hybrid 路由与执行（已完成）
                         ↓
            Hybrid merge + Weighted RRF（已完成）
                                      ↓
                    BGE-Reranker + Context Budget（已完成）
                                      ↓
                     E1-En Evidence Context（已完成）
                                      ↓
                   Evidence-bound Answer（已完成）
                                      ↓
                       FastAPI JSON / SSE（已完成）
                                      ↓
                       Retrieval / Answer Evaluation（已完成）
```

Milvus 回答“哪些文本与问题语义相近”；Neo4j 回答“已知疾病和哪些结构化医学实体有关”。Phase 5 已把两路结果归一为统一 evidence，使用 Weighted RRF 合并排名，并按 chunk、graph fact 和同疾病精确文本去重。每条证据保留原始来源、source rank、向量分数和 fusion contribution。Phase 6 再用 Cross-Encoder 同时阅读 query 与 evidence，输出独立的 `rerank_rank/rerank_score`；Context Builder 按 token、条数和来源类型配额选择完整证据块，并生成可追溯引用。

API 中的 Hybrid 双路调用已通过 `asyncio.gather + asyncio.to_thread` 并发执行，避免同步等待 Milvus 完成后才开始 Neo4j。两路完成后仍按固定顺序执行融合、BGE-Reranker 和 Context Budget，保证结果可复盘。系统同时支持无需 API Key 的抽取式引用回答，以及通过 OpenAI-compatible 接口连接 LM Studio/Qwen 的证据约束生成；两种后端都会校验 `[E1]` 等引用编号。

Phase 7 已能从自然语言问题中识别精确疾病实体与查询意图，再决定调用向量检索、图谱检索或两者同时调用。图谱底层仍只接受结构化的 `disease_name + relations + limit`；Hybrid Retrieval 会把同一个 `disease_name` 同步为 Milvus 的精确 metadata filter，并拒绝两路疾病不一致。若疾病缺失、同时出现多个疾病、参数越界或关系不在白名单，Agent 会在数据库调用前停止。Phase 8 通过 `MedicalChatService` 将计划、工具执行、证据和答案统一成一个请求 ID，并由 `/chat` 同时提供完整 JSON 与 SSE 两种传输方式。

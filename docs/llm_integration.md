# 真实大模型生成与大模型裁判

## 当前状态

代码已经支持 OpenAI 兼容对话接口，可以连接本地 LM Studio、Ollama 兼容网关或托管模型服务。当前机器已通过 LM Studio 接入 `qwen2.5-7b-instruct`，完成 500 个真实回答和 500/500 条裁判评分。实测纯向量/混合检索语义幻觉题目占比为 49.6%/29.6%，绝对下降 20.0 个百分点、相对下降 40.3%。该结果来自模型评审，不等于临床专家结论。

## 两个模型角色

生成模型和裁判模型应尽量使用不同配置，避免“自己生成、自己打分”：

- 答案生成器：读取上下文预算内的 E1-En 证据，生成带引用回答；
- 大模型裁判：读取问题、回答和证据，对忠实度、相关性、完整性评分，并列出无证据支持的陈述。

两者共享`src/llm/client.py`，但分别读取`GENERATION_LLM_*`和`JUDGE_LLM_*`环境变量；没有单独配置时才回退到通用`LLM_*`变量。

## 配置示例

```dotenv
ANSWER_GENERATOR_BACKEND=openai-compatible

GENERATION_LLM_BASE_URL=http://127.0.0.1:1234
GENERATION_LLM_MODEL=your-generation-model
GENERATION_LLM_API_KEY=
GENERATION_LLM_FALLBACK=true

JUDGE_LLM_BASE_URL=http://127.0.0.1:1234
JUDGE_LLM_MODEL=your-judge-model
JUDGE_LLM_API_KEY=
JUDGE_LLM_FAITHFULNESS_THRESHOLD=0.8
```

LM Studio 0.4只接受`response_format.type=json_schema`或`text`。本项目会在生成回答和Judge评分时发送明确的JSON Schema，由LM Studio约束字段和类型；对于只支持旧式`json_object`的其他兼容服务，客户端仍保留回退处理。这样可以避免本地模型偶尔输出缺少逗号的“近似JSON”导致长批次中断。

托管服务的密钥只写在本地`.env`中；该文件已被Git忽略。不要把Key写入README、命令行参数、评测JSON或截图。

## 执行顺序

先用真实生成模型重新生成Vector-only与Hybrid两套回答：

```powershell
python -m src.evaluation.runner --fail-fast
```

确认`results/evaluation/phase9_metrics.json`中的`answer_generation.mode`为`llm_grounded`，而不是`extractive_fallback`。随后运行Judge：

```powershell
python -m src.evaluation.judge_runner --concurrency 1 --fail-on-error
```

Judge输出：

- `results/evaluation/phase9_metrics.judge_results.jsonl`：每题逐项评分；
- `results/evaluation/phase9_judge_summary.json`：Vector/Hybrid聚合与幻觉率下降；
- `docs/evaluation_report.md`：检索、回答、Judge 与 SSE 性能的统一可读报告。

可先使用`--limit 10`验证模型输出schema和费用，再运行全部500条请求。本机LM Studio在并发2时会使长上下文请求不稳定，因此正式结果使用并发1。运行器每完成一条便写入结果；若进程意外中断，在相同命令末尾增加`--resume`即可跳过已完成项。

## 语义幻觉率口径

当裁判列出至少一个 `unsupported_claim`，或忠实度低于配置阈值时，该题计为“出现语义幻觉”。

```text
语义幻觉题目占比 = 出现幻觉的题数 / Judge成功完成的题数

相对下降比例 =
  (Vector幻觉率 - Hybrid幻觉率) / Vector幻觉率
```

只有真实裁判完成纯向量和混合检索两组评分，且纯向量幻觉率大于 0 时，才会输出相对下降比例。本次 250 组配对问题、500 条评分全部成功，实测相对下降 40.3%；不能修改阈值或删题来迎合目标。

当前答案生成器与裁判都使用 `qwen2.5-7b-instruct`。这能验证完整本地模型链路，但同模型自评可能存在偏差；固定测试集也来自本项目清洗数据，尚未经过独立临床专家标注。更强的结论应再用不同裁判模型和人工抽样复核。

## 引用安全

LLM生成器只接收预算内证据。模型以结构化的“事实句+citation_ids”返回，JSON Schema把可选编号限制为当前证据包中的E1-En，应用层再统一渲染`[E1]`格式；空答案、无引用或越界引用会触发一次修复，之后仍不合法才失败或显式回退。该检查能阻止伪造引用，但不能代替Judge对语义是否真的被证据支持的判断。

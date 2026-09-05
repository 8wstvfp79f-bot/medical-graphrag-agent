const byId = (id) => document.getElementById(id);

const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

function setVisible(activeId) {
  ["emptyState", "loadingState", "resultState", "errorState"].forEach((id) => {
    byId(id).classList.toggle("hidden", id !== activeId);
  });
}

async function checkHealth() {
  const badge = byId("healthBadge");
  try {
    const response = await fetch("/health");
    const payload = await response.json();
    if (!response.ok || !payload.service_ready) throw new Error("service unavailable");
    badge.className = "health online";
    badge.querySelector("span").textContent = "Milvus / Neo4j 服务已连接";
  } catch (_) {
    badge.className = "health offline";
    badge.querySelector("span").textContent = "后端服务未就绪";
  }
}

function renderResult(payload) {
  byId("answer").textContent = payload.answer || "没有生成回答。";
  const plan = payload.query_plan || {};
  const generation = payload.answer_generation || {};
  const budget = payload.context_budget || {};
  const evidence = Array.isArray(payload.evidence) ? payload.evidence : [];
  const citations = Array.isArray(payload.citations) ? payload.citations : [];
  byId("modeBadge").textContent = `${plan.selected_tool || "no-tool"} · ${generation.mode || "unknown"}`;
  byId("stats").innerHTML = [
    [evidence.length, "上下文证据"],
    [citations.length, "有效引用"],
    [budget.used_tokens ?? "—", "Context Tokens"],
  ].map(([value, label]) => `<div class="stat"><b>${escapeHtml(value)}</b><span>${label}</span></div>`).join("");

  byId("evidenceList").innerHTML = evidence.length
    ? evidence.map((item) => {
        const type = String(item.evidence_type || "vector");
        const title = item.citation_id || item.evidence_id || "Evidence";
        const meta = [item.disease_name, item.relation, item.source].filter(Boolean).join(" · ");
        const content = item.content || item.path_text || item.entity_name || "";
        return `<article class="evidence ${escapeHtml(type)}">
          <header><b>${escapeHtml(title)} · ${escapeHtml(type.toUpperCase())}</b><span>${escapeHtml(meta)}</span></header>
          <p>${escapeHtml(content)}</p>
        </article>`;
      }).join("")
    : '<p class="subtitle">本次没有返回可展示的证据。</p>';
  setVisible("resultState");
}

byId("chatForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const submit = byId("submitButton");
  const metadataFilter = {};
  const department = byId("department").value.trim();
  const category = byId("category").value.trim();
  if (department) metadataFilter.department = department;
  if (category) metadataFilter.category = category;

  const request = {
    query: byId("query").value.trim(),
    top_k: Number(byId("topK").value),
    vector_top_k: 6,
    graph_top_k: 6,
    allow_partial: false,
    stream: false,
  };
  const disease = byId("disease").value.trim();
  if (disease) request.disease_name = disease;
  if (Object.keys(metadataFilter).length) request.metadata_filter = metadataFilter;

  submit.disabled = true;
  setVisible("loadingState");
  byId("modeBadge").textContent = "检索中";
  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    const payload = await response.json();
    if (!response.ok) {
      const detail = payload.detail || payload;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    renderResult(payload);
  } catch (error) {
    byId("errorState").textContent = `请求失败：${error.message}。请确认 Milvus、Neo4j、LM Studio 与 FastAPI 均已启动。`;
    byId("modeBadge").textContent = "请求失败";
    setVisible("errorState");
  } finally {
    submit.disabled = false;
  }
});

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    byId("query").value = button.dataset.question;
    byId("disease").value = button.dataset.disease;
  });
});

checkHealth();

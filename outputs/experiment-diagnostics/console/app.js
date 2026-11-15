"use strict";
const byId = id => document.getElementById(id);
const fragment = new URLSearchParams(location.hash.slice(1));
byId("access").value = fragment.get("access") || "";
const requestedJob = fragment.get("job");
if (requestedJob && /^[a-f0-9]{32}$/.test(requestedJob)) sessionStorage.setItem("diagnosis-job-id", requestedJob);
history.replaceState(null, "", location.pathname);
byId("resume").disabled = !sessionStorage.getItem("diagnosis-job-id");
let activeRecord = null;
const displayText = value => typeof value === "string" ? value : JSON.stringify(value, null, 2);
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
async function request(path, payload) {
  const options = {headers: {Authorization: "Bearer " + byId("access").value}};
  if (payload !== undefined) {
    options.method = "POST";
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(payload);
  }
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) {
    const messages = {unauthorized: "访问口令不正确，请使用启动链接中的口令。", queue_full: "任务队列已满，请稍后重试。", invalid_materials: "材料格式或大小不符合要求，请检查 JSON 和字段长度。"};
    throw new Error(messages[data.error] || data.error || "请求失败");
  }
  return data;
}
function render(record) {
  activeRecord = record;
  const target = byId("result");
  target.replaceChildren();
  target.hidden = false;
  byId("empty").hidden = true;
  byId("download").disabled = false;
  target.scrollIntoView({behavior: "smooth", block: "start"});
  if (record.persistence_error) target.append(element("p", "结果暂时可见，但保存失败，服务重启后可能无法恢复。请下载记录。", "warn"));
  if (["failed", "interrupted"].includes(record.state)) {
    target.append(element("p", "任务失败：" + record.error, "warn"));
    return;
  }
  const result = record.result;
  const answer = result.answer;
  target.append(element("span", "验证：未执行", "badge"));
  target.append(element("span", "DeerFlow " + result.deerflow_version, "badge"));
  if (!result.contract_passed) target.append(element("p", "模型输出未通过结构检查，下面仅展示可用内容。请检查原始回答，不应视作已完成诊断。", "warn"));
  if (result.output_format?.recovered) target.append(element("p", "已从附带说明的回答中恢复唯一 JSON；严格 JSON 格式未通过。", "subtle"));
  if (result.report_source === "local_evidence_check") target.append(element("p", "本地检查发现材料不足，请补充下方信息。", "subtle"));
  if (result.normalized_fields?.length) target.append(element("p", "验证计划已整理为列表，原文见下方记录。", "subtle"));
  if (result.citation_audit) {
    const audit = result.citation_audit;
    target.append(element("p", "声明引用来源核对：" + (audit.passed ? "通过" : "未通过") + "。仅核对本次检索来源，不代表引用支持结论。", "subtle"));
    if (audit.unverified_version_citations.length) target.append(element("p", "部分引用的版本适用性未验证，请结合当前环境复核。", "warn"));
  }
  if (answer && typeof answer === "object" && !Array.isArray(answer)) {
    const labels = {observations: "观察证据", hypotheses: "原因假设", recommendations: "修改建议", verification_plan: "待执行验证计划", missing_information: "缺少的信息", limitations: "限制", references: "声明引用"};
    target.append(element("p", answer.status === "insufficient_evidence" ? "当前材料不足，需要补充证据。" : "已生成诊断假设，仍需执行验证。"));
    for (const [key, label] of Object.entries(labels)) {
      target.append(element("h3", label));
      const items = Array.isArray(answer[key]) ? answer[key] : [answer[key] ?? "未提供"];
      const list = element("ul");
      for (const item of items) list.append(element("li", displayText(item)));
      if (!items.length) list.append(element("li", "无"));
      target.append(list);
    }
  }
  const details = element("details");
  details.append(element("summary", "查看工具轨迹与原始回答"));
  details.append(element("pre", JSON.stringify({run_id: result.run_id, trace: result.trace, raw_answer: result.raw_answer, error: result.error}, null, 2)));
  target.append(details);
}
async function followJob(record) {
  const identifier = record.id;
  sessionStorage.setItem("diagnosis-job-id", identifier);
  byId("resume").disabled = false;
  while (["queued", "running"].includes(record.state)) {
    byId("status").textContent = (record.state === "queued" ? "排队中" : "正在检查材料与检索资料") + " · " + identifier.slice(0, 8);
    await new Promise(resolve => setTimeout(resolve, 1500));
    record = await request("/jobs/" + identifier);
  }
  render(record);
  byId("status").textContent = record.state === "completed" ? "报告已生成" : "任务失败";
}
async function resumeJob() {
  if (!byId("access").value) { byId("status").textContent = "请先填写服务访问口令。"; return; }
  const identifier = sessionStorage.getItem("diagnosis-job-id");
  if (!identifier) return;
  byId("submit").disabled = true;
  try { await followJob(await request("/jobs/" + identifier)); }
  catch (error) { byId("status").textContent = error.message; }
  finally { byId("submit").disabled = false; }
}
byId("material-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (!byId("access").value) { byId("status").textContent = "请填写服务访问口令。"; return; }
  byId("submit").disabled = true;
  byId("status").textContent = "正在提交…";
  try {
    const payload = {log: byId("log").value, code: byId("code").value, question: byId("question").value || "分析问题并给出证据与验证计划。", experiment_config: JSON.parse(byId("config").value), environment: JSON.parse(byId("environment").value)};
    await followJob(await request("/jobs", payload));
  } catch (error) {
    byId("status").textContent = error instanceof SyntaxError ? "配置或环境不是合法 JSON。" : error.message;
  } finally { byId("submit").disabled = false; }
});
byId("resume").addEventListener("click", resumeJob);
if (requestedJob && byId("access").value) resumeJob();
byId("sample").addEventListener("click", () => {
  byId("question").value = "请根据日志和配置排查 LoRA 模块匹配问题，并列出还需要哪些信息。";
  byId("log").value = "ValueError: Target modules {'unknown_proj'} not found in the base model. Please check the target modules and try again.";
  byId("config").value = JSON.stringify({model: "Qwen2", lora: {r: 8, target_modules: ["unknown_proj"]}}, null, 2);
  byId("environment").value = JSON.stringify({versions: {torch: "2.5.1", transformers: "4.46.3", peft: "0.13.2"}}, null, 2);
  byId("code").value = "model = get_peft_model(base_model, lora_config)";
});
byId("download").addEventListener("click", () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(activeRecord, null, 2)], {type: "application/json"}));
  const link = element("a");
  link.href = url;
  link.download = "diagnosis-" + activeRecord.id.slice(0, 8) + ".json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

/* Video-3D-LLM WebUI frontend (vanilla JS, no build step). */
const $ = (id) => document.getElementById(id);
const state = {
  datasets: [], defaults: {}, dataset: null, info: null,
  index: null, detail: null, selected: new Set(), showDepth: false,
  offset: 0, limit: 50, total: 0, search: "", job: null, timer: null,
  lastRun: null,
};

function toast(msg, isErr) {
  const el = $("toast");
  el.textContent = msg;
  el.className = "toast" + (isErr ? " err" : "");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add("hidden"), 4200);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  return res.json();
}

/* ------------------------------------------------------------------ boot */
async function boot() {
  try {
    const data = await api("/api/datasets");
    state.datasets = data.datasets;
    state.defaults = data.defaults;
    $("ckpt-label").textContent = "原生 VTC-R1-Glyph · 相同 uniform32 场景帧 · train-free 输入实验";
    const sel = $("dataset-select");
    sel.innerHTML = "";
    state.datasets.forEach((d) => {
      const o = document.createElement("option");
      o.value = d.key;
      o.textContent = `${d.label} · ${d.n_questions} 条 · ${d.task_type}`;
      if (d.error) o.textContent += " (错误)";
      sel.appendChild(o);
    });
    const st = $("opt-strategy");
    st.innerHTML = "";
    (state.defaults.strategies || []).forEach((s) => {
      const o = document.createElement("option"); o.value = s; o.textContent = s; st.appendChild(o);
    });
    st.value = state.defaults.frame_strategy;
const cotSel = $("cot-mode");
    cotSel.innerHTML = "";
    (state.defaults.cot_order || ["off"]).forEach((k) => {
      const o = document.createElement("option");
      o.value = k;
      o.textContent = (state.defaults.cot_presets[k] || {}).label || k;
      cotSel.appendChild(o);
    });
    cotSel.value = state.defaults.enable_thinking ? 'native' : 'off';
    cotSel.addEventListener("change", () => { syncCot(); });
    $("cot-force").addEventListener("change", () => { syncCot(); });
    syncCot();
    $("opt-maxframes").value = state.defaults.max_frames;
    $("opt-maxtokens").value = state.defaults.max_new_tokens;
    $("opt-temp").value = state.defaults.temperature;
    $("opt-beams").value = state.defaults.num_beams;
    $("hint-strategy").textContent = state.defaults.frame_strategy;
    $("hint-maxframes").textContent = state.defaults.max_frames;
    sel.addEventListener("change", () => selectDataset(sel.value));
    $("search-btn").addEventListener("click", doSearch);
    $("search-input").addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });
    $("search-clear").addEventListener("click", () => { $("search-input").value = ""; state.search = ""; state.offset = 0; loadQuestions(); });
    $("prev-page").addEventListener("click", () => { state.offset = Math.max(0, state.offset - state.limit); loadQuestions(); });
    $("next-page").addEventListener("click", () => { state.offset += state.limit; loadQuestions(); });
    $("frames-reset").addEventListener("click", applyOfficialFrames);
    $("frames-all").addEventListener("click", () => { state.detail.frames.forEach((f) => state.selected.add(f.i)); renderFrames(); });
    $("frames-clear").addEventListener("click", () => { state.selected.clear(); renderFrames(); });
    $("frames-stride").addEventListener("click", () => {
      const step = Math.max(1, parseInt($("frames-stride-n").value || "1", 10));
      state.selected.clear();
      state.detail.frames.forEach((f) => { if (f.i % step === 0) state.selected.add(f.i); });
      renderFrames();
    });
    $("frames-count").addEventListener("click", () => {
      const k = Math.max(1, parseInt($("frames-count-n").value || "1", 10));
      const n = state.detail.frames.length;
      state.selected.clear();
      linspace(0, n - 1, k).forEach((i) => state.selected.add(i));
      renderFrames();
    });
    $("show-depth").addEventListener("change", (e) => { state.showDepth = e.target.checked; renderFrames(); });
    $("prompt-reset").addEventListener("click", resetPrompt);
    $('prompt-version').value=state.defaults.prompt_version;
    $('prompt-version').addEventListener('change',syncPromptVersion);
    $('prompt-version-toggle').addEventListener('click',()=>{$('prompt-version').value=promptVersion()==='optimized'?'original':'optimized';syncPromptVersion();});
    $('extra-prompt').addEventListener('input',()=>{extraPromptOverrides[promptVersion()]=$('extra-prompt').value;});
    $('question-input').addEventListener('input',()=>{questionOverrides[questionPromptKey()]=$('question-input').value;});
    $('output-prompt').addEventListener('input',()=>{outputPromptOverrides[outputPromptKey()]=$('output-prompt').value;});
    $('output-prompt-reset').addEventListener('click',resetOutputPrompt);
    $("run-btn").addEventListener("click", runInference);
    $("preview-btn").addEventListener("click", previewInput);
    $('input-mode').addEventListener('change',syncInputMode);
    $('document-reset').addEventListener('click',()=>loadDocument(true));
    $('compare-btn').addEventListener('click',compareModes);
    $('unload-btn').addEventListener('click',async()=>{
      try {await api('/api/model/unload',{method:'POST'});await refreshStatus();toast('已释放本模型显存');}
      catch(e){toast(e.message,true);}
    });
    await selectDataset(state.datasets[0].key);
    initBatch();
    await loadHistory();
    refreshStatus();
    setInterval(refreshStatus, 8000);
  } catch (e) {
    toast("初始化失败: " + e.message, true);
  }
}

function linspace(a, b, n) {
  n = Math.max(1, Math.min(n, b - a + 1));
  const out = [];
  if (n === 1) return [a];
  for (let i = 0; i < n; i++) out.push(Math.floor(a + ((b - a) * i) / (n - 1)));
  return [...new Set(out)];
}

/* --------------------------------------------------------------- status */
async function refreshStatus() {
  try {
    const s = await api("/api/status");
    const dot = $("status-dot"), txt = $("status-text");
    if (s.busy) { dot.className = "dot busy"; txt.textContent = "推理中…"; }
    else if (s.model_loaded) {
      dot.className = "dot ok";
      const idle = (s.idle_unload_sec > 0) ? ` · 空闲 ${Math.round(s.idle_seconds)}s / ${s.idle_unload_sec}s 后自动释放显存` : ` · 空闲自动释放已关闭`;
      txt.textContent = `模型已加载 (${s.selected_gpu ? '物理 GPU'+s.selected_gpu.physical_index : s.device}) · 加载耗时 ${fmt(s.load_seconds)}s${idle}`;
    }
    else if (s.load_error) { dot.className = "dot err"; txt.textContent = "模型加载失败: " + s.load_error; }
    else { dot.className = "dot"; txt.textContent = "模型未加载（首次推理会自动加载）"; }
    if (s.gpu && s.gpu.gpus) {
      const g = s.gpu.gpus.map((x) => `GPU${x.index} ${x.memory_used_mb}/${x.memory_total_mb}MB`).join(" | ");
      txt.textContent += "  ·  " + g;
    }
  } catch (e) { /* ignore */ }
}

/* -------------------------------------------------------------- dataset */
async function selectDataset(key) {
  state.dataset = key;
  state.info = state.datasets.find((d) => d.key === key);
  $("dataset-select").value = key;
  state.offset = 0; state.search = "";
  $("q-badge").textContent = state.info ? `${state.info.label} · ${state.info.split}` : "";
  $("box-input-field").classList.toggle("hidden", !(state.info && state.info.expects_box_input));
  if (typeof syncCot === "function") syncCot();
  renderRefTable();
  await loadQuestions();
}

async function loadQuestions() {
  const q = state.search ? `&q=${encodeURIComponent(state.search)}` : "";
  const data = await api(`/api/datasets/${state.dataset}/questions?offset=${state.offset}&limit=${state.limit}${q}`);
  state.total = data.total;
  $("q-count").textContent = `(${data.total})`;
  $("page-info").textContent = data.mode === "search"
    ? `命中 ${data.total}`
    : `${data.offset + 1}–${Math.min(data.offset + state.limit, data.total)} / ${data.total}`;
  const list = $("question-list");
  list.innerHTML = "";
  data.items.forEach((it) => {
    const d = document.createElement("div");
    d.className = "q-item" + (it.index === state.index ? " active" : "");
    d.innerHTML = `<div class="q-id">#${it.index} · ${escapeHtml(String(it.id))} · ${escapeHtml(String(it.scene || ""))}</div>
                   <div class="q-txt">${escapeHtml(String(it.question || ""))}</div>`;
    d.addEventListener("click", () => selectQuestion(it.index));
    list.appendChild(d);
  });
}

function doSearch() {
  state.search = $("search-input").value.trim();
  state.offset = 0;
  loadQuestions();
}

/* ------------------------------------------------------------- question */
async function selectQuestion(index) {
  state.index = index;
  document.querySelectorAll(".q-item").forEach((el) => el.classList.remove("active"));
  const d = await api(`/api/datasets/${state.dataset}/questions/${index}`);
  state.detail = d;
  $('actual-inputs').innerHTML='';
  $('comparison-area').classList.add('hidden');
  $('input-summary').textContent='请选择模式并预览本题实际输入。';
  $("empty-hint").classList.add("hidden");
  $("work-body").classList.remove("hidden");
  $("result-area").classList.add("hidden");
  $("v-index").textContent = `#${d.index}`;
  $("v-id").textContent = String(d.id);
  $("v-scene").textContent = String(d.scene);
  $("v-type").textContent = String(d.question_type || "-");
  $("gt-box").textContent = prettyGT(d.gt);
  syncPromptVersion();
  if (d.box_input) $("box-input").value = d.box_input.slice(0, 3).join(", ");
  $("frame-meta").textContent = `${d.n_frames} 帧可用（官方 embodiedscan 顺序）`;
  syncOutputPrompt();
  $("cot-force").checked = false;
  $("cot-force-text").value = "";
  syncCot();
  applyOfficialFrames();
  loadQuestions();
  state.documentScene=null; syncInputMode();
}

function prettyGT(gt) {
  if (gt === null || gt === undefined) return "-";
  if (Array.isArray(gt)) {
    if (gt.length && Array.isArray(gt[0])) return gt.map((b, i) => `#${i}: [${b.map(fmtNum).join(", ")}]`).join("\n");
    if (gt.length === 6 && gt.every((x) => typeof x === "number")) return `[${gt.map(fmtNum).join(", ")}]`;
    return gt.map((x, i) => `${i + 1}. ${typeof x === "string" ? x : JSON.stringify(x)}`).join("\n");
  }
  if (typeof gt === "object") return JSON.stringify(gt, null, 2);
  return String(gt);
}
const fmtNum = (x) => (typeof x === "number" ? Number(x.toFixed(4)) : x);

function cotPreset() {
  const k = $("cot-mode").value;
  return (state.defaults.cot_presets || {})[k] || { label: k, instruction: "", split: false };
}

function cotEnabled() {
  return $("cot-mode").value !== "off";
}

function syncCot() {
  $('cot-hint').textContent='默认开启，与原 VTC 推理模板一致；关闭会加入 /nothink。默认输出预算为原项目的 8192 token，可在高级参数调整；这包含思考和答案，不是强制输出长度。';
}

const outputPromptOverrides = {};
const extraPromptOverrides = {};
const questionOverrides = {};
function promptVersion(){return $('prompt-version').value;}
function questionPromptKey(){return `${promptVersion()}:${state.dataset}:${state.index}`;}
function outputPromptKey(){return `${promptVersion()}:${state.dataset}:${$('input-mode').value}`;}
function versionDefaults(){return state.defaults.prompt_versions[promptVersion()];}
function versionQuestion(){
  const original=stripImage(state.detail.prompt);
  return promptVersion()==='optimized' ? original.replace(/Answer the question simply\b/gi,'Answer with one word or phrase') : original;
}
function syncPromptVersion(){
  $('prompt-version-toggle').textContent=promptVersion()==='optimized'?'切换到原版':'切换到新版';
  if(!state.detail) return;
  $('extra-prompt').value=Object.prototype.hasOwnProperty.call(extraPromptOverrides,promptVersion())
    ? extraPromptOverrides[promptVersion()] : versionDefaults().extra_prompt;
  $('question-input').value=Object.prototype.hasOwnProperty.call(questionOverrides,questionPromptKey())
    ? questionOverrides[questionPromptKey()] : versionQuestion();
  syncOutputPrompt();
}
function syncOutputPrompt(){
  if(!state.dataset) return;
  const key=outputPromptKey();
  $('output-prompt').value=Object.prototype.hasOwnProperty.call(outputPromptOverrides,key)
    ? outputPromptOverrides[key] : versionDefaults().output_prompts[state.dataset][$('input-mode').value];
}
function resetOutputPrompt(){delete outputPromptOverrides[outputPromptKey()];syncOutputPrompt();}

function stripImage(text) {
  return String(text || "").replace(/^\s*<image>\s*/, "");
}

function resetPrompt() {
  delete extraPromptOverrides[promptVersion()];
  delete questionOverrides[questionPromptKey()];
  delete outputPromptOverrides[outputPromptKey()];
  syncPromptVersion();
  toast("已恢复默认提示词与格式");
}

/* ---------------------------------------------------------------- frames */
function applyOfficialFrames() {
  state.selected = new Set(state.detail.default_frame_indices);
  renderFrames();
}

function renderFrames() {
  const grid = $("frame-grid");
  grid.innerHTML = "";
  state.detail.frames.forEach((f) => {
    const el = document.createElement("div");
    el.className = "frame" + (state.selected.has(f.i) ? " sel" : "");
    const url = (state.showDepth ? f.depth : f.rgb) + "&w=200";
    el.innerHTML = `<span class="tag">${f.i}</span><img loading="lazy" src="${url}" alt="frame ${f.i}" />
      <div class="fmeta"><span>${escapeHtml(f.name)}</span><span>${f.time}s</span></div>`;
    el.title = `frame index ${f.i} · ${f.name} · t≈${f.time}s`;
    el.addEventListener("click", () => {
      if (state.selected.has(f.i)) state.selected.delete(f.i); else state.selected.add(f.i);
      renderFrames();
    });
    grid.appendChild(el);
  });
  $("frames-selected-info").textContent = `已选 ${state.selected.size} 帧`;
}

/* -------------------------------------------------------------- inference */
function currentBoxInput() {
  if (!state.info || !state.info.expects_box_input) return null;
  const raw = $("box-input").value;
  const nums = (raw.match(/-?\d+(\.\d+)?([eE][-+]?\d+)?/g) || []).map(Number);
  return nums.length ? nums.slice(0, 3) : null;
}

function buildPayload() {
  return {
    input_mode:$('input-mode').value,
    description_mode:$('description-mode').value,
    description_text:state.documentScene===state.detail.scene ? $('description-text').value : null,
    relations_enabled:$('relations-enabled').checked,
    relation_text:state.documentScene===state.detail.scene ? $('relation-text').value : null,
    image_max_pixels:Number($('image-max-pixels').value),
    enable_thinking:$('cot-mode').value==='native',
    dataset: state.dataset,
    index: state.index,
    extra_prompt: $("extra-prompt").value,
    prompt_version: promptVersion(),
    output_prompt: $('output-prompt').value,
    question: $("question-input").value,
    frame_indices: [...state.selected].sort((a, b) => a - b),
    frame_strategy: $("opt-strategy").value,
    max_frames: parseInt($("opt-maxframes").value || "32", 10),
    decode_strategy: $("opt-decode").value,
    temperature: parseFloat($("opt-temp").value || "0"),
    num_beams: parseInt($("opt-beams").value || "1", 10),
    max_new_tokens: parseInt($("opt-maxtokens").value || "512", 10),
box_input: currentBoxInput(),
    cot_mode: $("cot-mode").value,
    cot_instruction: ($("cot-mode").value === "custom") ? $("cot-custom").value : null,
    cot_force_prefix: $("cot-force").checked,
    cot_force_prefix_text: $("cot-force-text").value,
  };
}

async function previewInput() {
  $("run-status").textContent = "构建中…";
  try {
    await syncInputMode();
    const p=await api('/api/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(buildPayload())});
    renderInput(p);
    $('r-files').innerHTML='';
    $('r-latency').textContent='仅预览，尚未运行推理';
    $("result-area").classList.remove("hidden");
    $("r-gt").textContent = prettyGT(state.detail ? state.detail.gt : null);
    $("r-gt-badge").textContent = gtBadge(state.detail ? state.detail.gt : null);
    $("r-match").textContent = "";
    $("r-match").className = "badge";
    $("r-prediction").textContent = "(尚未运行)";
    $("r-thinking").textContent = "";
    $("thinking-field").classList.add("hidden");
    $("r-raw").textContent = "(尚未运行)";
    $("r-prompt").textContent = p.model_prompt;
    $("r-extra").textContent = JSON.stringify({
      frame_indices: p.frame_indices, frame_names: p.frame_names,
      n_frames_available: p.n_frames_available, frame_strategy: p.frame_strategy,
      max_frames: p.max_frames, decode_strategy: p.decode_strategy,
    }, null, 2);
    $("metrics-table").innerHTML = "";
    $("metrics-note").textContent = "这是将要送入模型的输入（未运行推理）。";
    $("run-status").textContent = "";
  } catch (e) { toast(e.message, true); $("run-status").textContent = ""; }
}

async function runInference() {
  if (!state.selected.size) { toast("请至少选择一帧", true); return; }
  await syncInputMode();
  const payload = buildPayload();
  $("run-btn").disabled = true;
  $("run-status").textContent = "提交任务…";
  try {
    const { job_id } = await api("/api/infer", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    state.job = job_id;
    pollJob(job_id);
  } catch (e) {
    toast(e.message, true); $("run-btn").disabled = false; $("run-status").textContent = "";
  }
}

function pollJob(jobId) {
  clearInterval(state.timer);
  const tick = async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      if (job.status === "queued") $("run-status").textContent = "排队中…";
      else if (job.stage === "loading_model") $("run-status").textContent = "正在构建输入或加载原生 VTC 权重…";
      else if (job.status === "running") $("run-status").textContent = "推理中…（含官方指标计算）";
      if (job.status === "done") {
        clearInterval(state.timer);
        $("run-btn").disabled = false;
        $("run-status").textContent = "";
        renderResult(job.result);
        loadHistory();
      } else if (job.status === "error") {
        clearInterval(state.timer);
        $("run-btn").disabled = false;
        $("run-status").textContent = "";
        toast("推理失败: " + job.error, true);
        console.error(job.traceback);
      }
    } catch (e) { /* keep polling */ }
  };
  tick();
  state.timer = setInterval(tick, 2000);
}

function renderResult(r) {
  state.lastRun = r;
  renderInput(r);
  $("result-area").classList.remove("hidden");
  $("r-latency").textContent = `${r.dataset}#${r.index} · ${r.run_id} · 推理 ${fmt(r.latency_s)}s · ${r.frame_indices.length} 帧 · 输入 ${r.input_tokens} / 输出 ${r.output_tokens} tokens`;
const gt = (r.gt !== undefined && r.gt !== null) ? r.gt : (state.detail ? state.detail.gt : null);
  $("r-gt").textContent = prettyGT(gt);
  $("r-gt-badge").textContent = gtBadge(gt);
  const mb = matchBadge(r.metrics);
  $("r-match").textContent = mb.text;
  $("r-match").className = "badge" + (mb.cls ? " " + mb.cls : "");
  $("r-prediction").textContent = typeof r.prediction === "string" ? r.prediction : JSON.stringify(r.prediction);
  $("thinking-field").classList.remove("hidden");
  if (r.thinking) {
    $("r-thinking").textContent = r.thinking;
  } else {
    $("r-thinking").textContent = "本次没有完整的显式思考块。原始输出完整保留在下方。";
  }
  $("r-raw").textContent = r.raw_output == null ? "(该任务不产生文本输出)" : r.raw_output;
  $('r-files').innerHTML=r.artifacts_url ? `<a href="${escapeHtml(r.artifacts_url)}" target="_blank">打开本次完整图片与 Prompt</a> · <a href="${escapeHtml(r.artifacts_url.replace('inputs.html','result.json'))}" target="_blank">输入 / 输出 / GT / 指标 JSON</a>` : '';
  $("r-prompt").textContent = r.model_prompt;
  $("r-extra").textContent = JSON.stringify({
    frame_indices: r.frame_indices, frame_names: r.frame_names,
    frame_strategy: r.frame_strategy, max_frames: r.max_frames,
    decode_strategy: r.decode_strategy, temperature: r.temperature, num_beams: r.num_beams,
    input_mode:r.input_mode, description_mode:r.description_mode,
    parse_error:r.parse_error, artifact_dir:r.artifact_dir,
    input_tokens:r.input_tokens,output_tokens:r.output_tokens,output_limit_reached:r.output_limit_reached,
    answer_protocol:r.answer_protocol,provenance:r.provenance,
    extra: r.extra,
    cot_mode: r.cot_mode,
    cot_force_prefix: r.cot_force_prefix,
    metrics_subject: r.metrics_subject,
  }, null, 2);
  renderMetrics(r.metrics);
  renderRefTable();
  $("result-area").scrollIntoView({ behavior: "smooth", block: "start" });
}

function gtBadge(gt) {
  if (gt === null || gt === undefined) return "";
  if (Array.isArray(gt)) {
    if (gt.length && Array.isArray(gt[0])) return `${gt.length} 个 GT box`;
    if (gt.length === 6 && gt.every((x) => typeof x === "number")) return "1 个 GT box";
    return `${gt.length} 个参考答案`;
  }
  return "";
}

function matchBadge(m) {
  if (!m || typeof m !== "object") return { text: "", cls: "" };
  if ("EM" in m) return m.EM ? { text: "✓ 命中 GT", cls: "ok" } : { text: "✗ 未命中", cls: "bad" };
  if ("EM(acc)" in m) return m["EM(acc)"] ? { text: "✓ 命中 GT", cls: "ok" } : { text: "✗ 未命中", cls: "bad" };
  if ("IoU" in m) {
    const iou = Number(m.IoU);
    const cls = m["IoU@0.5"] ? "ok" : (m["IoU@0.25"] ? "warn" : "bad");
    const tag = m["IoU@0.5"] ? " ✓ IoU≥0.5" : (m["IoU@0.25"] ? " ~ IoU≥0.25" : " ✗ IoU<0.25");
    return { text: `3D IoU ${iou.toFixed(4)}` + tag, cls };
  }
  if ("F1@0.25" in m) {
    const f25 = Number(m["F1@0.25"]), f50 = Number(m["F1@0.5"]);
    const cls = f25 >= 0.999 ? "ok" : (f25 > 0 ? "warn" : "bad");
    return { text: `F1@0.25 ${f25.toFixed(2)} · F1@0.5 ${f50.toFixed(2)}`, cls };
  }
  return { text: "", cls: "" };
}

const METRIC_LABELS = {
  "CIDEr": "CIDEr（官方首要指标）", "BLEU-1": "BLEU-1", "BLEU-2": "BLEU-2", "BLEU-3": "BLEU-3",
  "BLEU-4": "BLEU-4", "METEOR": "METEOR", "ROUGE-L": "ROUGE-L", "EM": "EM（完全匹配）",
  "EM(acc)": "EM / Accuracy", "IoU": "3D IoU", "IoU@0.25": "IoU@0.25", "IoU@0.5": "IoU@0.5",
  "F1@0.25": "F1@0.25", "F1@0.5": "F1@0.5",
};

function renderMetrics(m) {
  const t = $("metrics-table");
  if (!m || typeof m !== "object") { t.innerHTML = ""; $("metrics-note").textContent = "无指标。"; return; }
  if (m.error) { t.innerHTML = ""; $("metrics-note").textContent = "指标计算失败：" + m.error; return; }
  const rows = [];
  Object.keys(m).filter((k) => !k.startsWith("_")).forEach((k) => {
    const v = m[k];
    rows.push(`<tr><th>${METRIC_LABELS[k] || k}</th><td class="val">${formatMetric(v,k)}</td></tr>`);
  });
  t.innerHTML = rows.join("");
  const extra = Object.keys(m).filter((k) => k.startsWith("_"));
  const bits = extra.filter((k) => !["_gt_normalized", "_pred_normalized", "_pred_box", "_gt_box", "_pred_boxes", "_gt_boxes", "_scores"].includes(k))
    .map((k) => `${k.slice(1)} = ${JSON.stringify(m[k])}`);
const r=state.lastRun || {};
const cotNote=' 原生 VTC 的预测通过参考项目的 scorer 计算；逐样本分数不等于完整榜单成绩。';
  $('metrics-note').innerHTML=escapeHtml(cotNote)+
    (bits.length ? " " + escapeHtml(bits.join(" · ")) : "");
}

function formatMetric(v,key='') {
  if (Array.isArray(v)) return v.map(fmt).join(", ");
  if (typeof v === "number") return /^(EM|EM\(acc\)|IoU|IoU@|F1@)/.test(key) ? `${(v * 100).toFixed(2)}%` : v.toFixed(4);
  return String(v);
}

function renderRefTable() {
  const info = state.info;
  const t = $("ref-table");
  if (!info || !info.reference_metrics) { t.innerHTML = ""; return; }
  const rows = Object.keys(info.reference_metrics).map((k) =>
    `<tr><th>${METRIC_LABELS[k] || k}</th><td class="val">${formatRef(k, info.reference_metrics[k])}</td></tr>`);
  rows.push(`<tr><th>评测脚本</th><td><code>${escapeHtml(info.official_eval_script)}</code></td></tr>`);
  rows.push(`<tr><th>复现命令</th><td><code>${escapeHtml(info.official_eval_cmd)}</code></td></tr>`);
  t.innerHTML = rows.join("");
  $("ref-note").textContent = info.reference_note || "";
}

function formatRef(key, v) {
  if (key.startsWith("F1@")) return v.toFixed(4);
  if (key === "EM") return (v * 100).toFixed(2) + "%";
  return v.toFixed(4);
}

/* --------------------------------------------------------------- history */
async function loadHistory() {
  try {
    const { items } = await api("/api/history?limit=40");
    const box = $("history-list");
    box.innerHTML = "";
    items.forEach((it) => {
      const d = document.createElement("div");
      d.className = "h-item";
      const p = typeof it.prediction === "string" ? it.prediction : JSON.stringify(it.prediction);
      d.textContent = `${it.dataset}#${it.index} · ${it.input_mode || "运行"}/${it.description_mode || "RGB"} · ${it.job_id} · ${(p || it.status || "").slice(0, 30)}`;
      d.addEventListener("click", () => {
        state.dataset = it.dataset;
        $("dataset-select").value = it.dataset;
        state.info = state.datasets.find((x) => x.key === it.dataset);
        selectQuestion(it.index).then(() => api(`/api/jobs/${it.job_id}`).then((j) => { if (j.result) renderResult(j.result); }));
      });
      box.appendChild(d);
    });
  } catch (e) { /* ignore */ }
}

/* ------------------------------------------------------------------ util */
function fmt(v) { return (v === null || v === undefined) ? "-" : Number(v).toFixed(2); }
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

window.addEventListener('DOMContentLoaded', boot);

async function loadDocument(force=false) {
  if(!state.detail) return;
  if(!force && state.documentScene===state.detail.scene) return;
  const scene=state.detail.scene;
  const d=await api('/api/scene-document?scene='+encodeURIComponent(scene));
  if(scene!==state.detail.scene) return;
  $('description-text').value=d.description_text;
  $('relation-text').value=d.relation_text;
  state.documentScene=scene;
}
async function syncInputMode() {
  syncOutputPrompt();
  const numbered=$('input-mode').value==='numbered';
  $('document-controls').classList.toggle('hidden',!numbered);
  $('description-mode').disabled=!numbered;
  if(numbered) await loadDocument();
}
function renderInput(r) {
  const images=r.actual_images || [];
  $('actual-inputs').innerHTML=images.map((im,i)=>`<a href="${escapeHtml(im.url)}" target="_blank" class="frame"><span class="tag">${i}: ${im.kind==='vtc_page'?'VTC页':'RGB'}</span><img loading="lazy" src="${escapeHtml(im.url)}"></a>`).join('');
  $('input-summary').textContent=`实际输入：${r.frame_indices.length}张场景画面，${images.filter(x=>x.kind==='vtc_page').length}张VTC页，${r.input_tokens} tokens；${r.input_mode}/${r.description_mode || 'RGB'}。${r.artifact_dir ? '全部记录：'+r.artifact_dir : '预览不启动GPU推理。'}`;
}
async function waitResult(payload) {
  const {job_id}=await api('/api/infer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  for(;;) {
    const job=await api('/api/jobs/'+job_id);
    if(job.status==='done') return job.result;
    if(job.status==='error') throw Error(job.error);
    await new Promise(resolve=>setTimeout(resolve,2000));
  }
}
async function compareModes() {
  if(!state.selected.size) return toast('请选择帧',true);
  $('run-btn').disabled=$('compare-btn').disabled=true;
  try {
    await loadDocument();
    const p={...buildPayload(),comparison_id:crypto.randomUUID()};
    $('comparison-area').classList.remove('hidden');
    $('comparison-results').innerHTML='';
    for(const mode of ['raw','numbered']) {
      $('run-status').textContent='同题对照：'+mode+' 运行中…';
      const key=`${promptVersion()}:${state.dataset}:${mode}`;
      const outputPrompt=Object.prototype.hasOwnProperty.call(outputPromptOverrides,key)
        ? outputPromptOverrides[key] : versionDefaults().output_prompts[state.dataset][mode];
      const r=await waitResult({...p,input_mode:mode,output_prompt:outputPrompt});
      const column=document.createElement('div'); column.className='cmp-col';
      column.innerHTML=`<h3>${mode==='raw'?'原始RGB':'编号RGB + '+p.description_mode}</h3><p>实验 ${r.run_id} · 输入 ${r.input_tokens} tokens</p><pre class="pre">${escapeHtml(r.answer_text || r.raw_output)}</pre><pre class="pre small">${escapeHtml(JSON.stringify(r.metrics,null,2))}</pre><button class="btn small">查看完整输入输出</button>`;
      column.querySelector('button').onclick=()=>renderResult(r);
      $('comparison-results').appendChild(column);
    }
    await loadHistory();
  } catch(e) {toast(e.message,true);}
  finally {$('run-btn').disabled=$('compare-btn').disabled=false;$('run-status').textContent='';}
}

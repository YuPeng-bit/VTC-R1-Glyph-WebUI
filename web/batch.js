/* Independent batch settings. Never share a single question/frame/document. */
const batchView = {id: null, timer: null};
const batchStatuses = {
  queued:'排队中', running:'运行中', scoring:'官方评分中', stopping:'当前题完成后停止',
  completed:'全部完成', completed_with_errors:'已跑完，但有失败题', cancelled:'已停止',
  interrupted:'服务重启中断，可继续', error:'失败，可查看记录并重试',
};

function batchDefaults() {
  const version = $('batch-prompt-version').value;
  const defaults = state.defaults.prompt_versions[version];
  $('batch-extra-prompt').value = defaults.extra_prompt;
  $('batch-output-prompt').value = defaults.output_prompts.scanqa[$('batch-mode').value];
}

function initBatch() {
  $('batch-prompt-version').value = state.defaults.prompt_version;
  $('batch-tokens').value = state.defaults.max_new_tokens;
  $('batch-thinking').value = state.defaults.enable_thinking ? 'on' : 'off';
  $('batch-model').textContent = `模型：${state.defaults.ckpt}。任务开始后参数固定；全量指 ScanQA val，不是隐藏答案的 test。`;
  batchDefaults();
  $('batch-prompt-version').addEventListener('change', batchDefaults);
  $('batch-mode').addEventListener('change', () => {
    const raw = $('batch-mode').value === 'raw';
    $('batch-description').disabled = raw;
    $('batch-relations').disabled = raw;
    batchDefaults();
  });
  $('batch-description').disabled = $('batch-relations').disabled = true;
  $('batch-selection').addEventListener('change', () => {$('batch-count').disabled = $('batch-selection').value === 'all';});
  $('batch-reset-prompts').addEventListener('click', batchDefaults);
  $('batch-copy-settings').addEventListener('click', () => {
    const mapping = {'batch-mode':'input-mode','batch-description':'description-mode','batch-pixels':'image-max-pixels',
      'batch-prompt-version':'prompt-version','batch-frames':'opt-maxframes','batch-temperature':'opt-temp',
      'batch-beams':'opt-beams','batch-tokens':'opt-maxtokens'};
    for (const [target, source] of Object.entries(mapping)) $(target).value = $(source).value;
    $('batch-relations').checked = $('relations-enabled').checked;
    $('batch-thinking').value = $('cot-mode').value === 'native' ? 'on' : 'off';
    batchDefaults();
    $('batch-extra-prompt').value = $('extra-prompt').value;
    if (state.dataset === 'scanqa') $('batch-output-prompt').value = $('output-prompt').value;
    const raw = $('batch-mode').value === 'raw';
    $('batch-description').disabled = $('batch-relations').disabled = raw;
    toast('已复制参数；每题问题、帧和场景描述会独立读取。');
  });
  $('batch-preview').addEventListener('click', async () => {
    try {
      const request = await api('/api/batch/preview', batchPost(batchPayload()));
      $('batch-settings').textContent = JSON.stringify(request, null, 2);
      $('batch-progress').textContent = `已核对：ScanQA val ${request.selected_count}/${request.total_available} 题；未启动 GPU。模型：${request.checkpoint}`;
    } catch (e) {toast(e.message, true);}
  });
  $('batch-start').addEventListener('click', async () => {
    $('batch-start').disabled = true;
    try {
      const job = await api('/api/batch/start', batchPost(batchPayload()));
      batchWatch(job.batch_id);
      batchHistory();
    } catch (e) {toast(e.message, true); $('batch-start').disabled = false;}
  });
  $('batch-stop').addEventListener('click', async () => {
    if (!batchView.id) return;
    try {batchRender(await api(`/api/batches/${batchView.id}/stop`, batchPost({})));}
    catch (e) {toast(e.message, true);}
  });
  $('batch-resume').addEventListener('click', async () => {
    if (!batchView.id) return;
    $('batch-resume').disabled = true;
    try {
      const job = await api(`/api/batches/${batchView.id}/resume`, batchPost({}));
      batchWatch(job.batch_id);
    } catch (e) {toast(e.message, true); $('batch-resume').disabled = false;}
  });
  $('batch-history').addEventListener('change', () => {
    if ($('batch-history').value) batchWatch($('batch-history').value);
  });
  $('batch-refresh').addEventListener('click', batchHistory);
  batchHistory(true);
}

function batchPost(payload) {
  return {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)};
}

function batchPayload() {
  return {dataset:'scanqa', selection:$('batch-selection').value, count:Number($('batch-count').value),
    seed:Number($('batch-seed').value), options:{
      input_mode:$('batch-mode').value, description_mode:$('batch-description').value,
      relations_enabled:$('batch-mode').value === 'numbered' && $('batch-relations').checked,
      prompt_version:$('batch-prompt-version').value, extra_prompt:$('batch-extra-prompt').value,
      output_prompt:$('batch-output-prompt').value, enable_thinking:$('batch-thinking').value === 'on',
      frame_strategy:'uniform', max_frames:Number($('batch-frames').value),
      image_max_pixels:Number($('batch-pixels').value), temperature:Number($('batch-temperature').value),
      top_p:$('batch-top-p').value.trim() ? Number($('batch-top-p').value) : null,
      num_beams:Number($('batch-beams').value), max_new_tokens:Number($('batch-tokens').value),
    }};
}

function batchRender(job) {
  const running = ['queued','running','scoring','stopping'].includes(job.status);
  $('batch-start').disabled = running;
  $('batch-stop').disabled = !running || job.status === 'stopping';
  $('batch-resume').disabled = running || job.status === 'completed';
  const em = job.em_over_attempted == null ? '-' : (100 * job.em_over_attempted).toFixed(2) + '%';
  const latest = job.latest;
  $('batch-progress').textContent = `批次 ${job.batch_id}：${batchStatuses[job.status] || job.status} / ${job.stage}\n`
    + `已尝试 ${job.attempted}/${job.selected}，成功 ${job.succeeded}，失败 ${job.failed}；当前已尝试题 EM ${em}\n`
    + (latest ? `最近：#${latest.index} ${latest.id}\n预测：${latest.prediction || '(无合法答案)'}\nGT：${JSON.stringify(latest.gt)}\n` : '')
    + (job.error ? `错误：${job.error}\n` : '') + `保存位置：${job.artifact_dir}`;
  $('batch-settings').textContent = JSON.stringify(job.request, null, 2);
  const m = job.metrics;
  $('batch-metrics').innerHTML = m ? '<p>' + (m.scope === 'selected_set' ? '本批选定样本的官方总分' : '未跑完：仅已尝试子集的分数，不能作为全量成绩')
    + '</p><table class="metrics">' + Object.entries(m).map(([k,v]) => `<tr><th>${escapeHtml(k)}</th><td>${escapeHtml(k === 'EM' ? (v*100).toFixed(2)+'%' : String(v))}</td></tr>`).join('') + '</table>'
    : `<p>官方总分：${escapeHtml(job.metrics_status)}。全部生成后统一评分；不会平均逐题 BLEU/CIDEr。</p>`;
  const links = Object.entries(job.files || {}).map(([name,url]) => `<a href="${escapeHtml(url)}" target="_blank">${escapeHtml(name)}</a>`);
  if (job.finished_at) links.unshift(`<a href="${escapeHtml(job.report_url)}" target="_blank">逐题图片、Prompt、输出、GT 与结果表</a>`);
  if (latest && latest.artifacts_url) links.unshift(`<a href="${escapeHtml(latest.artifacts_url)}" target="_blank">最近一题的完整输入</a>`);
  $('batch-links').innerHTML = links.join(' · ');
  return running;
}

function batchWatch(id) {
  batchView.id = id;
  clearInterval(batchView.timer);
  let polling = false;
  const tick = async () => {
    if (polling) return;
    polling = true;
    try {
      const job = await api('/api/batches/' + id);
      if (batchView.id !== id) return;
      if (!batchRender(job)) {clearInterval(batchView.timer); batchHistory();}
    } catch (e) {$('batch-progress').textContent = '读取进度失败，正在重试：' + e.message;}
    finally {polling = false;}
  };
  batchView.timer = setInterval(tick, 3000);
  tick();
}

async function batchHistory(autoSelect=false) {
  try {
    const {items} = await api('/api/batches');
    $('batch-history').innerHTML = '<option value="">选择一个已保存的实验</option>' + items.map(j =>
      `<option value="${j.batch_id}">${j.batch_id} · ${batchStatuses[j.status] || j.status} · ${j.attempted}/${j.selected} · ${j.request.options.input_mode}/${j.request.options.description_mode}</option>`).join('');
    if (batchView.id) $('batch-history').value = batchView.id;
    else if (autoSelect && items.length) {$('batch-history').value=items[0].batch_id; batchWatch(items[0].batch_id);}
  } catch (e) {toast('读取批量记录失败：' + e.message, true);}
}

# ScanQA 批量评测

在界面右侧展开 **ScanQA 批量评测**。默认随机抽 20 题、seed=42；可选择全量验证集。批量设置独立于单题设置，也可以点击“复制当前单题的参数设置”。

可设置原始 RGB / 编号 RGB、文字描述 / VTC 页、自动几何关系、原生思考、提示词版本与可编辑的附加 Prompt / 输出格式、uniform 最大帧数、每图像素上限、temperature、top_p、num_beams、max_new_tokens。使用当前服务 `VTC_CKPT` 指定的权重。

每题从 `data/processed/scanqa_val_llava_style.json` 读取自己的问题和场景。编号模式逐场景复用已有描述，关系由开关控制。不会把单题手动修改的问题、描述、关系或选中帧用于其他题。GT 仅用于展示与生成后评分。

先点“核对样本与设置”可检查抽样清单、数据来源和所有参数，不调用 GPU；点击“开始批量评测”才启动推理。参数及样本列表在任务开始时固定。随机抽样不重复；每题生成随机种子为 `(seed + question_index) % 2**31`。temperature=0 不启用采样。固定种子有助复现，不保证不同设备/软件版本输出逐位一致。

## 进度和继续

批量和单题推理使用同一队列，顺序执行，不重复加载第二份模型。页面显示尝试/成功/失败数量、最近一题答案和 GT，以及已尝试题的 EM。

关闭页面不会停止任务；刷新页面可从“已保存的批量实验”恢复显示。停止按钮在当前题结束后生效，不中断正在进行的生成。服务重启后，未完成任务标记为 `interrupted`，需要显式点击继续。继续时保持原样本和参数，只运行未完成或失败题；成功题不会重跑。评分失败时保留预测，再点继续仅重试评分。

数据文件或模型路径改变时拒绝继续原批次，应新建实验。模型加载失败会立即结束并保留批次，不连续重试数千题。CUDA OOM 保留已完成记录并结束，方便调整配置后创建新批次。

## 分数

复用 Video-3D-LLM 的 ScanQA 评测实现及官方 caption scorer：EM、CIDEr、BLEU-1/2/3/4、METEOR、ROUGE-L。EM 为 0–1，界面显示百分比；其余指标为 scorer 输出乘 100。

生成结束后统一在选定样本上运行 corpus scorer，**不平均逐题 BLEU/CIDEr**。随机子集的 CIDEr 文档频率来自该子集，与全量成绩不能直接等同。预测末尾句号按上游实现去掉后计算 caption 指标，EM 保持上游严格字符串匹配。METEOR 使用单行传输编码，不依据 GT 截词、选择答案或改写预测。

格式解析失败或单题推理失败保留在分母中，使用空预测计分，同时报告失败数量。停止/中断得到的分数标记 `partial_attempted_set`，仅代表已尝试题，不能声称全量完成。完成全部选定题时标记 `selected_set`。官方评分超时/失败会显式报错，不替换为近似评分。默认评分超时 300 秒，可用 `VTC_BATCH_SCORE_TIMEOUT_SEC` 调整。

本入口评测 **ScanQA val**，不是隐藏答案的官方 test。全量运行指当前配置的完整验证集。

## 保存位置

服务器 `results/webui/batches/<batch_id>/`：

- `request.json`：权重、参数、种子、数据哈希、选定题目。
- `selection.json`：固定索引与题目 ID。
- `attempts.jsonl`：追加保存的每次尝试、状态、预测、GT、EM、错误及单题记录链接。
- `predictions.jsonl`：当前预测，字段 `sample_id` / `pred_response`，可交给上游 ScanQA 脚本复评。
- `metrics.json`：官方总分、实际分母、失败数量、覆盖范围。
- `state.json`：当前进度；`report.html`：最终逐题结果表。
- `previous_*/`：继续运行前的历史汇总，旧记录保留。

每题的 `results/webui/<run_id>/` 继续保存实际图片引用与 SHA256、完整 Prompt、渲染页/描述、token、原始生成、最终答案和 GT。界面提供直接打开和下载链接。

## API

`POST /api/batch/preview` 或 `/api/batch/start`：

```json
{
  "dataset": "scanqa",
  "selection": "random",
  "count": 20,
  "seed": 42,
  "options": {
    "input_mode": "raw",
    "prompt_version": "optimized",
    "enable_thinking": true,
    "max_frames": 32,
    "image_max_pixels": 262144,
    "temperature": 0,
    "num_beams": 1,
    "max_new_tokens": 8192
  }
}
```

全量改为 `"selection": "all"`，忽略 count。`GET /api/batches` 列出历史；`GET /api/batches/<id>` 查询进度；`POST /api/batches/<id>/stop` 停止；`POST /api/batches/<id>/resume` 继续。API 明确拒绝单题 question/frame_indices/description_text 等整批复用。

CPU 回归检查：`python -m unittest tools.test_batch -v`。该检查使用明确标记的合成输出，只验证工程行为，不代表真实模型推理通过。

上游评分实现：[Video-3D-LLM eval_scanqa.py](https://github.com/LaVi-Lab/Video-3D-LLM/blob/main/llava/eval/eval_scanqa.py)。

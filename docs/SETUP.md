# 部署说明

## 范围与外部资产

本仓库是原生 VTC-R1-Glyph 的单次推理实验界面，不包含训练代码或多轮思考压缩循环。支持原始 RGB 与编号 RGB + 描述两种模式，描述可作为文字或 Glyph 渲染页输入。默认原生思考开启、输出预算 8192 token、用户优化提示词版本，可在页面修改或切回原版。

需要另行准备：

1. VTC-R1-Glyph 权重和原生 GLM 处理器。实验室所用 Hugging Face snapshot 为 `yiboowang/VTC-R1-Glyph` 的 `a28f42f1c80f1851ba274e01d7c13c6030f990c3`。
2. [Video-3D-LLM](https://github.com/LaVi-Lab/Video-3D-LLM) 参考代码、官方数据布局、ScanNet posed RGB/depth/pose、EmbodiedScan metadata 和五任务问题文件。只读复用 scorer 和帧索引。
3. [VTC-3D-MLLM](https://github.com/YuPeng-bit/VTC-3D-MLLM) 项目的 `src/t50` 代码及其已恢复的实验资产：场景 inventory/Qwen 描述、Glyph 官方 renderer/font、`runtime/t48_render_deps` 等。即使仅使用 RGB，导入和处理器协议也依赖该项目，不能只安装本仓库依赖后直接运行。
4. Java，用于官方 METEOR 评分；ScanNet 数据应由具有相应权限的使用者准备。

模型、场景数据、生成描述、编号照片、视频缓存、历史结果与日志均不在 GitHub 中。新机器需要自行核查输入与实际推理。

## 环境与启动

测试环境为 Python 3.10、Torch 2.8.0、Transformers 4.57.1；另按上游要求安装原生模型和 Glyph 依赖。`requirements-webui.txt` 只列界面层的已使用版本，不会安装整套模型环境。

```bash
git clone https://github.com/YuPeng-bit/VTC-R1-Glyph-WebUI.git
cd VTC-R1-Glyph-WebUI
export VTC_PYTHON=/path/to/native-vtc-env/bin/python
export VTC_CKPT=/path/to/VTC-R1-Glyph
export V3D_REPO=/path/to/Video-3D-LLM
export VTC_ASSETS_ROOT=/path/to/restored/VTC-3D-MLLM
export VTC_INVENTORY_INDEX=/path/to/scene_inventory/index.json
export VTC_CAPTIONS_ROOT=/path/to/captions
"$VTC_PYTHON" -m pip install -r requirements-webui.txt
bash run_webui.sh
```

默认监听 `127.0.0.1:8100`。远程访问使用 `ssh -N -L 8100:127.0.0.1:8100 USER@SSH_HOST`。可设置 `VTC_PORT`、`VTC_HOST`，不要把个人连接密码写入代码。

启动时暴露可见 GPU，首次推理前选空闲显存最多且至少 26 GiB 空闲的单卡。`CUDA_VISIBLE_DEVICES` 可限制卡号。模型不跨卡拆分；26 GiB 是加载门槛，并不保证任意长输入都能推理。页面提供释放显存按钮，空闲 600 秒自动卸载。

## 检查与记录

```bash
"$VTC_PYTHON" tools/test_answer_parser.py -v
# 外部资产就绪后执行真实 CPU 输入检查
"$VTC_PYTHON" tools/validate_prompt_versions.py
```

`validate_prompt_versions.py` 使用随源码保留的原版基线，不依赖实验室 Git 历史。页面“仅预览模型输入”可核对实际 prompt、32 帧和可选渲染页。`tools/validate_inputs.py`、`tools/validate_metrics.py` 需先恢复完整资产。`tools/smoke_inference.py` 会实际使用 GPU，应按硬件资源情况手动执行。

每次推理记录到 `results/webui/<run_id>/`，包含版本、实际三段提示词、输入图像引用、原始输出、解析状态和生成后的 GT/评分。历史记录不覆盖。

机器上的真实路径可保存到本机 `.env.local`，启动脚本会读取该文件；它被 `.gitignore` 排除。GitHub 中不包含任何实验室地址、绝对路径或私有状态记录。

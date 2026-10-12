# VTC-R1-Glyph WebUI

原生 VTC-R1-Glyph 的场景输入实验界面：观察 RGB、物体描述和几何关系如何影响回答。

- 原始 RGB：沿用 Video-3D-LLM 的 uniform32 帧采样，只输入 RGB 和问题。
- 编号 RGB + 描述：相同 RGB 帧叠加预测物体编号，与已有物体描述对齐。
- 描述可选直接文字或官方 Glyph 渲染页；关系独立开关，文本可编辑。
- 保留原版与用户优化版提示词，默认优化版；附加说明、问题、输出格式均可修改。
- 默认原生思考开启，输出预算 8192 token；完整保留原始输出、格式修复记录、输入和生成后的评分。
- 支持 ScanQA、SQA3D、Scan2Cap、ScanRefer、Multi3DRefer 的逐样本浏览。

## 部署

先读 [部署说明](docs/SETUP.md)，准备模型、上游代码及有权限使用的场景资产，再运行 `bash run_webui.sh`。默认监听 `127.0.0.1:8100`。

本仓库仅发布源码，不包含模型权重、数据、描述资产、实验结果、日志或机器登录信息。单独安装界面依赖并不等于模型和数据已就绪。

## 结构

`app/engine.py` 构建输入并调用原生模型；`app/assets.py` 加载描述、编号照片和 Glyph 页；`app/answer_parser.py` 解析完整最终答案；`app/metrics.py` 复用官方评分器；`app/server.py` 提供 API 和串行队列；`web/` 为无需构建的前端。

## 使用边界

这是单次原生推理，不包含自动取帧工具或多轮思考压缩循环。GT 只用于展示与生成后评分。原始 RGB 没有世界几何与编号对照，grounding 输出只展示；编号模式的合法 ID 映射到预测框后计分。逐样本指标不代表完整 benchmark 成绩。

编号描述可能出错；没有显示编号不代表物体不存在。自动关系明确采用对齐世界坐标、Z 轴向上，只描述上下和近邻距离，不假定相机左右。输入超长会明确拒绝，不静默裁剪。

## 来源

界面结构与五任务数据读取参考 [Video-3D-LLM-WebUI](https://github.com/YuPeng-bit/Video-3D-LLM-WebUI)。模型与渲染复用 VTC-R1/Glyph，上游 3D 代码复用 [Video-3D-LLM](https://github.com/LaVi-Lab/Video-3D-LLM)，物体描述与资产接口来自 [VTC-3D-MLLM](https://github.com/YuPeng-bit/VTC-3D-MLLM)。模型、数据及上游组件遵循各自许可。

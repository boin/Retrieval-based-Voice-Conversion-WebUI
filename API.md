# Web Based API

## 需求

外部APP现在有批量使用RVC的Inference需求，目前的流程如下：

1. 外部APP调用 <http://ttd-stage:7867/> 的 /infer_change_voice 接口切换参考音模型
2. 外部APP调用 <http://ttd-stage:7867/> 的 /infer_convert 接口转换音频

## 待改善的问题

1. /infer_change_voice 接口切换参考音模型，是APP全局的，导致无法多用户并发
2. /infer_convert 接口转换音频，依赖于 /infer_change_voice 接口切换参考音模型，也无法多用户并发

## 概要方案

- 现状与问题
  - 现有接口 `infer_change_voice` 与 `infer_convert` 依赖共享的 VC 实例全局状态，造成不同请求间相互影响，无法并发。
- 目标
  - 支持每请求独立指定模型与参数；一个接口完成“指定模型+推理”；在 GPU 单进程下保持或提升当前性能。
- 架构与缓存
  - 新增 FastAPI 服务层（与现有 gradio 并存/解耦），提供纯后端 API。完全与gradio解耦。
  - 引入 ModelRegistry（LRU+TTL）缓存 ModelSession，会话内持有独立 VC/net_g/pipeline/tgt_sr/if_f0/version，并注入共享 hubert_model。
- 并发与稳定性
  - 推荐单进程单 worker，避免多进程重复占用显存；按“每模型”配置 Semaphore/Lock 控制并发，必要时串行。
- API 设计（核心）
  - POST `/api/v1/convert`：一次性传入 model、spk_id、audio_file/audio_url、f0_up_key、f0_method、index_path/index_rate、filter_radius、resample_sr、rms_mix_rate、protect、loudnorm，返回 sr/时长统计/音频结果（音频以 WAV PCM 流返回）。
    - 说明：服务端不再进行后处理（重采样、loudnorm、格式转码等），业务方在客户端侧自行处理。
  - POST `/api/v1/convert/batch`、GET `/api/v1/models`、GET `/health`。
  - 说明：服务启动阶段会预加载公共资源（如 Hubert 模型），而具体模型 ckpt 不会在启动时加载，按需懒加载。
- 关键实现
  - 将 `VC.get_vc` 的加载逻辑迁移至 ModelSession；共享 hubert_model；复用 `vc.vc_single` 作为 `session.convert`；LRU 淘汰释放显存；异常统一 `logger.exception`。
- 迁移步骤
  - 新增 `rvc_server/{registry.py, session.py, schemas.py, app.py}`；先并存后替换外部调用；
- 系统集成
  - 在原有Gradio系统中增加一个按钮，可通过HTTP调用新增的FastAPI接口的清空模型缓存接口，以释放api占用的全部显存。
- 部署
  - 复用现有Docker部署，并监听独立端口
- 风险与回滚
  - 线程安全/显存压力；保留原 gradio 接口或增强兼容路由做回滚。

## 期望的改善

1. 创建一个可以承载独立convert请求的API，该API的调用不受其它请求影响
2. better to have：一个接口直接制定参考模型+infer，无需分步骤。
3. better to have：性能需要保持目前的状态。

## 测试与脚本

- 所有测试与相关脚本已统一放置在 `tests/` 目录下：
  - `tests/run_smoke_test.py`：基础健康/转换冒烟测试
  - `tests/e2e_convert_test.py`：端到端单模型多次转换耗时观测
  - `tests/e2e_concurrency_test.py`：同/异模型并发转换对比（验证同模型串行、异模型并行的策略）
  - `tests/e2e_cache_abc_test.py`：顺序对 A/B/C 三模型做冷启动与重复访问，验证缓存加速
  - `tests/e2e_convert.sh`：Shell 版 E2E 转换脚本，便于快速复验（可设置 HOST/MODEL/AUDIO/LOOPS 等环境变量）

- 运行示例：

  ```bash
  # 在本仓库根目录执行
  # 冒烟测试
  python3 tests/run_smoke_test.py

  # 端到端多次转换
  python3 tests/e2e_convert_test.py

  # 并发对比（CONCURRENCY 可调）
  CONCURRENCY=3 python3 tests/e2e_concurrency_test.py

  # A/B/C 模型缓存验证（可通过 MODELS 指定三模型、REPEATS 指定重复轮次）
  python3 tests/e2e_cache_abc_test.py

  # Shell 版端到端
  bash tests/e2e_convert.sh
  # 或指定参数
  HOST=http://127.0.0.1:7868 MODEL="xxx.pth" AUDIO=logs/111/0_gt_wavs/1_0.wav LOOPS=3 bash tests/e2e_convert.sh
  ```

注意：

- 建议在单进程（uvicorn 单 worker）下验证缓存与同模型串行行为；
- 若使用多进程 workers，进程内缓存（Registry/Session）在进程之间不共享；
- Python 环境如使用本地 .conda/.venv，请确保已正确激活再运行测试。

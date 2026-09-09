# Local Voice Gateway (本地语音交互网关)

**Local Voice Gateway** 是一个专为 AI Agent（DeepSeek Harness、Claude、Cursor、OpenClaw 等）打造的**纯本地、低延迟物理交互网关**。

它负责接管本地操作系统的麦克风与音响，完成从「环境拾音 → 自适应 VAD → 关键词唤醒 / 声纹辨识 → 极速音效反馈 → 本地 STT/TTS」的全链路调度，并通过标准 REST API 与 WebSocket 事件流将语音感知与发声能力暴露给外部 Agent。

---

## 🌟 核心特性

1. **双模智能感知 (Dual-Mode Trigger)**：
   - **关键词唤醒模式 (`wake_word`)**：集成轻量级 OpenWakeWord 引擎。适合开麦打游戏、看电影、多人聚会等嘈杂环境，彻底消灭误触发。
   - **声纹被动常开模式 (`voiceprint_passive`)**：自适应 VAD 检测人声，只对已录入的主人声纹做出响应，享受无感自然的直觉对话。
   - **混合模式 (`hybrid`)**：唤醒词激活的同时识别说话人身份，实现一人一权的个性化交互。
2. **CAM++ 毫秒级声纹识别**：
   - 基于达摩院 CAM++ ONNX 模型与 Kaldi Fbank 特征提取，支持多用户声纹特征聚类与动态门限。
   - 识别结果自动附带 `[说话人:xxx]` 标签，可直接作为 Prompt 注入给大模型。
3. **极速感官反馈机制 (`ding.pcm`)**：
   - 在人声截断且身份确认的瞬间，内存直写双声道平滑包络线提示音到声卡缓冲区，大幅消灭大模型首字生成前的“等待焦虑感”。
4. **纯本地离线隐私**：
   - 语音识别基于本地 SenseVoiceSmall (FunASR)，语音合成支持本地离线 TTS，不产生任何云端 API 调用费用，保护绝对声音隐私。
5. **标准接口与 Agent / MCP 原生兼容**：
   - 提供 `/v1/audio/speak` 主动播报接口、`/v1/system/mode` 动态切模接口，以及 `/v1/events` 实时事件流，可秒级包装为 MCP Server 或 DSH 插件。

---

## 🎯 适用场景与已知边界说明 (Known Scope)

- **推荐场景**：个人书房、开发者工位、独立 NAS / 迷你主机 24 小时常驻。
- **打游戏连麦 / Discord 开黑场景**：
  - 如果您将本网关运行在日常打游戏的游戏 PC 上，**请务必将 `TRIGGER_MODE` 设置为 `wake_word`（关键词唤醒模式）**；
  - 在唤醒词模式下，后台唤醒引擎仅占用 ~1% CPU 极轻量监听唤醒词。无论您在游戏中与开黑队友如何高频交流，只要未喊出唤醒词，系统 100% 保持绝对静默，绝不打扰游戏；
  - 只有在独处、不连麦的沉浸交互场景下，才建议切换至 `voiceprint_passive`（声纹被动无感常开）。
- **多人同时说话场景**：当多人在同一空间内重叠说话时，混合声学波形会导致声纹向量衰减。系统默认策略为**置信度不足时静默丢弃**，以防在混杂声音中误执行敏感操作。

---

## 🛠️ 快速开始

### 1. 一键安装与环境初始化

- **Linux / macOS**：
  ```bash
  ./install.sh
  ```
- **Windows (CMD / PowerShell / 直接双击)**：
  ```cmd
  install.bat
  ```

脚本将自动完成：
- 基础系统音频依赖（Linux 自动检测 `alsa-utils`，Windows 自动配置 `PyAudio` 跨平台音频通道）；
- 虚拟环境初始化（自动创建并隔离 `.venv`）；
- **模型自动下载向导**：若本地缺失模型，会自动引导从阿里 ModelScope 镜像源极速下载 FunASR 与 MOSS-TTS 预训练权重（无需翻墙与账号）。

> 💡 你也可以随时通过专属工具按需管理与拉取模型：
> ```bash
> python utils/download_models.py --funasr   # 仅下载 FunASR 识别模型 (约 1.2GB)
> python utils/download_models.py --moss     # 仅下载 MOSS-TTS 合成模型 (约 2.5GB)
> python utils/download_models.py --all      # 一键拉取全部模型
> ```

### 2. 准备声纹采样 (可选)
- 启动网关后，可在前端 WebUI 面板一键引导录入 3 句短语完成采样；
- 也可在 `models/voice_profiles/<你的名字>/` 目录下直接放入 3~5 段个人纯净朗读的 `.wav` 音频（16kHz 单声道）。

### 3. 一键启动

- **Linux / macOS**：
  ```bash
  ./start.sh
  ```
- **Windows (CMD / PowerShell / 直接双击)**：
  ```cmd
  start.bat
  ```

网关默认监听 `http://0.0.0.0:8765`，并在控制台实时输出人声活动与事件。系统将自动根据当前操作系统（Linux ALSA vs Windows PyAudio）加载最佳声卡通道与设备防独占机制。

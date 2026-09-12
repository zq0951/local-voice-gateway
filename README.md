# Local Voice Gateway (本地语音交互网关)

**Local Voice Gateway** 是一个专为 AI Agent（DeepSeek Harness、Claude、Cursor、OpenClaw 等）打造的**纯本地、低延迟物理交互网关**。

它负责接管本地操作系统的麦克风与音响，完成从「环境拾音 → 自适应 VAD → 关键词唤醒 / 声纹辨识 → 极速音效反馈 → 本地 STT/TTS」的全链路调度，并通过标准 REST API 与 WebSocket 事件流将语音感知与发声能力暴露给外部 Agent。

---

## 🌟 核心特性

1. **双模智能感知 (Dual-Mode Trigger)**：
   - **关键词唤醒模式 (`wake_word`)**：集成轻量级 OpenWakeWord 引擎。适合开麦打游戏、看电影、多人聚会等嘈杂环境，彻底消灭误触发。
   - **声纹被动常开模式 (`voiceprint_passive`)**：自适应 VAD 检测人声，只对已录入的主人声纹做出响应，享受无感自然的直觉对话。
   - **混合模式 (`hybrid`)**：双通道并发监听（二选一触发）：既支持说唤醒词（如“Hey Jarvis”）激活，也支持主人直接说话免唤醒直触发，兼顾全场景极速响应。
2. **CAM++ 毫秒级声纹识别**：
   - 基于达摩院 CAM++ ONNX 模型与 Kaldi Fbank 特征提取，支持多用户声纹特征聚类与动态门限。
   - 识别结果自动附带 `[说话人:xxx]` 标签，可直接作为 Prompt 注入给大模型。
3. **极速感官反馈机制 (`ding.pcm`)**：
   - 在人声截断且身份确认的瞬间，内存直写双声道平滑包络线提示音到声卡缓冲区，大幅消灭大模型首字生成前的“等待焦虑感”。
4. **纯本地离线隐私 (免 Docker 原生跨平台)**：
   - 语音识别原生支持本地 SenseVoiceSmall (FunASR) 进程内直接推理，彻底摆脱 Docker 依赖与网络端口开销，Windows / macOS / Linux 均可通过 Python 一键拉起；
   - 同时向下兼容 WebSocket 远程识别（支持将 ASR 跑在独立 Docker 或远端 GPU 算力机上）；
   - 语音合成支持本地离线 TTS，不产生任何云端 API 调用费用，保护绝对声音隐私。
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

网关默认监听 `http://127.0.0.1:8765`（仅本机访问；LAN/远程或 Docker 场景请设 `GATEWAY_HOST=0.0.0.0`），并在控制台实时输出人声活动与事件。系统将自动根据当前操作系统（Linux ALSA vs Windows PyAudio）加载最佳声卡通道与设备防独占机制。

---

## 🐳 Docker 容器化部署

针对 Linux / NAS / 迷你主机服务器常驻场景，项目提供了一套低延迟、声卡透传的容器化方案。

### 1. 前置准备 (拉取基础模型)

由于遵循开源合规性原则，模型权重不随代码打包进 Git，容器启动时将直接挂载宿主机模型。**在首次启动容器前，请务必在宿主机拉取基础模型**：

```bash
# 至少拉取 CAM++ 声纹与唤醒词模型 (约 30MB，避免宿主机因缺失文件被 Docker 误创建为空目录)
python utils/download_models.py --campplus --wakeword

# 若需在容器内进行纯本地语音识别与离线合成，请一并拉取大模型：
# python utils/download_models.py --funasr --moss
```

### 2. 构建镜像

在项目根目录下构建针对 CPU 深度优化的语音网关镜像（已预配置国内清华镜像源与 CPU 专用轻量 PyTorch）：

```bash
docker compose build
# 或者单命令原生构建：
# docker build -t local-voice-gateway:latest .
```

### 3. 启动容器

直接在项目根目录下启动网关单容器（内置纯本地离线 ASR、声纹与 TTS 推理）：

```bash
docker compose up
# 或后台守护运行：
# docker compose up -d
```

### 4. 容器状态检查与日志查看

```bash
# 查看网关实时运行日志与声学事件流
docker compose logs -f voice-gateway

# 停止容器
docker compose down
```

### 5. 容器化核心配置与持久化说明

- **模型与持久化挂载 (`volumes`)**：
  - `./models/campplus.onnx` & `./models/wakeword`：宿主机模型文件直接透传进入容器，保障混合双模（Hybrid）开箱即用；
  - `./models/config`：挂载运行时配置目录，容器内通过 REST API 修改的触发模式与自动朗读设置在容器重启/重建后仍完整持久化；
  - `./models/voice_profiles`：挂载用户生物声纹特征库，录入与采样数据在宿主机持久存储；
  - 移除了旧版全目录 `- .:/app` 挂载，彻底消除宿主机工作区对 Docker 镜像构建产物的覆盖遮蔽。
- **硬件声卡透传 (`devices: ["/dev/snd:/dev/snd"]` + `privileged: true`)**：
  使容器具备直接操作宿主机麦克风阵列与音箱声卡的物理权限，零损耗无二次重采样。
- **共享 ALSA 声卡配置 (`/etc/asound.conf:/etc/asound.conf:ro`)**：
  自动复用宿主机校准好的录音增益与播音通道（例如 USB 麦克风音响一体机），容器内无需额外调优。
- **网络直通模式 (`network_mode: "host"`)**：
  消除 Docker Bridge 虚拟网桥 NAT 开销与端口映射延迟，使前端 WebUI、宿主机 STT 与外部 Agent 能以毫秒级直连 `http://127.0.0.1:8765`。

---

## 📡 对外接口契约 (API & Event Contract)

Local Voice Gateway 遵循开箱即用的 REST + WebSocket 标准契约。外部 Agent、前端插件（如 DeepSeek Harness UI）或第三方系统均可基于此协议进行双向集成。

### 1. 系统配置与控制 REST API

| 端点 (Endpoint) | 方法 | 请求体 (Payload) | 返回示例 (Response) | 说明 |
|---|---|---|---|---|
| `/v1/system/status` | `GET` | 无 | `{"status":"running","trigger_mode":"hybrid","audio_duplex_mode":"half","auto_speak":true,"registered_speakers":["peter"]}` | 获取网关运行状态、模式、双工配置与已注册声纹 |
| `/v1/system/mode` | `POST` | `{"mode": "hybrid"}` | `{"status":"ok","current_mode":"hybrid"}` | 动态切换触发模式 (`wake_word` / `voiceprint_passive` / `hybrid`) |
| `/v1/system/autospeak` | `POST` | `{"enabled": true}` | `{"status":"ok","auto_speak":true}` | 动态切换回复自动朗读开关 (闭嘴开关联动) |
| `/v1/system/wakeword` | `GET` | 无 | `{"current_model":"hey_jarvis","threshold":0.5,"available_models":["hey_jarvis","alexa"]}` | 查询当前唤醒词模型及可用模型列表 |
| `/v1/system/wakeword` | `POST` | `{"model": "hey_jarvis", "threshold": 0.5}` | `{"status":"ok","current_model":"hey_jarvis"}` | 热切换唤醒词模型与置信度门限 |

### 2. 物理音频播报与反馈 REST API

| 端点 (Endpoint) | 方法 | 请求体 (Payload) | 返回示例 (Response) | 说明 |
|---|---|---|---|---|
| `/v1/audio/speak` | `POST` | `{"text": "你好，我是 Jarvis"}` | `{"status":"queued","task_id":"..."}` | 放入抢占式 TTS 合成并排队朗读，毫秒级响应不阻塞网络 |
| `/v1/audio/stop` | `POST` | 无 | `{"status":"ok"}` | 立即中断当前正在朗读的音频并清空队列 (闭嘴) |
| `/v1/audio/ding` | `POST` | 无 | `{"status":"ok"}` | 内存直写触发一次低延迟 `ding.pcm` 提示音 |

### 3. 声纹识别与向导式录入 REST API

| 端点 (Endpoint) | 方法 | 请求体 (Payload) | 返回示例 (Response) | 说明 |
|---|---|---|---|---|
| `/v1/voiceprint/profiles` | `GET` | 无 | `{"status":"ok","profiles":[{"name":"peter","sample_count":5}]}` | 列出声纹库中已录入的所有说话人信息 |
| `/v1/voiceprint/{name}` | `DELETE` | 无 | `{"status":"ok","deleted":"peter"}` | 删除指定说话人并热重载声纹特征库 |
| `/v1/voiceprint/enroll/start` | `POST` | `{"name": "peter", "steps": 5}` | `{"status":"ok","session":{"session_id":"s1","total_steps":5,...}}` | 开启多步短语引导录入会话，通知后台挂起麦克风占用 |
| `/v1/voiceprint/enroll/record_step` | `POST` | `{"session_id": "s1"}` | `{"status":"ok","success":true,"current_step":1,...}` | 执行单步短语麦克风录制，提取特征并推进引导步骤 |
| `/v1/voiceprint/enroll/finish` | `POST` | `{"session_id": "s1"}` | `{"status":"ok","saved_samples":5}` | 完成向导，执行离群特征剔除，持久化并热重载声纹库 |
| `/v1/voiceprint/enroll/abort` | `POST` | `{"session_id": "s1"}` | `{"status":"ok"}` | 放弃本次录入并安全释放声卡资源 |

### 4. 实时双向 WebSocket 事件流 (`/v1/events`)

客户端连接 `ws://<host>:8765/v1/events` 即可实时接收网关声学事件广播（支持多客户端并发监听）：

| 事件名称 (`event`) | 附带数据 (`data`) | 触发时机与业务含义 |
|---|---|---|
| `speech_recognized` | `{"speaker": "peter", "text": "打开客厅大灯", "emotion": "NEUTRAL"}` | 用户语音识别完成，附带说话人身份与情感标签 |
| `wake_word_detected` | `{"model": "hey_jarvis", "score": 0.85}` | 关键词唤醒命中 |
| `playback_started` / `tts_playing` | 无 | 本地音箱开始播放回复音频 |
| `playback_idle` / `playback_stopped` / `tts_idle` | 无 | 本地音箱播报结束，恢复待机或聆听态 |
| `tts_generating` | 无 | 本地 TTS 正在执行语音合成 |
| `mode_changed` | `{"new_mode": "wake_word"}` | 网关触发模式被修改 |
| `autospeak_changed` | `{"enabled": true}` | 自动朗读回复开关状态变更 |
| `enroll_started` / `enroll_step_recorded` / `enroll_finished` / `enroll_aborted` | 会话详细进度数据 | 声纹录入向导各阶段状态广播 |

---

## 🔌 与 DeepSeek Harness (DSH) 插件对接

网关与 DSH 前端插件（`packages/client/ui-voice`）之间采用纯解耦的分布式架构：

```text
┌─────────────────────────────────────────────────────────────┐
│ 🖥️ 本地语音网关 (Python Daemon, :8765)                        │
│ 拥有声卡物理控制权: 麦克风 / 音箱 / VAD / 唤醒词 / 声纹 / STT / TTS   │
└───────────────┬───────────────────────────────▲─────────────┘
                │ REST API (模式切换/朗读/向导)      │ 
                │ WebSocket /v1/events (事件广播) │ 
                ▼                               │
┌───────────────────────────────────────────────┴─────────────┐
│ 🌐 DSH Web Client (浏览器运行: dsh-client-ui-voice 插件)      │
│  - 监听 speech_recognized → 注入当前活动会话 Prompt             │
│  - 监听 assistant/message → 调用 /v1/audio/speak 物理播报   │
│  - 状态权威源 (SSOT): 模式与自动朗读状态始终以网关为准              │
└─────────────────────────────────────────────────────────────┘
```

### 1. 插件寻址逻辑与网络要求
- **自动寻址**：DSH 浏览器插件默认使用当前访问地址的 `window.location.hostname + ':8765'` 连接网关（如打开 `http://127.0.0.1:3080` 时自动寻找 `http://127.0.0.1:8765`）；
- **自定义配置覆盖**：
  - 用户可在插件下拉菜单直接点击「修改」网关地址并保存；
  - 也可通过浏览器控制台设置 `localStorage.setItem('dsh.voice.gateway_url', 'http://<IP>:8765')`；
- **跨机 / 远程连接**：如果 DSH 运行在云端或远端服务器，浏览器与本地网关所在设备必须能够建立网络直连（如在同局域网内，或通过 FRP / Tailscale 等反向代理暴露网关端口，同时网关配置 `GATEWAY_HOST=0.0.0.0`）。

### 2. MCP (Model Context Protocol) 扩展通道 (`mcp_server.py`)
除 REST / WS 外，网关还内置了基于 FastMCP 的标准 stdio MCP Server，可作为 DSH 工具或独立接入 Claude Desktop、Cursor 等宿主：

```bash
# 启动 MCP Server
python mcp_server.py
```

暴露的 5 个原生工具：
- `speak(text)`: 排队播报文本给用户听；
- `stop_speaking()`: 立即打断当前播报 (闭嘴)；
- `play_ding()`: 在扬声器上播放提示音；
- `set_mode(mode)`: 动态切换触发模式 (`wake_word` / `voiceprint_passive` / `hybrid`)；
- `get_status()`: 获取当前网关硬件与运行状态。

---

## 📄 开源许可证与模型版权声明 (License & Model Disclaimer)

### 1. 代码许可证 (Software License)
本项目工程代码（包括调度逻辑、网关服务、前后端契约、工具脚本等）基于 **[MIT License](LICENSE)** 开源，与 DeepSeek Harness (DSH) 插件生态保持一致。

### 2. 模型权重与二进制资产分离原则 (Model Weight Separation)
**特别说明：模型权重 ≠ 本项目代码。**
- 本 Git 仓库**严格遵守开源合规性原则，仅分发源代码、接口契约与下载管理脚本，不包含任何第三方预训练模型权重（`.onnx` / `.pt` / `.bin`）及二进制音频**；
- 运行时所需的声音模型统一通过项目提供的 `utils/download_models.py` 脚本或 `install.sh` 引导程序由用户在本地按需下载；
- 提示音 `ding.pcm` 由算法在本地按当前声卡硬件物理规格自适应合成，不作为静态二进制预先打包。

### 3. 第三方模型版权归属与许可清单 (Third-Party Models)
网关集成的各底层 AI 模型均属于其原始研发机构与作者所有。用户在下载与使用相关模型时，须严格遵守其各自的开源协议与使用规范：

| 模块类别 | 模型名称 | 研发机构 / 贡献者 | 原始开源许可证 | 官方来源 / 规范 |
|---|---|---|---|---|
| **声纹识别** | CAM++ (ONNX) | 阿里巴巴达摩院 (Alibaba DAMO) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | [ModelScope 官方模型库](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common) |
| **关键词唤醒** | OpenWakeWord Models | David Scripka | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | [openWakeWord GitHub](https://github.com/dscripka/openWakeWord) |
| **语音识别 & VAD** | SenseVoiceSmall & FSMN-VAD | 阿里巴巴达摩院 / FunASR | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | [FunASR ModelScope](https://modelscope.cn/models/iic/SenseVoiceSmall) |
| **语音合成** | MOSS-TTS-Nano | 复旦大学自然语言处理实验室 (OpenMOSS) | [MOSS-TTS 社区开源许可 / Apache 2.0](https://github.com/OpenMOSS/MOSS-TTS-Nano) | [OpenMOSS GitHub](https://github.com/OpenMOSS/MOSS-TTS-Nano) |

使用者需保证遵守上述模型各自的许可限制。因超出许可范围使用或二次商用模型所产生的法律风险，由使用者自行承担。


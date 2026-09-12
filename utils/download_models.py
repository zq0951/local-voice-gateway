#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Voice Gateway - 官方模型一键高速下载器
支持免登录、免科学上网从 ModelScope (魔搭社区) 与镜像源高速拉取：
- FunASR (SenseVoiceSmall + FSMN-VAD)
- MOSS-TTS-Nano 离线合成模型
- CAM++ 声纹识别 ONNX 模型
- OpenWakeWord 唤醒词基础与关键词模型
"""

import os
import sys
import argparse
import subprocess
import shutil
import tempfile
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def ensure_modelscope_installed():
    """确保当前 Python 环境已安装 modelscope 库"""
    try:
        import modelscope
        return True
    except ImportError:
        print("📦 正在安装 modelscope 高速下载组件...")
        ret = subprocess.call([sys.executable, "-m", "pip", "install", "modelscope", "-q"])
        return ret == 0

def ensure_onnx_installed():
    """确保当前 Python 环境已安装 onnx 及 onnxscript 导出库"""
    missing = []
    try:
        import onnx
    except ImportError:
        missing.append("onnx>=1.16.0")
    try:
        import onnxscript
    except ImportError:
        missing.append("onnxscript>=0.1.0")

    if missing:
        print(f"📦 正在自动配置 ONNX 导出组件 ({' '.join(missing)})...")
        ret = subprocess.call([sys.executable, "-m", "pip", "install", *missing, "-q"])
        return ret == 0
    return True

def download_file_with_fallback(urls, dest_path, desc=""):
    """多源回退下载单个文件"""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
        print(f"✅ {desc or os.path.basename(dest_path)} 已存在，跳过下载")
        return True

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".tmp"

    for url in urls:
        print(f"⏳ 正在拉取 {desc or os.path.basename(dest_path)}: {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LocalVoiceGateway-Downloader/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(temp_path, "wb") as f:
                shutil.copyfileobj(resp, f)
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1024:
                shutil.move(temp_path, dest_path)
                print(f"✅ {desc or os.path.basename(dest_path)} 下载成功")
                return True
        except Exception as e:
            print(f"⚠️ 从 {url} 下载失败: {e}")
            if os.path.exists(temp_path):
                try: os.remove(temp_path)
                except: pass

    print(f"❌ {desc or os.path.basename(dest_path)} 所有下载源均失败")
    return False

def download_campplus(target_path=None):
    """从魔搭 (ModelScope) 官方拉取 CAM++ 声纹识别模型原生权重并在本地转换为 ONNX 推理格式"""
    if not target_path:
        target_path = os.path.join(MODELS_DIR, "campplus.onnx")

    print("\n" + "=" * 60)
    print("📥 [CAM++] 正在配置 CAM++ 声纹识别模型 (ModelScope 官方源)...")
    print("=" * 60)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1024 * 1024:
        print("✅ CAM++ 声纹 ONNX 模型已就绪，跳过下载")
        return True

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    temp_bin = os.path.join(MODELS_DIR, "campplus_cn_common.bin")

    # 1. 优先通过 ModelScope 官方 SDK 下载官方权重
    bin_path = None
    if ensure_modelscope_installed():
        try:
            from modelscope.hub.file_download import model_file_download
            print("⏳ 正在从魔搭官方 (iic/speech_campplus_sv_zh-cn_16k-common) 下载原生权重...")
            bin_path = model_file_download('iic/speech_campplus_sv_zh-cn_16k-common', 'campplus_cn_common.bin')
        except Exception as e:
            print(f"⚠️ ModelScope Hub SDK 异常: {e}，切入官方直链下载...")

    # 2. 备用官方直链 (ModelScope 官方 CDN)
    if not bin_path or not os.path.exists(bin_path):
        urls = [
            "https://www.modelscope.cn/api/v1/models/iic/speech_campplus_sv_zh-cn_16k-common/repo?Revision=master&FilePath=campplus_cn_common.bin",
            "https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/resolve/master/campplus_cn_common.bin",
        ]
        if download_file_with_fallback(urls, temp_bin, "CAM++ 官方模型权重"):
            bin_path = temp_bin

    if not bin_path or not os.path.exists(bin_path):
        print("❌ 未能从魔搭官方获取到 CAM++ 模型权重")
        return False

    # 3. 将官方 PyTorch 权重在本地导出为高性能 CPU 推理引擎 (ONNX)
    ensure_onnx_installed()
    print("⚙️ 官方权重就绪，正在本地导出为高性能轻量 ONNX 引擎 (约需 1~2 秒)...")
    try:
        import torch
        from funasr.models.campplus.model import CAMPPlus

        model = CAMPPlus(feat_dim=80, embedding_size=192)
        state_dict = torch.load(bin_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        dummy_input = torch.randn(1, 100, 80)
        export_kwargs = {
            "input_names": ["fbank"],
            "output_names": ["embedding"],
            "dynamic_axes": {"fbank": {1: "time"}, "embedding": {0: "batch"}},
            "opset_version": 14,
        }
        try:
            # 明确指定 dynamo=False 使用经典 TorchScript 导出器，避免触发 onnxscript 缺失
            torch.onnx.export(
                model,
                dummy_input,
                target_path,
                dynamo=False,
                **export_kwargs,
            )
        except TypeError:
            torch.onnx.export(
                model,
                dummy_input,
                target_path,
                **export_kwargs,
            )
        except Exception as ex:
            if "onnxscript" in str(ex):
                print("📦 检测到导出器需要 onnxscript，正在自动配置...")
                subprocess.call([sys.executable, "-m", "pip", "install", "onnxscript", "-q"])
                torch.onnx.export(
                    model,
                    dummy_input,
                    target_path,
                    **export_kwargs,
                )
            else:
                raise ex

        if os.path.exists(target_path) and os.path.getsize(target_path) > 1024 * 1024:
            print("✅ CAM++ 声纹 ONNX 模型构建成功 (纯本地/免外部托管)！")
            return True
        else:
            print("❌ 本地 ONNX 构建异常，产物大小不合预期")
            return False
    except Exception as e:
        print(f"❌ 本地 ONNX 转换失败: {e}")
        return False

def download_wakewords(target_dir=None):
    """精准下载 OpenWakeWord 核心必备组件 (仅下载 3 个必备 ONNX 模型，杜绝多余模型与 tflite)"""
    if not target_dir:
        target_dir = os.path.join(MODELS_DIR, "wakeword")
    os.makedirs(target_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("📥 [OpenWakeWord] 正在配置核心唤醒词组件...")
    print("=" * 60)

    # 必需的 3 个基础与核心模型列表 (按需精准拉取，杜绝无用冗余)
    models = {
        "melspectrogram.onnx": [
            "https://ghproxy.net/https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
            "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
            "https://huggingface.co/dscripka/openwakeword/resolve/main/melspectrogram.onnx",
        ],
        "embedding_model.onnx": [
            "https://ghproxy.net/https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
            "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
            "https://huggingface.co/dscripka/openwakeword/resolve/main/embedding_model.onnx",
        ],
        "hey_jarvis_v0.1.onnx": [
            "https://ghproxy.net/https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx",
            "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx",
            "https://huggingface.co/dscripka/openwakeword/resolve/main/hey_jarvis_v0.1.onnx",
        ],
        "alexa_v0.1.onnx": [
            "https://ghproxy.net/https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/alexa_v0.1.onnx",
            "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/alexa_v0.1.onnx",
            "https://huggingface.co/dscripka/openwakeword/resolve/main/alexa_v0.1.onnx",
        ],
    }

    all_ok = True
    for filename, urls in models.items():
        dest = os.path.join(target_dir, filename)
        if not os.path.exists(dest):
            ok = download_file_with_fallback(urls, dest, f"唤醒词组件: {filename}")
            if not ok:
                all_ok = False
        else:
            print(f"✅ 唤醒词组件已存在: {filename}")

    # 清理历史无用的 .tflite 和非本项目唤醒词
    cleaned_count = 0
    for f in os.listdir(target_dir):
        if f.endswith(".tflite") or (f.endswith(".onnx") and f not in models and f != "silero_vad.onnx"):
            try:
                os.remove(os.path.join(target_dir, f))
                cleaned_count += 1
            except OSError:
                pass
    if cleaned_count > 0:
        print(f"🧹 已自动清理 {cleaned_count} 个历史冗余唤醒词与 .tflite 格式文件，仅保留核心按需组件！")

    return all_ok

def download_funasr(target_dir=None):
    """从 ModelScope 下载 FunASR (SenseVoiceSmall + VAD) 识别模型"""
    if not target_dir:
        target_dir = os.path.join(MODELS_DIR, "funasr")
    os.makedirs(target_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("📥 [FunASR] 正在从 ModelScope 镜像源高速下载语音识别模型...")
    print("=" * 60)

    if not ensure_modelscope_installed():
        print("❌ modelscope 安装失败，请手动执行: pip install modelscope")
        return False

    from modelscope import snapshot_download

    # 1. 下载核心识别模型 SenseVoiceSmall (约 1.2GB)
    sv_dir = os.path.join(target_dir, "SenseVoiceSmall")
    if not os.path.exists(os.path.join(sv_dir, "model.onnx")) and not os.path.exists(os.path.join(sv_dir, "model.pt")):
        print("⏳ 正在拉取 SenseVoiceSmall 核心语音大模型 (支持中/英/粤/日/韩/富文本情绪)...")
        snapshot_download('iic/SenseVoiceSmall', local_dir=sv_dir)
        print("✅ SenseVoiceSmall 下载完成")
    else:
        print("✅ SenseVoiceSmall 模型已存在，跳过下载")

    # 2. 下载 FSMN-VAD 人声切片模型 (约 10MB)
    vad_dir = os.path.join(target_dir, "speech_fsmn_vad_zh-cn-16k-common-onnx")
    if not os.path.exists(vad_dir):
        print("⏳ 正在拉取 FSMN-VAD 人声切片模型...")
        snapshot_download('iic/speech_fsmn_vad_zh-cn-16k-common-onnx', local_dir=vad_dir)
        print("✅ FSMN-VAD 下载完成")
    else:
        print("✅ FSMN-VAD 模型已存在，跳过下载")

    print("🎉 FunASR 语音识别模型库准备就绪！")
    return True

def download_moss_tts(target_dir=None):
    """从 ModelScope / GitHub 下载复旦 MOSS-TTS-Nano 离线合成模型"""
    if not target_dir:
        target_dir = os.path.join(MODELS_DIR, "moss_tts")
    os.makedirs(target_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("📥 [MOSS-TTS] 正在配置 MOSS-TTS 离线语音合成模型...")
    print("=" * 60)

    # 1. 检查并拉取 MOSS-TTS-Nano 运行时代码
    infer_script = os.path.join(target_dir, "infer.py")
    if not os.path.exists(infer_script):
        print("⏳ 正在拉取 MOSS-TTS-Nano 运行时代码仓库...")
        git_urls = [
            "https://github.com/OpenMOSS/MOSS-TTS-Nano.git",
            "https://gitclone.com/github.com/OpenMOSS/MOSS-TTS-Nano.git",
            "https://ghproxy.net/https://github.com/OpenMOSS/MOSS-TTS-Nano.git"
        ]
        cloned_ok = False
        temp_dir = tempfile.mkdtemp(prefix="moss_clone_")
        try:
            for url in git_urls:
                try:
                    res = subprocess.run(
                        ["git", "clone", "--depth=1", url, temp_dir],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60
                    )
                    if res.returncode == 0 and os.path.exists(os.path.join(temp_dir, "infer.py")):
                        cloned_ok = True
                        break
                    else:
                        print(f"⚠️ 从 {url} 拉取未完成，尝试备用源...")
                except Exception as ex:
                    print(f"⚠️ 克隆源异常 ({url}): {ex}")

            if cloned_ok:
                for item in os.listdir(temp_dir):
                    if item == ".git":
                        continue
                    s = os.path.join(temp_dir, item)
                    d = os.path.join(target_dir, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
                print("✅ MOSS-TTS-Nano 运行时代码合并完成")
            else:
                print("⚠️ 自动克隆代码失败，后续可手动将 MOSS-TTS-Nano 仓库代码放置于 models/moss_tts 目录")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    if not ensure_modelscope_installed():
        print("❌ modelscope 安装失败，请手动执行: pip install modelscope")
        return False

    from modelscope import snapshot_download

    # 2. 下载 MOSS-TTS-Nano 与 Audio-Tokenizer 权重缓存 (约 2.5GB)
    cache_dir = os.path.join(target_dir, "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    print("⏳ 正在从 ModelScope 拉取 MOSS-TTS-Nano 声学模型与 Tokenizer 预训练权重...")
    try:
        snapshot_download('openmoss/MOSS-TTS-Nano', cache_dir=cache_dir)
        snapshot_download('openmoss/MOSS-Audio-Tokenizer-Nano', cache_dir=cache_dir)
        print("🎉 MOSS-TTS 离线语音合成模型准备就绪！")
        return True
    except Exception as e:
        print(f"⚠️ MOSS-TTS 权重下载异常: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Local Voice Gateway 模型一键高速下载器")
    parser.add_argument("--all", action="store_true", help="一键下载所有预训练模型 (ASR + TTS + 声纹 + 唤醒词)")
    parser.add_argument("--funasr", action="store_true", help="仅下载 FunASR 语音识别模型")
    parser.add_argument("--moss", action="store_true", help="仅下载 MOSS-TTS 语音合成模型")
    parser.add_argument("--campplus", action="store_true", help="仅下载 CAM++ 声纹识别 ONNX 模型")
    parser.add_argument("--wakeword", action="store_true", help="仅下载 OpenWakeWord 唤醒词模型")
    args = parser.parse_args()

    if not any([args.all, args.funasr, args.moss, args.campplus, args.wakeword]):
        print("=" * 60)
        print("Local Voice Gateway 模型管理向导")
        print("=" * 60)
        print("1) 下载全部基础模型 (ASR + 声纹 + 唤醒词，推荐)")
        print("2) 下载 FunASR 语音识别模型 (约 1.2GB)")
        print("3) 下载 CAM++ 声纹识别 ONNX 模型 (约 26MB)")
        print("4) 下载 OpenWakeWord 唤醒词模型 (约 5MB)")
        print("5) 下载 MOSS-TTS 离线语音合成模型 (约 2.5GB，可选)")
        print("6) 退出")
        choice = input("请输入选项 [1-6]: ").strip()
        if choice == "1":
            download_campplus()
            download_wakewords()
            download_funasr()
        elif choice == "2":
            download_funasr()
        elif choice == "3":
            download_campplus()
        elif choice == "4":
            download_wakewords()
        elif choice == "5":
            download_moss_tts()
        else:
            print("已退出")
            sys.exit(0)
        return

    if args.all or args.campplus:
        download_campplus()
    if args.all or args.wakeword:
        download_wakewords()
    if args.all or args.funasr:
        download_funasr()
    if args.all or args.moss:
        download_moss_tts()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Voice Gateway - 官方模型一键高速下载器
支持免登录、免科学上网从 ModelScope (魔搭社区) 高速拉取 FunASR 与 MOSS-TTS 预训练权重。
"""

import os
import sys
import argparse
import subprocess

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
        # 优先使用国内镜像或官方 github
        cmd = f"git clone --depth=1 https://github.com/OpenMOSS/MOSS-TTS-Nano.git '{target_dir}'"
        ret = os.system(cmd)
        if ret != 0 or not os.path.exists(infer_script):
            print("⚠️ GitHub 克隆可能受网络影响，尝试通过镜像拉取...")
            os.system(f"git clone --depth=1 https://hub.fastgit.xyz/OpenMOSS/MOSS-TTS-Nano.git '{target_dir}' 2>/dev/null || git clone --depth=1 https://gitclone.com/github.com/OpenMOSS/MOSS-TTS-Nano.git '{target_dir}'")

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
    parser.add_argument("--all", action="store_true", help="一键下载所有预训练模型 (FunASR + MOSS-TTS)")
    parser.add_argument("--funasr", action="store_true", help="仅下载 FunASR 语音识别模型")
    parser.add_argument("--moss", action="store_true", help="仅下载 MOSS-TTS 语音合成模型")
    args = parser.parse_args()

    if not any([args.all, args.funasr, args.moss]):
        print("Local Voice Gateway 模型管理向导")
        print("1) 下载 FunASR 语音识别模型 (约 1.2GB，必选)")
        print("2) 下载 MOSS-TTS 语音合成模型 (约 2.5GB，可选)")
        print("3) 下载全部模型 (约 3.7GB)")
        print("4) 退出")
        choice = input("请输入选项 [1-4]: ").strip()
        if choice == "1":
            download_funasr()
        elif choice == "2":
            download_moss_tts()
        elif choice == "3":
            download_funasr()
            download_moss_tts()
        else:
            print("已取消操作")
            sys.exit(0)
        return

    if args.all or args.funasr:
        download_funasr()
    if args.all or args.moss:
        download_moss_tts()

if __name__ == "__main__":
    main()

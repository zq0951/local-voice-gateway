import asyncio
import httpx

import os
from mcp.server.fastmcp import FastMCP

import config

mcp = FastMCP("LocalVoiceGateway")
BASE_URL = os.getenv("GATEWAY_URL", f"http://{getattr(config, 'GATEWAY_HOST', '127.0.0.1')}:{config.GATEWAY_PORT}")

@mcp.tool()
async def speak(text: str) -> str:
    """让本地音箱使用拟真音色朗读一段文本给用户听"""
    if not text.strip():
        return "错误: 朗读文本不能为空"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{BASE_URL}/v1/audio/speak", json={"text": text})
            if resp.status_code == 200:
                return f"✅ 音箱已排队播报: '{text}'"
            return f"❌ 播报请求失败: HTTP {resp.status_code}"
    except Exception as e:
        return f"❌ 连接本地语音网关失败: {e}"

@mcp.tool()
async def stop_speaking() -> str:
    """立即打断音箱当前的播报并清空语音播放队列 (闭嘴)"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{BASE_URL}/v1/audio/stop")
            if resp.status_code == 200:
                return "✅ 已停止音箱播报"
            return f"❌ 停止播报失败: HTTP {resp.status_code}"
    except Exception as e:
        return f"❌ 连接本地语音网关失败: {e}"

@mcp.tool()
async def play_ding() -> str:
    """在本地物理扬声器上播放一次提示音 (ding.pcm)，用于提醒用户注意或标记工作流完成"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{BASE_URL}/v1/audio/ding")
            return "✅ 提示音已播放"
    except Exception as e:
        return f"❌ 播放提示音失败: {e}"

@mcp.tool()
async def set_mode(mode: str) -> str:
    """
    动态切换语音网关的工作模式:
    - 'wake_word': 仅关键词唤醒 (打游戏连麦、看电影等嘈杂环境首选，绝不误触)
    - 'voiceprint_passive': 声纹被动常开 (独处书房首选，只认主人声音，免喊唤醒词)
    - 'hybrid': 混合模式 (关键词激活 + 声纹识别身份)
    """
    valid = ["wake_word", "voiceprint_passive", "hybrid"]
    if mode not in valid:
        return f"错误: 模式无效，仅支持 {valid}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{BASE_URL}/v1/system/mode", json={"mode": mode})
            if resp.status_code == 200:
                return f"✅ 语音网关模式已切换为: {mode}"
            return f"❌ 切换失败: HTTP {resp.status_code}"
    except Exception as e:
        return f"❌ 连接本地网关失败: {e}"

@mcp.tool()
async def get_gateway_status() -> dict:
    """获取本地语音网关的运行状态、当前触发模式以及已录入声纹的主人列表"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{BASE_URL}/v1/system/status")
            if resp.status_code == 200:
                return resp.json()
            return {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run(transport="stdio")

import os
import re
import logging
import subprocess

logger = logging.getLogger("LocalVoiceGateway")

class AudioDeviceManager:
    """
    音频硬件智能探测与 ALSA 自愈管理器
    - 自动发现可用麦克风与扬声器
    - 自动生成防独占 (dmix/dsnoop) 与自动重采样 (plug) 的 asound.conf
    - 自动解除静音，避免环境差异导致的运行失败
    """

    @staticmethod
    def get_capture_devices():
        """枚举系统所有录音输入设备"""
        devices = []
        try:
            out = subprocess.check_output(["arecord", "-l"], stderr=subprocess.DEVNULL).decode("utf-8")
            # 解析: card 3: M1A [EMEET OfficeCore M1A], device 0: USB Audio [USB Audio]
            pattern = re.compile(r"card (\d+):\s+([\w\-]+)\s+\[(.*?)\],\s+device (\d+):\s+(.*)")
            for line in out.splitlines():
                m = pattern.search(line)
                if m:
                    card_num, card_id, card_desc, dev_num, dev_desc = m.groups()
                    devices.append({
                        "card_num": int(card_num),
                        "card_id": card_id,
                        "card_desc": card_desc,
                        "device_num": int(dev_num),
                        "device_desc": dev_desc,
                        "type": "capture"
                    })
        except Exception as e:
            logger.warning(f"获取录音设备列表失败: {e}")
        return devices

    @staticmethod
    def get_playback_devices():
        """枚举系统所有播放输出设备"""
        devices = []
        try:
            out = subprocess.check_output(["aplay", "-l"], stderr=subprocess.DEVNULL).decode("utf-8")
            pattern = re.compile(r"card (\d+):\s+([\w\-]+)\s+\[(.*?)\],\s+device (\d+):\s+(.*)")
            for line in out.splitlines():
                m = pattern.search(line)
                if m:
                    card_num, card_id, card_desc, dev_num, dev_desc = m.groups()
                    devices.append({
                        "card_num": int(card_num),
                        "card_id": card_id,
                        "card_desc": card_desc,
                        "device_num": int(dev_num),
                        "device_desc": dev_desc,
                        "type": "playback"
                    })
        except Exception as e:
            logger.warning(f"获取播放设备列表失败: {e}")
        return devices

    @classmethod
    def select_best_device(cls, devices, preferred_keyword=None):
        """
        根据优先级智能推荐最佳声卡:
        1. 命中用户环境变量指定关键字 (如 USB, M1A, Jabra)
        2. 常见优质外置麦克风/音响特征词 (USB, Audio, Mic, Headset)
        3. 排除显卡纯 HDMI 等无效设备
        4. 兜底选择第一张非空声卡
        """
        if not devices:
            return None

        # 1. 优先匹配环境变量指定的设备关键字
        pref = preferred_keyword or os.getenv("AUDIO_DEVICE_KEYWORD", "").strip().lower()
        if pref:
            for d in devices:
                full_name = f"{d['card_id']} {d['card_desc']} {d['device_desc']}".lower()
                if pref in full_name:
                    return d

        # 2. 权重评分机制
        def score_device(d):
            full_name = f"{d['card_id']} {d['card_desc']} {d['device_desc']}".lower()
            score = 0
            if any(k in full_name for k in ["m1a", "jabra", "emeet", "anker", "speakerphone"]):
                score += 100
            if "usb" in full_name:
                score += 50
            if any(k in full_name for k in ["headset", "mic", "analog"]):
                score += 30
            if any(k in full_name for k in ["hdmi", "nvidia"]):
                score -= 80 # 显卡 HDMI 优先级调低
            return score

        sorted_devs = sorted(devices, key=score_device, reverse=True)
        return sorted_devs[0]

    @classmethod
    def auto_configure_alsa(cls, force=False):
        """
        根据宿主机探测到的实际硬件，自动生成健壮的 /etc/asound.conf:
        - 引入 plug 插件：无论硬件支持什么原生采样率，自动重采样，绝不报错
        - 引入 dmix / dsnoop：允许多进程共享麦克风与声卡，绝不出现 Device busy
        """
        cap_devs = cls.get_capture_devices()
        play_devs = cls.get_playback_devices()

        if not cap_devs and not play_devs:
            logger.warning("⚠️ 未检测到任何物理声卡设备，维持系统默认配置")
            return False

        best_cap = cls.select_best_device(cap_devs, os.getenv("CAPTURE_DEVICE_KEYWORD"))
        best_play = cls.select_best_device(play_devs, os.getenv("PLAYBACK_DEVICE_KEYWORD"))

        cap_target = f"hw:{best_cap['card_id']},{best_cap['device_num']}" if best_cap else "hw:0,0"
        play_target = f"hw:{best_play['card_id']},{best_play['device_num']}" if best_play else "hw:0,0"
        cap_card_id = best_cap['card_id'] if best_cap else "0"
        play_card_id = best_play['card_id'] if best_play else "0"

        logger.info(f"🎙️ 选定最佳录音硬件: {best_cap['card_desc'] if best_cap else '默认'} -> {cap_target}")
        logger.info(f"🔊 选定最佳播放硬件: {best_play['card_desc'] if best_play else '默认'} -> {play_target}")

        asound_template = f"""# -------------------------------------------------------------
# Local Voice Gateway 自动生成的自愈 ALSA 配置文件
# 支持: 自动重采样 (plug) + 录音防独占 (dsnoop) + 播放防独占 (dmix)
# -------------------------------------------------------------

pcm.!default {{
    type asym
    playback.pcm "plug:dmixer"
    capture.pcm "plug:dsnooper"
}}

pcm.dmixer {{
    type dmix
    ipc_key 1024
    ipc_key_add_uid false
    ipc_perm 0666
    slave {{
        pcm "{play_target}"
        period_time 0
        period_size 1024
        buffer_size 4096
        rate 44100
    }}
    bindings {{
        0 0
        1 1
    }}
}}

pcm.dsnooper {{
    type dsnoop
    ipc_key 2048
    ipc_key_add_uid false
    ipc_perm 0666
    slave {{
        pcm "{cap_target}"
        period_time 0
        period_size 1024
        buffer_size 4096
        rate 16000
    }}
}}

pcm.music {{
    type softvol
    slave.pcm "dmixer"
    control {{
        name "Music"
        card "{play_card_id}"
    }}
}}

ctl.!default {{
    type hw
    card "{play_card_id}"
}}
"""
        asound_conf_path = "/etc/asound.conf"
        try:
            # 只有当 /etc/asound.conf 不存在或要求强制刷新时才覆盖
            if force or not os.path.exists(asound_conf_path):
                with open(asound_conf_path, "w") as f:
                    f.write(asound_template)
                logger.info(f"✅ 成功生成自适应 ALSA 配置: {asound_conf_path}")
            else:
                logger.info(f"ℹ️ 检测到系统已存在 {asound_conf_path}，使用现有配置 (若需重新自愈请设置 force=True)")
        except PermissionError:
            logger.warning(f"无权限写入 {asound_conf_path} (通常发生在非 root 容器或普通用户下)，尝试写入用户级 ~/.asoundrc")
            try:
                user_asound = os.path.expanduser("~/.asoundrc")
                with open(user_asound, "w") as f:
                    f.write(asound_template)
                logger.info(f"✅ 成功写入用户级 ALSA 配置: {user_asound}")
            except Exception as ex:
                logger.error(f"写入用户级 ALSA 配置失败: {ex}")
        except Exception as e:
            logger.error(f"生成 ALSA 配置异常: {e}")

        # 解除静音并初始化音量
        cls.unmute_devices(cap_card_id, play_card_id)
        return True

    @staticmethod
    def unmute_devices(cap_card, play_card):
        """使用系统默认音量，不主动通过代码强制设置音量百分比"""
        pass

if __name__ == "__main__":
    # 命令行测试诊断工具
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("🔍 Local Voice Gateway 音频硬件自检诊断工具")
    print("=" * 60)
    caps = AudioDeviceManager.get_capture_devices()
    print(f"\n[检测到 {len(caps)} 个录音输入设备]:")
    for c in caps:
        print(f"  - Card {c['card_num']}: [{c['card_id']}] {c['card_desc']} (Device {c['device_num']})")

    plays = AudioDeviceManager.get_playback_devices()
    print(f"\n[检测到 {len(plays)} 个播放输出设备]:")
    for p in plays:
        print(f"  - Card {p['card_num']}: [{p['card_id']}] {p['card_desc']} (Device {p['device_num']})")

    print("\n[开始智能自愈配置]...")
    AudioDeviceManager.auto_configure_alsa(force=False)
    print("=" * 60)

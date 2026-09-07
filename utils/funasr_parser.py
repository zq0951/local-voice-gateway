import re

def parse_funasr_tags(text):
    """
    解析 SenseVoiceSmall 识别结果中的情绪、语言和事件标签
    示例: <|zh|><|HAPPY|><|Speech|>你好呀 -> text: "你好呀", emotion: "HAPPY", lang: "zh", is_speech: True
    """
    if not text:
        return {"text": "", "emotion": None, "lang": None, "event": None, "is_speech": False}

    tags = re.findall(r'<\|(.*?)\|>', text)
    clean_text = re.sub(r'<\|.*?\|>', '', text).strip()

    emotion = None
    lang = None
    event = None

    emotions_map = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "EMO_UNKNOWN"}
    langs_map = {"zh", "en", "jp", "ko", "yue", "auto"}
    events_map = {"Speech", "Music", "Laughter", "Applause"}

    for tag in tags:
        if tag in emotions_map:
            if not emotion or emotion in ["NEUTRAL", "EMO_UNKNOWN"]:
                emotion = tag
        elif tag in langs_map:
            lang = tag
        elif tag in events_map:
            event = tag

    is_speech = False
    if event == "Speech":
        is_speech = True
    elif not event and clean_text:
        is_speech = True
    
    if "[IGNORE]" in clean_text or not clean_text:
        is_speech = False

    return {
        "text": clean_text,
        "emotion": emotion,
        "lang": lang,
        "event": event,
        "is_speech": is_speech
    }

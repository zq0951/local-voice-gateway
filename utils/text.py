import re

def simple_t2s(text):
    """极其轻量级的常见繁简转换，用于处理 STT 偶发的繁体输出"""
    mapping = {
        '閉': '闭', '嘴': '嘴', '說': '说', '別': '别', '講': '讲',
        '停': '停', '止': '止', '安': '安', '靜': '静', '為': '为',
        '聽': '听', '開': '开', '關': '关', '燈': '灯', '溫': '温',
        '氣': '气', '天': '天', '預': '预', '報': '报', '啟': '启',
        '處': '处', '理': '理', '現': '现', '場': '场', '機': '机'
    }
    for t_char, s_char in mapping.items():
        text = text.replace(t_char, s_char)
    return text

def filter_symbols(text):
    """清洗文本，剔除思维链及不适合 TTS 发音的特殊符号"""
    if not text:
        return ""

    # 1. 移除思维链标签及其内容
    tags = ["think", "thinking", "reasoning", "thought", "REASONING_SCRATCHPAD"]
    for tag in tags:
        text = re.sub(rf'<{tag}>.*?</{tag}>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(rf'</?{tag}>', '', text, flags=re.IGNORECASE)

    # 2. 符号口语化预处理
    symbol_map = {
        '℃': '度',
        '~': '到',
        '～': '到',
        '%': '百分之',
        ':': '：',
        '+': '加',
        '-': '减',
        '*': '乘',
        '/': '除',
        '=': '等于',
        '>': '大于',
        '<': '小于'
    }
    for k, v in symbol_map.items():
        text = text.replace(k, v)

    # 3. 移除 Markdown 装饰符
    text = text.replace('**', '')
    text = text.replace('__', '')
    text = re.sub(r'^[-\*\#]\s*', '', text, flags=re.MULTILINE) 
    
    # 4. 移除注释风格的括号内容
    text = re.sub(r'[\(\uff08](思考|动作|笑|哭|叹气|语气|暂停|停顿|背景音).*?[\uff09\)]', '', text)
    text = text.replace('(', '').replace(')', '').replace('（', '').replace('）', '')
    
    # 5. 保留核心标点及中英文数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？.,!?]', '', text)
    
    text = text.strip()
    if not text:
        return ""
    
    # 6. 强力 EOS 注入，防止无终止标点导致的杂音
    if not re.search(r'[，。！？.,!?]$', text):
        text += "。"
    
    weak_to_strong = {
        '，': '。',
        ',': '。',
        '：': '。',
        ':': '。',
        '；': '。',
        ';': '。'
    }
    if text[-1] in weak_to_strong:
        text = text[:-1] + weak_to_strong[text[-1]]
        
    return text

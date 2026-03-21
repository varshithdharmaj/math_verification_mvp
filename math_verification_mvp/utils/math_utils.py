import re

# CJK character ranges (Chinese, Japanese, Korean)
CJK_PATTERN = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u3000-\u303f\uff00-\uffef]')

# CJK character ranges (Chinese, Japanese, Korean) including punctuation
CJK_PATTERN = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u3000-\u303f\uff00-\uffef\u3001\u3002\uff0c\uff0e\uff1a\uff1b\uff1f\uff01]')

def clean_latex(text: str) -> str:
    """Standardized cleaning for both OCR and LLM logic."""
    if not text: return ""
    # Remove CJK characters and punctuation
    text = CJK_PATTERN.sub('', text)
    # Remove LaTeX wrappers
    text = text.replace('\\', '').replace('{', '').replace('}', '').replace('$', '')
    # Remove common conversational prefixes in math problems
    text = re.sub(r'(?i)\b(prove|solve|calculate|find|simplify|evaluate|where)\b', '', text)
    # Expand implicit multiplication: 2x -> 2*x
    text = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', text)
    text = re.sub(r'([a-zA-Z\)])(\d)', r'\1*\2', text)
    # Normalize whitespace and strip
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_math_string(s: str) -> str:
    """Normalize mathematical strings for comparison."""
    if not s: return ""
    s = s.replace(" ", "").lower()
    # Try to normalize numeric parts
    try:
        # Split by comma for multi-value answers correctly
        if ',' in s:
            parts = [normalize_math_string(p) for p in s.split(',')]
            return ','.join(sorted(parts))
        f = float(s)
        return str(int(f)) if f == int(f) else str(round(f, 6))
    except:
        return s

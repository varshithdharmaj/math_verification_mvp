import re
text = "沪州批发发大米有发中司中。"
CJK_PATTERN = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u3000-\u303f\uff00-\uffef]')
print("Original:", text)
cleaned = CJK_PATTERN.sub('', text)
print("Cleaned regex 1:", cleaned)

# Let's try another regex to be absolutely sure we catch everything
res = "".join(c for c in text if not ('\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf' or '\u3000' <= c <= '\u303f' or '\uff00' <= c <= '\uffef'))
print("Cleaned regex 2:", res)

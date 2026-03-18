import unicodedata

text = "沪州批发发大米有发中司中。"
for c in text:
    print(f"Char: '{c}' | Code: {ord(c):04X} | Name: {unicodedata.name(c, 'Unknown')}")

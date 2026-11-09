import re
from typing import Any


def boxed_content(text: Any) -> str | None:
    text = "" if text is None else str(text)
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    if not starts:
        return None
    start = starts[-1].end()
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index].strip()
    return text[start:].strip()


THINK_PATTERN = re.compile(r"<think>.*?</think>", re.S)
OFFICIAL_PREFIXES = ("The best answer is", "The correct answer is", "The answer is", "The answer",
                     "The best option is", "The correct option is", "Best answer:", "Best option:")


def strip_think(text: Any) -> str:
    return THINK_PATTERN.sub("", "" if text is None else str(text)).strip()


def official_choice(text: Any, letters: str = "ABCD") -> str:
    """Video-MME's official extract_characters_regex, run on the boxed answer when there is one.

    Unboxed replies such as 'C. Shells.' are read the way the official script reads them."""
    reply = strip_think(text)
    s = boxed_content(reply)
    s = reply if s is None else s
    for prefix in OFFICIAL_PREFIXES:
        s = s.replace(prefix, "")
    if len(s.split()) > 10 and not re.search(f"[{letters}]", s):
        return ""
    match = re.search(f"[{letters}]", s)
    return match.group(0) if match else ""

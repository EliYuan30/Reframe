import re
from decimal import Decimal

_LATEX_PUNCTUATION = re.compile(r"\\([#$%&_{}])")
_ALIAS_GROUPS = (
    ("bike", "bicycle"),
    ("cell phone", "mobile phone"),
)
_ALIAS_INDEX = {alias: group[0] for group in _ALIAS_GROUPS for alias in group}
_STATE_ALIASES = {"nc": "north carolina", "north carolina": "north carolina"}
_STATE_CONTEXT = re.compile(
    r"\b(?:what|which)\b[^\n?]{0,40}\b(?:state|province)\b",
    re.IGNORECASE,
)


def unescape_latex_punctuation(text):
    return _LATEX_PUNCTUATION.sub(r"\1", str(text))


def _alias_key(text):
    text = unescape_latex_punctuation(text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _alias_match(prediction, target, context):
    target_key = _alias_key(target)
    target_alias = _ALIAS_INDEX.get(target_key)
    if target_alias is not None:
        return _ALIAS_INDEX.get(_alias_key(prediction)) == target_alias

    target_alias = _STATE_ALIASES.get(target_key)
    if target_alias is None or not _STATE_CONTEXT.search(context or ""):
        return None
    return _STATE_ALIASES.get(_alias_key(prediction)) == target_alias


_FIELD_LABELS = [
    ("invoice_id", re.compile(r"^invoice[\s_]+(?:id|number|no\.?|#)(?:\s*:\s*|\s+)", re.IGNORECASE)),
    (
        "tax",
        re.compile(
            r"^(?:tax(?:\s*:\s*(?:vat|gst|sales[\s_-]+tax))?|vat|gst)"
            r"(?:\s*\(\s*[+-]?[0-9]+(?:\.[0-9]+)?\s*%\s*\))?"
            r"\s*:\s*",
            re.IGNORECASE,
        ),
    ),
    ("subtotal", re.compile(r"^sub[\s_-]*total\b\s*:?\s*", re.IGNORECASE)),
    ("total", re.compile(r"^total(?:[\s_]+amount)?\b\s*:?\s*", re.IGNORECASE)),
    ("gross", re.compile(r"^gross[\s_]+amount\b\s*:?\s*", re.IGNORECASE)),
    ("balance", re.compile(r"^(?:balance[\s_]+due|due[\s_]+amount)\b\s*:?\s*", re.IGNORECASE)),
]
_CURRENCIES = {"$": "USD", "US$": "USD", "\u20ac": "EUR", "\u00a3": "GBP"}
_CURRENCY = r"(?:US\$|USD|EUR|GBP|CAD|AUD|CHF|JPY|CNY|INR|\$|\u20ac|\u00a3)"
_AMOUNT = re.compile(
    rf"(?P<before>{_CURRENCY})?\s*"
    r"(?P<number>[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)"
    rf"\s*(?P<after>{_CURRENCY})?", re.IGNORECASE,
)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9/._-]*")


def _field(text):
    text = unescape_latex_punctuation(text).strip()
    for kind, pattern in _FIELD_LABELS:
        match = pattern.match(text)
        if match:
            return kind, text[match.end():].strip()
    return None, text


def _amount(text):
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    match = _AMOUNT.fullmatch(text)
    if not match or (negative and match["number"].startswith(("+", "-"))):
        return None
    currencies = [
        _CURRENCIES.get(value.upper(), value.upper())
        for value in (match["before"], match["after"]) if value
    ]
    if len(set(currencies)) > 1:
        return None
    value = Decimal(match["number"].replace(",", ""))
    return (value.copy_negate() if negative else value), (currencies[0] if currencies else None)


def match_document_answer(prediction, target, context=None):
    target_literal = unescape_latex_punctuation(target).strip()
    if target_literal == "$":
        return unescape_latex_punctuation(prediction).strip() == target_literal

    alias_match = _alias_match(prediction, target, context)
    if alias_match is not None:
        return alias_match
    target_kind, target_value = _field(target)
    pred_kind, pred_value = _field(prediction)
    if target_kind == "invoice_id":
        if not _ID.fullmatch(target_value):
            return None
        return (pred_kind in (None, target_kind) and bool(_ID.fullmatch(pred_value))
                and pred_value.upper() == target_value.upper())

    target_amount = _amount(target_value)
    if target_amount is None or (target_kind is None and target_amount[1] is None):
        return None
    if pred_kind is not None and pred_kind != target_kind:
        return False
    pred_amount = _amount(pred_value)
    if pred_amount is None:
        return False
    value, currency = target_amount
    pred_value, pred_currency = pred_amount
    return (pred_value == value
            and (currency is None or pred_currency is None or currency == pred_currency))

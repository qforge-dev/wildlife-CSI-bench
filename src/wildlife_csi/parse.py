"""Conservative extraction of one animal name from a model response.

The prompt requests only a name. This parser also accepts common response
wrappers, but never uses the ground truth to choose between candidates.
"""

from __future__ import annotations

import json
import re

SYSTEM_PROMPT = (
    "Identify animal traces from photos. Give one best species guess, preferably a scientific "
    "name. Reply on one line: ANIMAL: <name>. No explanation or alternatives."
)


def user_prompt(location: dict) -> str:
    """Render the same country prompt for every task."""
    country = location["country"]
    return (
        "I found this animal trace in the field. "
        f"The photo's country is {country}. Which species most likely left it?"
    )


# A Latin binomial in parentheses or prose is usually more precise than the
# surrounding common name. Multiple distinct binomials make the answer ambiguous.
BINOMIAL = re.compile(r"\b([A-Z][a-z]{2,} [a-z][a-z-]{2,}(?: [a-z][a-z-]{2,})?)\b")
PARENTHESIZED_BINOMIAL = re.compile(r"\(([A-Z][a-z]{2,} [a-z][a-z-]{2,}(?: [a-z][a-z-]{2,})?)\)")
NON_TAXON_START = {
    "This",
    "There",
    "These",
    "Those",
    "The",
    "I",
    "It",
    "My",
    "Based",
    "Given",
    "Looking",
    "Likely",
    "Probably",
    "Maybe",
    "Because",
    "From",
    "With",
    "What",
}
NON_TAXON_SECOND = {
    "because",
    "which",
    "that",
    "looks",
    "seems",
    "might",
    "would",
    "could",
    "appears",
    "likely",
    "probably",
    "from",
    "with",
}
NAME_WORD = re.compile(r"^[\w'’\-]+(?:\s+[\w'’\-]+){0,5}$", re.UNICODE)
ALTERNATIVE_WORD = re.compile(r"(?:^|\s)(?:or|and|either|alternatively)(?:\s|$)", re.IGNORECASE)
FIELD = re.compile(r"^(?:species|animal|answer|taxon)\s*:\s*(.+)$", re.IGNORECASE)
FINAL_FIELD = re.compile(r"^\s*ANIMAL\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
PREFACE = re.compile(
    r"^(?:I (?:think|believe|would guess)(?: (?:it|this|the (?:bird|animal|species)) is)?|"
    r"(?:this|it) (?:is|looks like|appears to be)|"
    r"the (?:animal|species|bird|egg|feather|answer) (?:is|looks like|belongs to)|"
    r"my (?:best )?guess is)\s+",
    re.IGNORECASE,
)


def _clean_name(text: str) -> str:
    text = text.strip().lstrip("-*• ").strip("`*_\"' ")
    text = re.split(r"\s+[—–-]\s+", text, maxsplit=1)[0]
    text = re.split(
        r"\s+(?:because|as evidenced by|due to|judging by)\b", text, maxsplit=1, flags=re.IGNORECASE
    )[0]
    text = text.split("\n", 1)[0].strip().rstrip(".!;: ")
    text = re.sub(
        r"^(?:(?:probably|likely|perhaps|possibly)\s+)*(?:a|an)\s+", "", text, flags=re.IGNORECASE
    )
    return text.strip("`*_\"' ")


def _one_name(text: str) -> str | None:
    name = _clean_name(text)
    if not name or not NAME_WORD.fullmatch(name) or ALTERNATIVE_WORD.search(name):
        return None
    return name


def _plain_emphasis(text: str) -> str:
    """Remove balanced Markdown emphasis around names, without changing their words."""
    return re.sub(r"\*{1,2}([^*\n]+)\*{1,2}", r"\1", text)


def parse_answer(raw: str | None) -> dict:
    """Return {status, trace, answer}; status answered, ambiguous, or invalid."""
    if raw is None or not str(raw).strip():
        return {"status": "invalid", "trace": "", "answer": ""}
    text = str(raw).strip()
    if len(text) > 2000:
        return {"status": "invalid", "trace": "", "answer": ""}
    final_fields = FINAL_FIELD.findall(text)
    if final_fields:
        if len(final_fields) != 1:
            return {"status": "ambiguous", "trace": "", "answer": ""}
        value = _plain_emphasis(final_fields[0].strip().rstrip(".! "))
        if name := _one_name(value):
            return {"status": "answered", "trace": "field", "answer": name}
        scientific = PARENTHESIZED_BINOMIAL.findall(value)
        if len(scientific) == 1:
            common = PARENTHESIZED_BINOMIAL.sub("", value).strip()
            if _one_name(common) and "(" not in common and ")" not in common:
                return {"status": "answered", "trace": "field_binomial", "answer": scientific[0]}
        leading = re.fullmatch(r"([^()]+)\s+\(([^()]+)\)", value)
        if leading and BINOMIAL.fullmatch(leading.group(1)) and _one_name(leading.group(2)):
            return {"status": "answered", "trace": "field_binomial", "answer": leading.group(1)}
        return {"status": "invalid", "trace": "field", "answer": ""}
    if re.search(r"\b(?:either|or)\b", text, re.IGNORECASE):
        # An explicit alternative is not one final identification.
        return {"status": "ambiguous", "trace": "", "answer": ""}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        fields = [
            str(payload[k])
            for k in ("species", "animal", "answer", "taxon")
            if isinstance(payload.get(k), str) and payload[k].strip()
        ]
        if len(fields) == 1 and (name := _one_name(fields[0])):
            return {"status": "answered", "trace": "json", "answer": name}
        return {"status": "ambiguous" if len(fields) > 1 else "invalid", "trace": "", "answer": ""}

    binomials = list(
        dict.fromkeys(
            m.group(1)
            for m in BINOMIAL.finditer(text)
            if m.group(1).split()[0] not in NON_TAXON_START
            and m.group(1).split()[1] not in NON_TAXON_SECOND
        )
    )
    if len(binomials) > 1:
        return {"status": "ambiguous", "trace": "", "answer": ""}
    if len(binomials) == 1:
        return {"status": "answered", "trace": "binomial", "answer": binomials[0]}

    first = text.splitlines()[0].strip()
    field = FIELD.match(first)
    if field and (name := _one_name(field.group(1))):
        return {"status": "answered", "trace": "labeled", "answer": name}
    if match := PREFACE.match(first):
        if name := _one_name(first[match.end() :].split(".", 1)[0]):
            return {"status": "answered", "trace": "preface", "answer": name}
    if ". " in first and (name := _one_name(first.split(". ", 1)[0])):
        return {"status": "answered", "trace": "sentence", "answer": name}
    if len(text.splitlines()) == 1 and (name := _one_name(text)):
        return {"status": "answered", "trace": "plain", "answer": name}
    return {"status": "invalid", "trace": "", "answer": ""}

"""ค้นคืนข้อความแบบ BM25 ที่รองรับภาษาไทย

ภาษาไทยไม่เว้นวรรคระหว่างคำ การตัดคำด้วยช่องว่างจึงใช้ไม่ได้ โมดูลนี้ใช้
character n-gram (2-3 ตัวอักษร) สำหรับอักษรไทย ร่วมกับการตัดคำตามช่องว่าง
สำหรับอักษรละตินและตัวเลข ทำให้ค้นคืนภาษาไทยได้โดยไม่ต้องพึ่งไลบรารีตัดคำ
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

_THAI_RANGE = re.compile(r"[฀-๿]+")
_ASCII_TOKEN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9.\-]*")
# คำที่พบบ่อยจนไม่ช่วยแยกเอกสาร (ตัดออกเพื่อลด noise)
_STOP_NGRAMS = {"การ", "ที่", "ของ", "และ", "ให้", "ใน", "เป็น", "ได้", "ไม่", "มี", "จะ", "กับ"}


def tokenize(text: str) -> list[str]:
    """แปลงข้อความเป็นโทเคน: n-gram สำหรับไทย และคำสำหรับอังกฤษ/ตัวเลข"""
    if not text:
        return []
    text = text.lower()
    tokens: list[str] = [m.group(0) for m in _ASCII_TOKEN.finditer(text)]
    for chunk in _THAI_RANGE.findall(text):
        for n in (2, 3):
            if len(chunk) < n:
                continue
            for i in range(len(chunk) - n + 1):
                gram = chunk[i : i + n]
                if gram not in _STOP_NGRAMS:
                    tokens.append(gram)
    return tokens


def flatten_text(value: Any, depth: int = 0) -> str:
    """รวมข้อความจากโครงสร้าง dict/list ให้เป็นสตริงเดียวสำหรับทำดัชนี"""
    if depth > 8:
        return ""
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, dict):
        return " ".join(flatten_text(v, depth + 1) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(flatten_text(v, depth + 1) for v in value)
    return ""


@dataclass
class Document:
    """เอกสารหนึ่งชิ้นในดัชนีค้นคืน"""

    doc_id: str
    kind: str
    title: str
    text: str
    payload: dict[str, Any] = field(default_factory=dict)
    boost: float = 1.0


class BM25Index:
    """ดัชนี BM25 ขนาดเล็กที่ไม่ต้องใช้ dependency ภายนอก"""

    def __init__(self, documents: Sequence[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[Document] = list(documents)
        self._term_freqs: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._doc_freq: Counter[str] = Counter()
        for doc in self.documents:
            tf = Counter(tokenize(f"{doc.title} {doc.title} {doc.text}"))
            self._term_freqs.append(tf)
            self._lengths.append(sum(tf.values()) or 1)
            self._doc_freq.update(tf.keys())
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 1.0
        self._n = len(self.documents)

    def _idf(self, term: str) -> float:
        df = self._doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))

    @staticmethod
    def _name_bonus(title: str, query_lower: str) -> float:
        """ให้คะแนนพิเศษเมื่อชื่อหลักของเอกสารปรากฏตรงตัวในคำถาม

        ช่วยให้การถามถึงสารตัวเดียว เช่น 'แมนโคเซบ' ไม่ถูกแย่งอันดับโดยสูตรผสม
        ที่มีชื่อนั้นอยู่ด้วยแต่มีคำอื่นมากกว่า
        """
        name = title.split(" (")[0].split(" [")[0].split(" - ")[0].strip().lower()
        if len(name) >= 3 and name in query_lower:
            return 1.8
        return 1.0

    def search(
        self,
        query: str,
        limit: int = 6,
        kinds: Iterable[str] | None = None,
    ) -> list[tuple[Document, float]]:
        """คืนรายการเอกสารที่เกี่ยวข้องที่สุดพร้อมคะแนน"""
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        query_lower = query.lower()
        allowed = set(kinds) if kinds else None
        q_counts = Counter(q_tokens)
        scored: list[tuple[Document, float]] = []
        for idx, doc in enumerate(self.documents):
            if allowed is not None and doc.kind not in allowed:
                continue
            tf = self._term_freqs[idx]
            if not tf:
                continue
            length = self._lengths[idx]
            score = 0.0
            for term, q_tf in q_counts.items():
                f = tf.get(term)
                if not f:
                    continue
                idf = self._idf(term)
                if idf <= 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * length / self._avg_len)
                score += idf * (f * (self.k1 + 1) / denom) * min(q_tf, 3)
            if score > 0:
                scored.append((doc, score * doc.boost * self._name_bonus(doc.title, query_lower)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

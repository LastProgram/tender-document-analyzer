import re
from dataclasses import dataclass

from app.schemas.document import DocumentChunk, DocumentPage

_PARAGRAPH_BOUNDARY = re.compile(r"((?<=\n)\s*\n)")
_SENTENCE_BOUNDARY = re.compile(r"((?<=[.!?])\s+)")
_WORD_BOUNDARY = re.compile(r"((?<=\S)\s+)")
_TEXT_BOUNDARIES = (_PARAGRAPH_BOUNDARY, _SENTENCE_BOUNDARY, _WORD_BOUNDARY)

# Пустая строка отделяет размеченные фрагменты и не смешивает их текст.
_PAGE_FRAGMENT_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class _PageFragment:
    page_number: int
    text: str


class PageChunker:
    """Формирует ограниченные по размеру блоки без потери нумерации страниц."""

    def __init__(self, max_chars: int, overlap_pages: int = 1) -> None:
        if max_chars < len(self._page_marker(1)):
            raise ValueError("max_chars must fit a page marker")
        if overlap_pages < 0:
            raise ValueError("overlap_pages must not be negative")

        self._max_chars = max_chars
        self._overlap_pages = overlap_pages

    def split(self, pages: list[DocumentPage]) -> list[DocumentChunk]:
        fragments = [fragment for page in pages for fragment in self._split_page(page)]
        if not fragments:
            return []

        groups: list[list[_PageFragment]] = []
        current: list[_PageFragment] = []

        for fragment in fragments:
            if not current or self._fits([*current, fragment]):
                current.append(fragment)
                continue

            groups.append(current)
            current = self._overlap_from(current)
            # Целые страницы в overlap сохраняют корректные границы источника.
            while current and not self._fits([*current, fragment]):
                first_page = current[0].page_number
                current = [item for item in current if item.page_number != first_page]
            current.append(fragment)

        groups.append(current)
        return [self._to_chunk(group) for group in groups]

    def _split_page(self, page: DocumentPage) -> list[_PageFragment]:
        marker = self._page_marker(page.number)
        if len(marker) > self._max_chars:
            raise ValueError("max_chars must fit every page marker")

        # Пустые страницы сохраняются для соответствия физическим страницам PDF.
        if not page.text:
            return [_PageFragment(page.number, "")]

        text_limit = self._max_chars - len(marker) - 1
        if text_limit < 1:
            raise ValueError("max_chars must fit a page marker and page text")

        parts = self._split_text(page.text, text_limit, 0)
        return [_PageFragment(page.number, part) for part in parts]

    def _split_text(self, text: str, limit: int, boundary_index: int) -> list[str]:
        if len(text) <= limit:
            return [text]
        if boundary_index == len(_TEXT_BOUNDARIES):
            return [text[index : index + limit] for index in range(0, len(text), limit)]

        # Крупные границы текста уменьшают число разорванных предложений и слов.
        boundary = _TEXT_BOUNDARIES[boundary_index]
        pieces = boundary.split(text)
        if len(pieces) == 1:
            return self._split_text(text, limit, boundary_index + 1)

        result: list[str] = []
        current = ""
        for piece in pieces:
            if len(piece) > limit:
                if current:
                    result.append(current)
                    current = ""
                result.extend(self._split_text(piece, limit, boundary_index + 1))
            elif len(current) + len(piece) <= limit:
                current += piece
            else:
                result.append(current)
                current = piece

        if current:
            result.append(current)
        return result

    def _overlap_from(self, fragments: list[_PageFragment]) -> list[_PageFragment]:
        if self._overlap_pages == 0:
            return []

        page_numbers = list(dict.fromkeys(item.page_number for item in fragments))
        overlap_numbers = set(page_numbers[-self._overlap_pages :])
        return [item for item in fragments if item.page_number in overlap_numbers]

    def _fits(self, fragments: list[_PageFragment]) -> bool:
        return len(self._render(fragments)) <= self._max_chars

    def _to_chunk(self, fragments: list[_PageFragment]) -> DocumentChunk:
        return DocumentChunk(
            page_from=fragments[0].page_number,
            page_to=fragments[-1].page_number,
            text=self._render(fragments),
        )

    @classmethod
    def _render(cls, fragments: list[_PageFragment]) -> str:
        rendered = []
        for fragment in fragments:
            marker = cls._page_marker(fragment.page_number)
            rendered.append(f"{marker}\n{fragment.text}" if fragment.text else marker)
        return _PAGE_FRAGMENT_SEPARATOR.join(rendered)

    @staticmethod
    def _page_marker(page_number: int) -> str:
        return f"[PAGE {page_number}]"

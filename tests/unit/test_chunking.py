import pytest

from app.schemas.document import DocumentPage
from app.services.chunking import PageChunker


def test_short_document_creates_one_chunk() -> None:
    pages = [
        DocumentPage(number=1, text="First page"),
        DocumentPage(number=2, text="Second page"),
    ]

    chunks = PageChunker(max_chars=100).split(pages)

    assert len(chunks) == 1
    assert chunks[0].page_from == 1
    assert chunks[0].page_to == 2
    assert chunks[0].text == "[PAGE 1]\nFirst page\n\n[PAGE 2]\nSecond page"


def test_long_document_is_split_with_one_page_overlap() -> None:
    pages = [DocumentPage(number=number, text="x" * 10) for number in range(1, 5)]

    chunks = PageChunker(max_chars=42).split(pages)

    assert [(chunk.page_from, chunk.page_to) for chunk in chunks] == [
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    assert all(len(chunk.text) <= 42 for chunk in chunks)


def test_pages_keep_their_original_order() -> None:
    pages = [
        DocumentPage(number=7, text="first"),
        DocumentPage(number=3, text="second"),
        DocumentPage(number=9, text="third"),
    ]

    chunks = PageChunker(max_chars=35, overlap_pages=0).split(pages)

    rendered = "\n".join(chunk.text for chunk in chunks)
    assert rendered.index("[PAGE 7]") < rendered.index("[PAGE 3]")
    assert rendered.index("[PAGE 3]") < rendered.index("[PAGE 9]")


def test_overlap_is_only_copied_to_the_next_chunk() -> None:
    pages = [
        DocumentPage(number=number, text=str(number) * 10) for number in range(1, 5)
    ]

    chunks = PageChunker(max_chars=42).split(pages)

    assert chunks[0].text.count("[PAGE 2]") == 1
    assert chunks[1].text.count("[PAGE 2]") == 1
    assert "[PAGE 2]" not in chunks[2].text


def test_empty_intermediate_page_keeps_its_number() -> None:
    pages = [
        DocumentPage(number=1, text="first"),
        DocumentPage(number=2, text=""),
        DocumentPage(number=3, text="third"),
    ]

    [chunk] = PageChunker(max_chars=100).split(pages)

    assert chunk.text == "[PAGE 1]\nfirst\n\n[PAGE 2]\n\n[PAGE 3]\nthird"


def test_exact_limit_does_not_create_an_empty_chunk() -> None:
    page = DocumentPage(number=1, text="content")
    expected = "[PAGE 1]\ncontent"

    chunks = PageChunker(max_chars=len(expected)).split([page])

    assert [chunk.text for chunk in chunks] == [expected]


def test_oversized_page_keeps_all_content_within_limit() -> None:
    text = "First paragraph.\n\nSecond sentence. " + "word " * 8 + "x" * 30

    chunks = PageChunker(max_chars=30).split([DocumentPage(number=4, text=text)])

    restored = "".join(chunk.text.removeprefix("[PAGE 4]\n") for chunk in chunks)
    assert restored == text
    assert all(chunk.page_from == chunk.page_to == 4 for chunk in chunks)
    assert all(len(chunk.text) <= 30 for chunk in chunks)


def test_markers_count_towards_the_limit() -> None:
    pages = [
        DocumentPage(number=1, text="12345"),
        DocumentPage(number=2, text="12345"),
    ]

    chunks = PageChunker(max_chars=20, overlap_pages=0).split(pages)

    assert len(chunks) == 2


def test_overlap_larger_than_chunk_uses_available_pages() -> None:
    pages = [DocumentPage(number=number, text="x" * 10) for number in range(1, 4)]

    chunks = PageChunker(max_chars=42, overlap_pages=10).split(pages)

    assert [(chunk.page_from, chunk.page_to) for chunk in chunks] == [(1, 2), (2, 3)]


def test_same_input_produces_the_same_chunks() -> None:
    pages = [DocumentPage(number=number, text="text " * 5) for number in range(1, 5)]
    chunker = PageChunker(max_chars=50)

    assert chunker.split(pages) == chunker.split(pages)


@pytest.mark.parametrize("max_chars", [0, 7])
def test_limit_smaller_than_page_marker_is_rejected(max_chars: int) -> None:
    with pytest.raises(ValueError, match="page marker"):
        PageChunker(max_chars=max_chars)

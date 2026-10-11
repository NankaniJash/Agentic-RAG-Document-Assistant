from app.ingestion.chunker import chunk_pages
from app.ingestion.loaders import PageText


def test_splits_at_paragraph_boundaries_once_size_exceeded():
    paragraphs = [f"Paragraph {i}. " + ("x" * 300) for i in range(5)]
    pages = [PageText(page=3, text="\n\n".join(paragraphs))]

    chunks = chunk_pages(pages, size=700, overlap=50)

    assert len(chunks) > 1
    assert all(c.page == 3 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_overlap_carries_chunk_boundary_context_forward():
    paragraphs = [f"Paragraph {i}. " + ("x" * 300) for i in range(5)]
    pages = [PageText(page=1, text="\n\n".join(paragraphs))]

    chunks = chunk_pages(pages, size=700, overlap=50)

    tail_of_first = chunks[0].text[-50:]
    assert tail_of_first[:20] in chunks[1].text


def test_short_text_stays_as_a_single_chunk():
    pages = [PageText(page=1, text="Short paragraph.\n\nAnother short one.")]

    chunks = chunk_pages(pages, size=1000, overlap=100)

    assert len(chunks) == 1


def test_blank_page_produces_no_chunks():
    pages = [PageText(page=1, text="   \n\n   ")]

    assert chunk_pages(pages) == []


def test_multiple_pages_get_independent_page_numbers():
    pages = [
        PageText(page=1, text="First page content."),
        PageText(page=2, text="Second page content."),
    ]

    chunks = chunk_pages(pages, size=1000, overlap=50)

    assert [c.page for c in chunks] == [1, 2]

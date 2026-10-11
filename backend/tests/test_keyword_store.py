from app.retrieval.keyword_store import _tokenize


def test_lowercases_and_strips_punctuation():
    assert _tokenize("Hello, World! v2.0") == ["hello", "world", "v2", "0"]


def test_empty_string_yields_no_tokens():
    assert _tokenize("") == []


def test_ignores_whitespace_only_tokens():
    assert _tokenize("   \n\t  ") == []

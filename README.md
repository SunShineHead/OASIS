Here's how to parametrize encoding tests in pytest:

```python
import pytest

@pytest.mark.parametrize("encoding", ["utf-8", "latin-1", "utf-16", "ascii"])
def test_encode_decode(encoding):
    text = "hello"
    assert text.encode(encoding).decode(encoding) == text


@pytest.mark.parametrize("text,encoding", [
    ("hello", "utf-8"),
    ("café", "utf-8"),
    ("café", "latin-1"),
    ("日本語", "utf-8"),
])
def test_encoding_roundtrip(text, encoding):
    assert text.encode(encoding).decode(encoding) == text


@pytest.mark.parametrize("text,encoding,expected_bytes", [
    ("A", "ascii", b"\x41"),
    ("A", "utf-8", b"\x41"),
    ("\u00e9", "utf-8", b"\xc3\xa9"),   # é
    ("\u00e9", "latin-1", b"\xe9"),
])
def test_encoding_output(text, encoding, expected_bytes):
    assert text.encode(encoding) == expected_bytes
```

**Key points:**

- `@pytest.mark.parametrize("arg", [val1, val2])` for a single parameter
- `@pytest.mark.parametrize("arg1,arg2", [(v1, v2), (v3, v4)])` for multiple parameters (tuples)
- You can stack `@pytest.mark.parametrize` decorators to get the cartesian product of values

**Stacked example:**
```python
@pytest.mark.parametrize("text", ["hello", "café"])
@pytest.mark.parametrize("encoding", ["utf-8", "latin-1"])
def test_stacked(text, encoding):
    try:
        assert text.encode(encoding).decode(encoding) == text
    except (UnicodeEncodeError, UnicodeDecodeError):
        pytest.skip(f"{text!r} not supported in {encoding}")
```
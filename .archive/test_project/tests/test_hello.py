"""Hello unit test module."""

from x.hello import hello


def test_hello():
    """Test the hello function."""
    assert hello() == "Hello x"

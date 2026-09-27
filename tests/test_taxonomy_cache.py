import pytest

from wildlife_csi.taxonomy_cache import CachedTaxonResolver


class CountingResolver:
    def __init__(self):
        self.calls = []
        self.fail = False

    def resolve_name(self, name):
        self.calls.append(name)
        if self.fail:
            raise RuntimeError("temporary API failure")
        return None if name == "no match" else {"taxon_id": 42, "taxon": name}


def test_name_cache_survives_new_resolver_and_caches_negative_results(tmp_path):
    path = tmp_path / "names.sqlite3"
    first = CountingResolver()
    cache = CachedTaxonResolver(first, path, min_lookup_interval_s=0)
    assert cache.resolve_name("  Red   Fox ") == {"taxon_id": 42, "taxon": "  Red   Fox "}
    assert cache.resolve_name("no match") is None
    second = CountingResolver()
    reopened = CachedTaxonResolver(second, path, min_lookup_interval_s=0)
    assert reopened.resolve_name("red fox") == {"taxon_id": 42, "taxon": "  Red   Fox "}
    assert reopened.resolve_name("NO MATCH") is None
    assert first.calls == ["  Red   Fox ", "no match"]
    assert second.calls == []


def test_failed_lookup_is_not_cached(tmp_path):
    upstream = CountingResolver()
    upstream.fail = True
    cache = CachedTaxonResolver(upstream, tmp_path / "names.sqlite3", min_lookup_interval_s=0)
    with pytest.raises(RuntimeError):
        cache.resolve_name("wolf")
    upstream.fail = False
    assert cache.resolve_name("wolf") == {"taxon_id": 42, "taxon": "wolf"}
    assert cache.resolve_name(" WOLF ") == {"taxon_id": 42, "taxon": "wolf"}
    assert upstream.calls == ["wolf", "wolf"]

from m02.tokens import BOUNDARY, bigrams, build_vocab, decode, encode


def test_vocab_puts_boundary_first_then_sorted_chars():
    stoi, itos = build_vocab(["emma", "ava", "zoe"])
    assert itos[0] == BOUNDARY, "the boundary '.' should be token 0"
    assert itos == [".", "a", "e", "m", "o", "v", "z"], (
        f"expected '.' then the characters in sorted order, got {itos}")
    assert all(stoi[ch] == i for i, ch in enumerate(itos)), "stoi and itos should be inverses"


def test_vocab_ignores_duplicates_and_existing_boundary():
    stoi, itos = build_vocab(["aa.", "a"])
    assert itos == [".", "a"], f"each character once, '.' only at 0; got {itos}"


def test_encode_decode_round_trip():
    stoi, itos = build_vocab(["hello", "world"])
    ids = encode("lower", stoi)
    assert all(isinstance(i, int) for i in ids)
    assert decode(ids, itos) == "lower"
    assert encode("", stoi) == []


def test_bigrams_include_both_boundaries():
    assert bigrams("ab") == [(".", "a"), ("a", "b"), ("b", ".")]
    assert len(bigrams("emma")) == 5, "a word of n characters has n + 1 bigrams"
    assert bigrams("") == [(".", ".")]

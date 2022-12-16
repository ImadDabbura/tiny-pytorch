from tiny_pytorch.utils import listify, setify, tuplify


def test_listify():
    assert listify(None) == []
    assert listify([]) == []
    assert listify([1]) == [1]
    assert listify([1, 2]) == [1, 2]
    assert listify((1,)) == [1]
    assert listify("") == [""]
    assert listify("str") == ["str"]


def test_tuplify():
    assert tuplify(None) == ()
    assert tuplify([]) == ()
    assert tuplify([1]) == (1,)
    assert tuplify([1, 2]) == (1, 2)
    assert tuplify((1,)) == (1,)
    assert tuplify("") == ("",)
    assert tuplify("str") == ("str",)
    assert tuplify(()) == ()
    assert tuplify((1,)) == (1,)


def test_setify():
    assert setify(None) == set()
    assert setify([]) == set()
    assert setify(set()) == set()
    assert setify([1]) == {1}
    assert setify({1}) == {1}
    assert setify([1, 2]) == {1, 2}
    assert setify((1,)) == {1}
    assert setify("") == {""}
    assert setify("str") == {"str"}

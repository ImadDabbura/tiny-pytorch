from tiny_pytorch.nn import Parameter


def test_parameter():
    param = Parameter([1])
    assert isinstance(param, Parameter)
    assert param.requires_grad

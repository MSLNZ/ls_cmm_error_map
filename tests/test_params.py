import pytest

model_parameters_dict = {
    "Txx": 0,
    "Txy": 0,
    "Txz": 0,
}


def make_tests():
    tests = []
    for key in model_parameters_dict:
        test = model_parameters_dict.copy()
        test[key] = 10
        tests.append(test)
    return tests


@pytest.mark.parametrize("pdict", make_tests(), ids=model_parameters_dict.keys())
def test_foobar(pdict):
    assert list(pdict.values()).count(10) == 1

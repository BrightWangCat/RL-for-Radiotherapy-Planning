import pytest

try:
    import matlab
except ImportError:
    pytest.skip("Matlab not installed", allow_module_level=True)


@pytest.fixture
def engine():
    from pyRadPlan.matRad import MatRadEngineMatlab

    return MatRadEngineMatlab()


def test_engineBasicCall(engine):
    basicZeros = engine.zeros(10)
    assert basicZeros.size == (10, 10)  # Matlab returns matlab specific array (matlab.double)


def test_engineParameters(engine):
    assert isinstance(engine.matRadVersionStr, str)
    assert isinstance(engine.engineVersionStr, str)
    assert isinstance(engine.matRadPath, str)


def test_engineInitMatrad(engine):
    engine.matRad_rc

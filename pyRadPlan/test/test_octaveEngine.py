import pytest

try:
    import oct2py
except ImportError:
    pytest.skip("Octave not installed", allow_module_level=True)


@pytest.fixture
def engine():
    from pyRadPlan.matRad import MatRadEngineOctave

    return MatRadEngineOctave()


def test_engineBasicCall(engine):
    basicZeros = engine.zeros(10)
    assert basicZeros.shape == (10, 10)  # Octave returns numpy arrays


def test_engineParameters(engine):
    assert isinstance(engine.matRadVersionStr, str)
    assert isinstance(engine.engineVersionStr, str)
    assert isinstance(engine.matRadPath, str)


def test_engineInitMatrad(engine):
    engine.matRad_rc

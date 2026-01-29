from __future__ import annotations

import numpy as np
import pytest

import numbatalib.compat.talib as talib_nb


talib_ref = pytest.importorskip("talib")
abstract_ref = pytest.importorskip("talib.abstract")


def _reset_globals():
    talib_ref.set_compatibility(0)
    talib_nb.set_compatibility(0)
    for fn in ["EMA", "RSI", "CMO"]:
        try:
            talib_ref.set_unstable_period(fn, 0)
        except KeyError:
            pass
        try:
            talib_nb.set_unstable_period(fn, 0)
        except KeyError:
            pass


@pytest.fixture(autouse=True)
def _clean_global_state():
    _reset_globals()
    yield
    _reset_globals()


def test_error_messages_match_talib() -> None:
    x = np.arange(10.0)

    for call in [
        lambda t: t.SMA(x, timeperiod=1),
        lambda t: t.SMA(x.reshape(2, 5), timeperiod=3),
        lambda t: t.ADD(np.arange(10.0), np.arange(9.0)),
    ]:
        with pytest.raises(Exception) as e_ref:
            call(talib_ref)
        with pytest.raises(Exception) as e_nb:
            call(talib_nb)
        assert type(e_ref.value) is type(e_nb.value)
        assert str(e_ref.value) == str(e_nb.value)

    with pytest.raises(TypeError) as e_ref:
        talib_ref.SMA(x, timeperiod="a")
    with pytest.raises(TypeError) as e_nb:
        talib_nb.SMA(x, timeperiod="a")
    assert str(e_ref.value) == str(e_nb.value)


def test_stream_api_matches_talib() -> None:
    x = np.arange(10.0)

    assert talib_ref.stream.SMA(x, timeperiod=3) == talib_nb.stream.SMA(x, timeperiod=3)
    assert talib_ref.stream.BBANDS(x, timeperiod=3) == talib_nb.stream.BBANDS(x, timeperiod=3)

    assert talib_ref.stream_SMA(x, timeperiod=3) == talib_nb.stream_SMA(x, timeperiod=3)
    assert talib_ref.stream_BBANDS(x, timeperiod=3) == talib_nb.stream_BBANDS(x, timeperiod=3)


def test_abstract_api_basic_parity() -> None:
    f_ref = abstract_ref.Function("SMA")
    import numbatalib.compat.talib.abstract as abstract_nb

    f_nb = abstract_nb.Function("SMA")

    assert f_ref.input_names == f_nb.input_names
    assert f_ref.output_names == f_nb.output_names
    assert dict(f_ref.parameters) == dict(f_nb.parameters)

    x = np.arange(10.0)
    np.testing.assert_allclose(f_ref(x, timeperiod=3), f_nb(x, timeperiod=3), equal_nan=True)

    with pytest.raises(Exception) as e_ref:
        f_ref({"real": x})
    with pytest.raises(Exception) as e_nb:
        f_nb({"real": x})
    assert str(e_ref.value) == str(e_nb.value)


def test_compatibility_affects_ema_rsi_cmo_like_talib() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=512).cumsum().astype(np.float64)

    # EMA: any non-zero compatibility triggers metastock behavior.
    talib_ref.set_compatibility(1)
    talib_nb.set_compatibility(1)
    np.testing.assert_allclose(
        talib_ref.EMA(x, timeperiod=20),
        talib_nb.EMA(x, timeperiod=20),
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    )

    # RSI/CMO: only compatibility==1 triggers metastock behavior (extra early value).
    talib_ref.set_compatibility(1)
    talib_nb.set_compatibility(1)
    np.testing.assert_allclose(
        talib_ref.RSI(x, timeperiod=14),
        talib_nb.RSI(x, timeperiod=14),
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        talib_ref.CMO(x, timeperiod=14),
        talib_nb.CMO(x, timeperiod=14),
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    )


def test_unstable_period_masks_leading_outputs_like_talib() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=512).cumsum().astype(np.float64)

    talib_ref.set_unstable_period("EMA", 5)
    talib_nb.set_unstable_period("EMA", 5)

    np.testing.assert_allclose(
        talib_ref.EMA(x, timeperiod=20),
        talib_nb.EMA(x, timeperiod=20),
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(KeyError):
        talib_ref.get_unstable_period("SMA")
    with pytest.raises(KeyError):
        talib_nb.get_unstable_period("SMA")

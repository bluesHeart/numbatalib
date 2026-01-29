from __future__ import annotations

import math

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like


@njit(cache=True)
def _ht_dcphase_kernel(real: np.ndarray, out: np.ndarray) -> None:
    n = real.shape[0]
    lookback_total = 63
    if n <= lookback_total:
        return

    start_idx = lookback_total
    end_idx = n - 1

    # Constants.
    a = 0.0962
    b = 0.5769
    atan1 = math.atan(1.0)
    rad2deg = 45.0 / atan1  # 180/pi
    const_deg2rad_by_360 = atan1 * 8.0  # 2*pi

    # Circular buffer of smoothed prices.
    smooth_price = np.zeros(50, dtype=np.float64)
    smooth_price_idx = 0
    max_smooth_idx = 49

    # Price smoother (4-period WMA).
    trailing_wma_idx = start_idx - lookback_total
    today = trailing_wma_idx

    temp_real = real[today]
    today += 1
    period_wma_sub = temp_real
    period_wma_sum = temp_real
    temp_real = real[today]
    today += 1
    period_wma_sub += temp_real
    period_wma_sum += temp_real * 2.0
    temp_real = real[today]
    today += 1
    period_wma_sub += temp_real
    period_wma_sum += temp_real * 3.0

    trailing_wma_value = 0.0
    smoothed_value = 0.0

    # Warm-up smoother (34 iterations).
    for _ in range(34):
        temp_real = real[today]
        today += 1
        period_wma_sub += temp_real
        period_wma_sub -= trailing_wma_value
        period_wma_sum += temp_real * 4.0
        trailing_wma_value = real[trailing_wma_idx]
        trailing_wma_idx += 1
        smoothed_value = period_wma_sum * 0.1
        period_wma_sum -= period_wma_sub

    # Hilbert transform state.
    hilbert_idx = 0

    detrender_odd = np.zeros(3, dtype=np.float64)
    detrender_even = np.zeros(3, dtype=np.float64)
    q1_odd = np.zeros(3, dtype=np.float64)
    q1_even = np.zeros(3, dtype=np.float64)
    ji_odd = np.zeros(3, dtype=np.float64)
    ji_even = np.zeros(3, dtype=np.float64)
    jq_odd = np.zeros(3, dtype=np.float64)
    jq_even = np.zeros(3, dtype=np.float64)

    detrender = 0.0
    q1 = 0.0
    ji = 0.0
    jq = 0.0

    prev_detrender_odd = 0.0
    prev_detrender_even = 0.0
    prev_detrender_input_odd = 0.0
    prev_detrender_input_even = 0.0

    prev_q1_odd = 0.0
    prev_q1_even = 0.0
    prev_q1_input_odd = 0.0
    prev_q1_input_even = 0.0

    prev_ji_odd = 0.0
    prev_ji_even = 0.0
    prev_ji_input_odd = 0.0
    prev_ji_input_even = 0.0

    prev_jq_odd = 0.0
    prev_jq_even = 0.0
    prev_jq_input_odd = 0.0
    prev_jq_input_even = 0.0

    period = 0.0
    smooth_period = 0.0
    dc_phase = 0.0

    prev_i2 = 0.0
    prev_q2 = 0.0
    re = 0.0
    im = 0.0

    i1_for_odd_prev2 = 0.0
    i1_for_odd_prev3 = 0.0
    i1_for_even_prev2 = 0.0
    i1_for_even_prev3 = 0.0

    q2 = 0.0
    i2 = 0.0

    while today <= end_idx:
        adjusted_prev_period = (0.075 * period) + 0.54

        today_value = real[today]
        # DO_PRICE_WMA(today_value, smoothed_value)
        period_wma_sub += today_value
        period_wma_sub -= trailing_wma_value
        period_wma_sum += today_value * 4.0
        trailing_wma_value = real[trailing_wma_idx]
        trailing_wma_idx += 1
        smoothed_value = period_wma_sum * 0.1
        period_wma_sum -= period_wma_sub

        # Remember smoothed value.
        smooth_price[smooth_price_idx] = smoothed_value

        if (today % 2) == 0:
            # Even.
            hilbert_temp = a * smoothed_value
            detrender = -detrender_even[hilbert_idx]
            detrender_even[hilbert_idx] = hilbert_temp
            detrender += hilbert_temp
            detrender -= prev_detrender_even
            prev_detrender_even = b * prev_detrender_input_even
            detrender += prev_detrender_even
            prev_detrender_input_even = smoothed_value
            detrender *= adjusted_prev_period

            hilbert_temp = a * detrender
            q1 = -q1_even[hilbert_idx]
            q1_even[hilbert_idx] = hilbert_temp
            q1 += hilbert_temp
            q1 -= prev_q1_even
            prev_q1_even = b * prev_q1_input_even
            q1 += prev_q1_even
            prev_q1_input_even = detrender
            q1 *= adjusted_prev_period

            hilbert_temp = a * i1_for_even_prev3
            ji = -ji_even[hilbert_idx]
            ji_even[hilbert_idx] = hilbert_temp
            ji += hilbert_temp
            ji -= prev_ji_even
            prev_ji_even = b * prev_ji_input_even
            ji += prev_ji_even
            prev_ji_input_even = i1_for_even_prev3
            ji *= adjusted_prev_period

            hilbert_temp = a * q1
            jq = -jq_even[hilbert_idx]
            jq_even[hilbert_idx] = hilbert_temp
            jq += hilbert_temp
            jq -= prev_jq_even
            prev_jq_even = b * prev_jq_input_even
            jq += prev_jq_even
            prev_jq_input_even = q1
            jq *= adjusted_prev_period

            hilbert_idx += 1
            if hilbert_idx == 3:
                hilbert_idx = 0

            q2 = (0.2 * (q1 + ji)) + (0.8 * prev_q2)
            i2 = (0.2 * (i1_for_even_prev3 - jq)) + (0.8 * prev_i2)

            i1_for_odd_prev3 = i1_for_odd_prev2
            i1_for_odd_prev2 = detrender
        else:
            # Odd.
            hilbert_temp = a * smoothed_value
            detrender = -detrender_odd[hilbert_idx]
            detrender_odd[hilbert_idx] = hilbert_temp
            detrender += hilbert_temp
            detrender -= prev_detrender_odd
            prev_detrender_odd = b * prev_detrender_input_odd
            detrender += prev_detrender_odd
            prev_detrender_input_odd = smoothed_value
            detrender *= adjusted_prev_period

            hilbert_temp = a * detrender
            q1 = -q1_odd[hilbert_idx]
            q1_odd[hilbert_idx] = hilbert_temp
            q1 += hilbert_temp
            q1 -= prev_q1_odd
            prev_q1_odd = b * prev_q1_input_odd
            q1 += prev_q1_odd
            prev_q1_input_odd = detrender
            q1 *= adjusted_prev_period

            hilbert_temp = a * i1_for_odd_prev3
            ji = -ji_odd[hilbert_idx]
            ji_odd[hilbert_idx] = hilbert_temp
            ji += hilbert_temp
            ji -= prev_ji_odd
            prev_ji_odd = b * prev_ji_input_odd
            ji += prev_ji_odd
            prev_ji_input_odd = i1_for_odd_prev3
            ji *= adjusted_prev_period

            hilbert_temp = a * q1
            jq = -jq_odd[hilbert_idx]
            jq_odd[hilbert_idx] = hilbert_temp
            jq += hilbert_temp
            jq -= prev_jq_odd
            prev_jq_odd = b * prev_jq_input_odd
            jq += prev_jq_odd
            prev_jq_input_odd = q1
            jq *= adjusted_prev_period

            q2 = (0.2 * (q1 + ji)) + (0.8 * prev_q2)
            i2 = (0.2 * (i1_for_odd_prev3 - jq)) + (0.8 * prev_i2)

            i1_for_even_prev3 = i1_for_even_prev2
            i1_for_even_prev2 = detrender

        # Period update.
        re = (0.2 * ((i2 * prev_i2) + (q2 * prev_q2))) + (0.8 * re)
        im = (0.2 * ((i2 * prev_q2) - (q2 * prev_i2))) + (0.8 * im)
        prev_q2 = q2
        prev_i2 = i2

        temp_real = period
        if (im != 0.0) and (re != 0.0):
            period = 360.0 / (math.atan(im / re) * rad2deg)

        temp_real2 = 1.5 * temp_real
        if period > temp_real2:
            period = temp_real2
        temp_real2 = 0.67 * temp_real
        if period < temp_real2:
            period = temp_real2
        if period < 6.0:
            period = 6.0
        elif period > 50.0:
            period = 50.0
        period = (0.2 * period) + (0.8 * temp_real)

        smooth_period = (0.33 * period) + (0.67 * smooth_period)

        # Dominant cycle phase (DFT of smoothPrice).
        dc_period = smooth_period + 0.5
        dc_period_int = int(dc_period)
        real_part = 0.0
        imag_part = 0.0

        idx = smooth_price_idx
        for i in range(dc_period_int):
            ang = (float(i) * const_deg2rad_by_360) / float(dc_period_int)
            x = smooth_price[idx]
            real_part += math.sin(ang) * x
            imag_part += math.cos(ang) * x
            if idx == 0:
                idx = max_smooth_idx
            else:
                idx -= 1

        abs_im = math.fabs(imag_part)
        if abs_im > 0.0:
            dc_phase = math.atan(real_part / imag_part) * rad2deg
        elif abs_im <= 0.01:
            if real_part < 0.0:
                dc_phase -= 90.0
            elif real_part > 0.0:
                dc_phase += 90.0
        dc_phase += 90.0

        # Compensate for one bar lag of the WMA.
        dc_phase += 360.0 / smooth_period
        if imag_part < 0.0:
            dc_phase += 180.0
        if dc_phase > 315.0:
            dc_phase -= 360.0

        if today >= start_idx:
            out[today] = dc_phase

        # Next bar.
        smooth_price_idx += 1
        if smooth_price_idx > max_smooth_idx:
            smooth_price_idx = 0
        today += 1


def HT_DCPHASE(real):
    """
    Hilbert Transform - Dominant Cycle Phase
    """
    real_arr = as_1d_float64(real)
    out = nan_like(real_arr, dtype=np.float64)
    _ht_dcphase_kernel(real_arr, out)
    return out


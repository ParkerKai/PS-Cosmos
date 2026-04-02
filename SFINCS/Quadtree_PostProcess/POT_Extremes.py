import numpy as np
import xarray as xr
from typing import Tuple, Optional

# ----------------------------
# Utilities
# ----------------------------

_UNIT_TO_NS = {
    "ns": 1,
    "us": 1_000,
    "µs": 1_000,  # microseconds alt symbol
    "ms": 1_000_000,
    "s": 1_000_000_000,
    "m": 60 * 1_000_000_000,
    "h": 3600 * 1_000_000_000,
    "D": 86400 * 1_000_000_000,
    "d": 86400 * 1_000_000_000,
}


def _parse_timedelta_to_ns(r: str | np.timedelta64) -> int:
    """
    Parse duration to nanoseconds (int). Supports numpy.timedelta64 or simple strings (e.g., '24h').
    Integer magnitudes only for strings.
    """
    import pandas as pd

    if isinstance(r, np.timedelta64):
        # Direct cast to ns then to int
        return int(r.astype("timedelta64[ns]").astype(np.int64))

    # Optional: support pandas Timedelta
    if isinstance(r, pd.Timedelta):
        return int(r.to_numpy().astype("timedelta64[ns]").astype(np.int64))

    if not isinstance(r, str):
        raise ValueError(f"`r` must be str, numpy.timedelta64, or pandas.Timedelta; got {type(r)}")

    s = r.strip()
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    if i == 0 or i == len(s):
        raise ValueError(f"Invalid time delta string: {r!r}")

    magnitude = s[:i]
    unit = s[i:].strip()
    if unit not in _UNIT_TO_NS:
        raise ValueError(f"Unsupported unit in {r!r}. Use one of {sorted(_UNIT_TO_NS)}")

    try:
        mag = int(magnitude)
    except Exception:
        raise ValueError(f"Time delta magnitude must be an integer in {r!r}")

    return mag * _UNIT_TO_NS[unit]



def _validate_1d_time_series(da: xr.DataArray, time_dim: str) -> None:
    """Ensure DataArray is 1D along `time_dim` and coordinate is datetime64."""
    if time_dim not in da.dims:
        raise ValueError(
            f"`time_dim` {time_dim!r} not found in DataArray"
        )
    
    if not np.issubdtype(da.dtype, np.number):
        raise TypeError(f"DataArray values must be numeric; got dtype={da.dtype}")
    
    # Require ONLY the time dimension (1D series)
    if da.ndim != 1 or da.dims[0] != time_dim:
        raise ValueError(
            "Function expects a 1D DataArray indexed by `time_dim` only.\n"
            f"Got dims={da.dims}. If multi-dimensional, select a single location first, e.g., "
            "da.sel(lat=..., lon=...)"
        )

    coord = da[time_dim]
    if not np.issubdtype(coord.dtype, np.datetime64):
        raise TypeError(
            f"Coordinate {time_dim!r} must be datetime64; got dtype={coord.dtype}"
        )

    # Ensure strictly increasing time (required for gap comparison)
    t = coord.values
    if t.size < 2:
        return
    if not np.all(np.diff(t).astype("timedelta64[ns]") > np.timedelta64(0, "ns")):
        raise ValueError("Time coordinate must be strictly increasing.")


def _cluster_extrema_1d(
    values: np.ndarray,
    times: np.ndarray,
    threshold: float,
    r_ns: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given 1D values and datetime64 times, return cluster maxima (times, values)
    under a strict threshold (> threshold) with independence window r_ns.
    """
    # Exceedances mask
    mask = (values > float(threshold)) & np.isfinite(values)
    if not np.any(mask):
        # No exceedances
        return np.array([], dtype=times.dtype), np.array([], dtype=np.float64)

    ex_vals = values[mask]
    ex_times = times[mask]

    if ex_vals.size == 1:
        return ex_times.copy(), ex_vals.astype(np.float64)

    # Compute gaps between consecutive exceedances
    dt_ns = (
        ex_times[1:].astype("datetime64[ns]") - ex_times[:-1].astype("datetime64[ns]")
    ).astype(np.int64)
    gap_indices = np.flatnonzero(dt_ns > r_ns)

    # Build clusters by slicing between gaps
    cluster_starts = np.r_[0, gap_indices + 1]
    cluster_ends = np.r_[gap_indices + 1, ex_vals.size]  # exclusive end
    out_times = []
    out_vals = []

    for s, e in zip(cluster_starts, cluster_ends):
        cvals = ex_vals[s:e]
        ctimes = ex_times[s:e]
        imax = np.nanargmax(cvals)
        out_vals.append(float(cvals[imax]))
        out_times.append(ctimes[imax])

    return np.array(out_times, dtype=times.dtype), np.array(out_vals, dtype=np.float64)


def _count_peaks_1d(
    values: np.ndarray, times: np.ndarray, threshold: float, r_ns: int
) -> int:
    """Count declustered peaks for given threshold (helper for search)."""
    t_out, v_out = _cluster_extrema_1d(values, times, threshold, r_ns)
    return int(v_out.size)


def get_extremes_pot_xr(
    da: xr.DataArray,
    threshold: float,
    r: str = "24h",
    time_dim: str = "time",
    num_exce: Optional[int] = None,
) -> xr.DataArray:
    """
    Decluster exceedances (Peaks Over Threshold) and return cluster maxima as an xr.DataArray.

    Parameters
    ----------
    da : xr.DataArray
        A **1D** data array indexed by `time_dim` with dtype numeric.
    threshold : float
        Strict threshold; values > threshold are candidates.
    r : str, default '24h'
        Independence window (e.g., '24h', '48h', '3600s', '1D'). Integer magnitudes only.
    time_dim : str, default 'time'
        Name of the time dimension.
    num_exce : int, optional
        If provided, limit the output to the **top `num_exce` declustered peaks by value**.
        Ties are broken by **latest** time. Returned events are sorted back to **chronological**
        order for usability. If `num_exce` >= number of declustered peaks, all peaks are returned.

    Returns
    -------
    extremes : xr.DataArray
        1D DataArray of extreme values indexed by event times (subset of `time_dim`).

    Notes
    -----
    - If you need multi-dimensional declustering (e.g., time × lat × lon),
      apply location selection first: `get_extremes_pot_xr(da.sel(lat=..., lon=...), ...)`
    """
    _validate_1d_time_series(da, time_dim)
    r_ns = _parse_timedelta_to_ns(r)

    values = da.values.astype(float)
    times = da[time_dim].values

    evt_times, evt_vals = _cluster_extrema_1d(values, times, threshold, r_ns)

    if evt_vals.size == 0:
        raise ValueError("Threshold yields zero exceedances.")

    # Optional: cap to exact number of extremes by selecting top-k by value
    if num_exce is not None:
        if not isinstance(num_exce, int) or num_exce <= 0:
            raise ValueError("`num_exce` must be a positive integer when provided.")
        if evt_vals.size > num_exce:
            # Sort by value desc, then time desc (prefer later time),
            # then return selected events in chronological order.
            t_ns = evt_times.astype("datetime64[ns]").astype(np.int64)
            order_by_value_then_time = np.lexsort((-t_ns, -evt_vals))
            select_idx = order_by_value_then_time[:num_exce]
            chrono = np.argsort(t_ns[select_idx])
            evt_times = evt_times[select_idx][chrono]
            evt_vals = evt_vals[select_idx][chrono]

    extremes = xr.DataArray(
        data=evt_vals,
        coords={time_dim: evt_times},
        dims=(time_dim,),
        name=da.name if da.name is not None else "extreme_values",
        attrs=dict(**da.attrs),
    )
    return extremes



def pot_threshold_set_num_xr(
    da: xr.DataArray,
    r: str,
    num_exce: int,
    time_dim: str = "time",
    strategy: str = "geq",
) -> float:
    """
    Select a threshold for POT declustering to obtain ~`num_exce` extreme events (1D series).

    Parameters
    ----------
    da : xr.DataArray
        **1D** numeric DataArray indexed by `time_dim`.
    r : str
        Independence window (e.g., '24h').
    num_exce : int
        Desired number of declustered peaks.
    time_dim : str, default 'time'
        Name of time dimension.
    strategy : {'geq','leq','closest'}, default 'geq'
        - 'geq': smallest threshold that yields >= num_exce peaks (conservative).
        - 'leq': largest threshold that yields <= num_exce peaks.
        - 'closest': threshold minimizing |count - num_exce|.

    Returns
    -------
    threshold : float
        Selected threshold (data units).

    Notes
    -----
    - Uses **binary search** over the sorted unique data values (monotonic non-increasing counts).
    - Robust to NaNs; ignores non-finite values.
    """
    if not isinstance(num_exce, int) or num_exce < 0:
        raise ValueError("`num_exce` must be a non-negative integer.")

    _validate_1d_time_series(da, time_dim)
    r_ns = _parse_timedelta_to_ns(r)

    values = da.values.astype(float)
    times = da[time_dim].values

    vals = values[np.isfinite(values)]
    if vals.size == 0:
        raise ValueError("Input contains no finite values.")
    unique_vals = np.unique(vals)
    unique_vals.sort()  # ascending

    # Helper: count peaks at a data threshold
    def count_at(i: int) -> int:
        return _count_peaks_1d(values, times, unique_vals[i], r_ns)

    # Monotonic property: counts decrease as threshold increases.
    # Binary search depends on strategy.
    if strategy == "geq":
        # Find the first threshold achieving >= num_exce
        lo, hi = 0, unique_vals.size - 1
        ans = None
        while lo <= hi:
            mid = (lo + hi) // 2
            c = count_at(mid)
            if c >= num_exce:
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        if ans is None:
            # Never achieves >= num_exce; return highest threshold
            return float(unique_vals[-1])
        return float(unique_vals[ans])

    elif strategy == "leq":
        # Find the last threshold with <= num_exce
        lo, hi = 0, unique_vals.size - 1
        ans = None
        while lo <= hi:
            mid = (lo + hi) // 2
            c = count_at(mid)
            if c <= num_exce:
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if ans is None:
            # Always > num_exce; return smallest threshold
            return float(unique_vals[0])
        return float(unique_vals[ans])

    
    elif strategy == "closest":
            lo, hi = 0, unique_vals.size - 1
            geq_idx = None
            while lo <= hi:
                mid = (lo + hi) // 2
                c = count_at(mid)
                if c >= num_exce:
                    geq_idx = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            candidates = []
            if geq_idx is not None:
                candidates.append(geq_idx)
                if geq_idx + 1 < unique_vals.size:
                    candidates.append(geq_idx + 1)
            else:
                # Never reaches >= num_exce: choose the smallest threshold (max counts)
                candidates.append(0)
            best_i, best_diff = None, None
            for i in candidates:
                diff = abs(count_at(i) - num_exce)
                if best_diff is None or diff < best_diff:
                    best_diff, best_i = diff, i
            return float(unique_vals[best_i])


    else:
        raise ValueError("`strategy` must be one of {'geq','leq','closest'}")


def pot_threshold_set_num_map(
    da: xr.DataArray,
    r: str,
    num_exce: int,
    time_dim: str = "time",
    strategy: str = "geq",
    dask_parallelized: bool = True,
) -> xr.DataArray:
    """
    Map deterministic threshold selection across all non-time dims of a multi-dimensional DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Multi-dimensional numeric DataArray with `time_dim` present (e.g., time × lat × lon).
    r : str
        Independence window (e.g., '24h').
    num_exce : int
        Desired number of declustered peaks per location.
    time_dim : str, default 'time'
        Name of the time dimension.
    strategy : {'geq','leq','closest'}, default 'geq'
    dask_parallelized : bool, default True
        If da is Dask-backed, allow parallelized execution.

    Returns
    -------
    thresh_da : xr.DataArray
        A DataArray of thresholds for each non-time point (dims: da.dims without `time_dim`).

    Notes
    -----
    - This returns **thresholds per location** (scalar per point).
    - Declustering returns ragged event lists; those cannot be stored densely with xarray.
      For per-location extremes, loop over selected points or use apply_ufunc on coordinate indices.
    """
    if time_dim not in da.dims:
        raise ValueError(f"`time_dim` {time_dim!r} not found in da.dims={da.dims}")
    if not np.issubdtype(da[time_dim].dtype, np.datetime64):
        raise TypeError(
            f"Coordinate {time_dim!r} must be datetime64; got dtype={da[time_dim].dtype}"
        )
    if not isinstance(num_exce, int) or num_exce < 0:
        raise ValueError("`num_exce` must be a non-negative integer.")

    r_ns = _parse_timedelta_to_ns(r)

    def _threshold_ufunc(values_1d: np.ndarray, times_1d: np.ndarray) -> np.float64:
        # Require strictly increasing time per vector
        if values_1d.size == 0:
            return np.nan
        t = times_1d
        if t.size >= 2:
            dt_ns = (
                t[1:].astype("datetime64[ns]") - t[:-1].astype("datetime64[ns]")
            ).astype(np.int64)
            if not np.all(dt_ns > 0):
                # Attempt to sort by time if unsorted
                order = np.argsort(t)
                t = t[order]
                values_1d = values_1d[order]

        vals = values_1d[np.isfinite(values_1d)]
        if vals.size == 0:
            return np.nan
        unique_vals = np.unique(vals)
        unique_vals.sort()

        def count_at(i: int) -> int:
            return _count_peaks_1d(values_1d, t, float(unique_vals[i]), r_ns)

        # Binary searches as above
        if strategy == "geq":
            lo, hi = 0, unique_vals.size - 1
            ans = None
            while lo <= hi:
                mid = (lo + hi) // 2
                c = count_at(mid)
                if c >= num_exce:
                    ans = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            return np.float64(
                unique_vals[ans if ans is not None else unique_vals.size - 1]
            )

        elif strategy == "leq":
            lo, hi = 0, unique_vals.size - 1
            ans = None
            while lo <= hi:
                mid = (lo + hi) // 2
                c = count_at(mid)
                if c <= num_exce:
                    ans = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            return np.float64(unique_vals[ans if ans is not None else 0])

        elif strategy == "closest":
            lo, hi = 0, unique_vals.size - 1
            geq_idx = None
            while lo <= hi:
                mid = (lo + hi) // 2
                c = count_at(mid)
                if c >= num_exce:
                    geq_idx = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            candidates = []
            if geq_idx is not None:
                candidates.append(geq_idx)
                if geq_idx + 1 < unique_vals.size:
                    candidates.append(geq_idx + 1)
            else:
                # Never reaches >= num_exce: choose the smallest threshold (max counts)
                candidates.append(0)
            best_i, best_diff = None, None
            for i in candidates:
                diff = abs(count_at(i) - num_exce)
                if best_diff is None or diff < best_diff:
                    best_diff, best_i = diff, i
            return np.float64(unique_vals[best_i])

        else:
            return np.nan

    # Apply across all non-time dims
    input_core_dims = [[time_dim], [time_dim]]
    output_core_dims = [[]]  # scalar per location
    kwargs = {}
    dask_opt = "parallelized" if dask_parallelized else None

    thresh = xr.apply_ufunc(
        _threshold_ufunc,
        da,
        da[time_dim],
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        vectorize=True,
        dask=dask_opt,
        output_dtypes=[np.float64],
    )

    # Name and attrs
    thresh.name = (da.name or "var") + "_threshold"
    return thresh

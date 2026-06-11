"""
PS-CoSMoS post-processing helpers.

PS-CoSMoS-specific routines used by `downscale_sfincs_results.py`. Generic
downscaling and disconnected-flood removal live in
`hydromt_sfincs.workflows.downscaling`; this module covers:

  - Water-year aggregation of SFINCS NetCDF outputs (per-cell annual maxima
    of zsmax + extras at the peak time).
  - Cell-level extreme value analysis: empirical Weibull plotting position,
    per-cell GEV fit, and Peaks-Over-Threshold (POT) with declustering.
  - Mapping quadtree-grid variables (e.g. qmax, tmax) to a high-resolution
    DEM grid via a pre-computed index COG.
  - Hazard-category binning of depth / velocity rasters and a unified
    Cloud-Optimized GeoTIFF writer.
  - Infastructure to convert COG rasters to vectorized shapefiles with geopandas and rasterio.

Sections 0a-0e inline utilities formerly kept in `POT_Extremes.py` and
`Xarray_NCtools.py` so the whole PS-CoSMoS-specific stack is one import.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple
import os
from dataclasses import dataclass
import tempfile
import shutil

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from scipy.stats import genextreme
from scipy.ndimage import gaussian_filter
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes, sieve
from rasterio.windows import Window



# =============================================================================
# SECTION 0: Inlined utilities
# =============================================================================

# --- 0a: time-delta parsing + 1D series validation --------------------------

_UNIT_TO_NS = {
    "ns": 1,
    "us": 1_000,
    "µs": 1_000,
    "ms": 1_000_000,
    "s": 1_000_000_000,
    "m": 60 * 1_000_000_000,
    "h": 3600 * 1_000_000_000,
    "D": 86_400 * 1_000_000_000,
    "d": 86_400 * 1_000_000_000,
}


def rp_tag(rp: float) -> str:
    return f"RP{int(round(rp)):03d}"


def _parse_timedelta_to_ns(r) -> int:
    """Parse '24h' / '72h' / timedelta64 / pd.Timedelta to nanoseconds."""
    if isinstance(r, np.timedelta64):
        return int(r.astype("timedelta64[ns]").astype(np.int64))
    if isinstance(r, pd.Timedelta):
        return int(r.to_numpy().astype("timedelta64[ns]").astype(np.int64))
    if not isinstance(r, str):
        raise ValueError(f"`r` must be str/timedelta64/Timedelta; got {type(r)}")

    s = r.strip()
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    if i == 0 or i == len(s):
        raise ValueError(f"Invalid time delta string: {r!r}")
    mag = int(s[:i])
    unit = s[i:].strip()
    if unit not in _UNIT_TO_NS:
        raise ValueError(f"Unsupported unit in {r!r}; use {sorted(_UNIT_TO_NS)}")
    return mag * _UNIT_TO_NS[unit]


def _validate_1d_time_series(da: xr.DataArray, time_dim: str) -> None:
    if time_dim not in da.dims:
        raise ValueError(f"`time_dim` {time_dim!r} not in DataArray dims")
    if not np.issubdtype(da.dtype, np.number):
        raise TypeError(f"DataArray must be numeric; got dtype={da.dtype}")
    if da.ndim != 1 or da.dims[0] != time_dim:
        raise ValueError(
            f"Expected 1D DataArray indexed by {time_dim!r}; got dims={da.dims}"
        )
    coord = da[time_dim]
    if not np.issubdtype(coord.dtype, np.datetime64):
        raise TypeError(f"{time_dim!r} must be datetime64; got {coord.dtype}")
    t = coord.values
    if t.size >= 2 and not np.all(
        np.diff(t).astype("timedelta64[ns]") > np.timedelta64(0, "ns")
    ):
        raise ValueError("Time coordinate must be strictly increasing.")


# --- Assumes you already have rp_tag(rp: float) -> str defined elsewhere ---
# Example fallback if needed:
# def rp_tag(rp: float, decimals: int = 0) -> str:
#     """Safe tag for floats: 2.5 -> 'rp2p5'; 100 -> 'rp100'."""
#     s = f"{rp:.{decimals}f}" if decimals > 0 else str(int(rp)) if rp.is_integer() else str(rp)
#     return f"rp{s.replace('.', 'p')}"


@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path
    shapefile_dir: Path
    domain_stem: str
    provenance_tag: str
    raster_ext: str = ".tif"
    vector_ext: str = ".shp"

    # --- Core builders ---
    def raster(self, name: str, rp: float) -> Path:
        return (
            self.output_dir
            / f"{name}_{rp_tag(rp)}_{self.domain_stem}_{self.provenance_tag}{self.raster_ext}"
        )

    def vector(self, name: str, rp: float) -> Path:
        return (
            self.shapefile_dir
            / f"{name}_{rp_tag(rp)}_{self.domain_stem}_{self.provenance_tag}{self.vector_ext}"
        )

    # --- Convenience for masked variants ---
    @staticmethod
    def masked(p: Path) -> Path:
        return p.parent / f"{p.stem}_masked{p.suffix}"

    # --- Specific raster helpers (mirroring your original names) ---
    def hmax_path(self, rp: float) -> Path:
        return self.raster("hmax", rp)

    def zsmax_path(self, rp: float) -> Path:
        return self.raster("zsmax", rp)

    def hmax_masked_path(self, rp: float) -> Path:
        return self.masked(self.hmax_path(rp))

    def zsmax_masked_path(self, rp: float) -> Path:
        return self.masked(self.zsmax_path(rp))

    def connection_path(self, rp: float) -> Path:
        return self.raster("connection", rp)

    def extra_path(self, var: str, rp: float) -> Path:
        return self.raster(var, rp)

    def depth_bins_path(self, rp: float) -> Path:
        return self.raster("depth_bins", rp)

    def qmax_bins_path(self, rp: float) -> Path:
        return self.raster("qmax_bins", rp)

    # --- Vector (shapefile) helpers ---
    def depth_shapefile_path(self, rp: float) -> Path:
        return self.vector("depth_bins", rp)

    def qmax_shapefile_path(self, rp: float) -> Path:
        return self.vector("qmax_bins", rp)

    def extent_connected_shapefile_path(self, rp: float) -> Path:
        return self.vector("extent_connected", rp)

    def extent_disconnected_shapefile_path(self, rp: float) -> Path:
        return self.vector("extent_disconnected", rp)

    def extent_min_shapefile_path(self, rp: float) -> Path:
        return self.vector("extent_min", rp)

    def extent_max_shapefile_path(self, rp: float) -> Path:
        return self.vector("extent_max", rp)

    # --- Utilities ---
    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shapefile_dir.mkdir(parents=True, exist_ok=True)


# --- 0b: POT declustering + threshold search --------------------------------


def _cluster_extrema_1d(
    values: np.ndarray,
    times: np.ndarray,
    threshold: float,
    r_ns: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return cluster maxima (times, values) for values > threshold with
    independence window `r_ns` (nanoseconds)."""
    mask = (values > float(threshold)) & np.isfinite(values)
    if not np.any(mask):
        return np.array([], dtype=times.dtype), np.array([], dtype=np.float64)

    ex_vals = values[mask]
    ex_times = times[mask]
    if ex_vals.size == 1:
        return ex_times.copy(), ex_vals.astype(np.float64)

    dt_ns = (
        ex_times[1:].astype("datetime64[ns]") - ex_times[:-1].astype("datetime64[ns]")
    ).astype(np.int64)
    gap_idx = np.flatnonzero(dt_ns > r_ns)

    starts = np.r_[0, gap_idx + 1]
    ends = np.r_[gap_idx + 1, ex_vals.size]
    out_t, out_v = [], []
    for s, e in zip(starts, ends):
        cvals = ex_vals[s:e]
        ctimes = ex_times[s:e]
        imax = np.nanargmax(cvals)
        out_v.append(float(cvals[imax]))
        out_t.append(ctimes[imax])
    return np.array(out_t, dtype=times.dtype), np.array(out_v, dtype=np.float64)


def _count_peaks_1d(values, times, threshold, r_ns) -> int:
    _, v = _cluster_extrema_1d(values, times, threshold, r_ns)
    return int(v.size)


def get_extremes_pot_xr(
    da: xr.DataArray,
    threshold: float,
    r: str = "24h",
    time_dim: str = "time",
    num_exce: Optional[int] = None,
) -> xr.DataArray:
    """Decluster Peaks-Over-Threshold from a 1D xr.DataArray."""
    _validate_1d_time_series(da, time_dim)
    r_ns = _parse_timedelta_to_ns(r)
    values = da.values.astype(float)
    times = da[time_dim].values
    evt_t, evt_v = _cluster_extrema_1d(values, times, threshold, r_ns)
    if evt_v.size == 0:
        raise ValueError("Threshold yields zero exceedances.")
    if num_exce is not None:
        if not isinstance(num_exce, int) or num_exce <= 0:
            raise ValueError("`num_exce` must be a positive integer.")
        if evt_v.size > num_exce:
            t_ns = evt_t.astype("datetime64[ns]").astype(np.int64)
            order = np.lexsort((-t_ns, -evt_v))
            sel = order[:num_exce]
            chrono = np.argsort(t_ns[sel])
            evt_t = evt_t[sel][chrono]
            evt_v = evt_v[sel][chrono]
    return xr.DataArray(
        data=evt_v,
        coords={time_dim: evt_t},
        dims=(time_dim,),
        name=da.name or "extreme_values",
        attrs=dict(**da.attrs),
    )


def pot_threshold_set_num_xr(
    da: xr.DataArray,
    r: str,
    num_exce: int,
    time_dim: str = "time",
    strategy: Literal["geq", "leq", "closest"] = "closest",
) -> float:
    """Binary-search a threshold yielding ~num_exce declustered peaks."""
    if not isinstance(num_exce, int) or num_exce < 0:
        raise ValueError("`num_exce` must be a non-negative integer.")
    _validate_1d_time_series(da, time_dim)
    r_ns = _parse_timedelta_to_ns(r)

    values = da.values.astype(float)
    times = da[time_dim].values
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Input contains no finite values.")
    uniq = np.sort(finite)

    if np.unique(uniq).size == 1:
        u = float(np.unique(uniq)[0])
        return float(np.nextafter(u, -np.inf))

    def count_at(i: int) -> int:
        return _count_peaks_1d(values, times, uniq[i], r_ns)

    n = uniq.size

    if strategy == "geq":
        lo, hi, ans = 0, n - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if count_at(mid) >= num_exce:
                ans, hi = mid, mid - 1
            else:
                lo = mid + 1
        return float(uniq[ans] if ans is not None else uniq[-1])

    if strategy == "leq":
        lo, hi, ans = 0, n - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if count_at(mid) <= num_exce:
                ans, lo = mid, mid + 1
            else:
                hi = mid - 1
        return float(uniq[ans] if ans is not None else uniq[0])

    if strategy == "closest":
        lo, hi, geq_idx = 0, n - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if count_at(mid) >= num_exce:
                geq_idx, hi = mid, mid - 1
            else:
                lo = mid + 1
        candidates = [geq_idx, geq_idx + 1] if geq_idx is not None else [0]
        candidates = [c for c in candidates if c is not None and 0 <= c < n]
        best_i, best_diff = candidates[0], abs(count_at(candidates[0]) - num_exce)
        for c in candidates[1:]:
            d = abs(count_at(c) - num_exce)
            if d < best_diff:
                best_i, best_diff = c, d
        return float(uniq[best_i])

    raise ValueError("`strategy` must be one of {'geq','leq','closest'}.")


def rp_axis(
    n: int,
    plotting: Literal["weibull", "gringorten"] = "weibull",
    order: Literal["ascending", "descending"] = "descending",
) -> np.ndarray:
    """Plotting-position return-period axis for n ranked samples."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")
    m = np.arange(1, n + 1, dtype=float)
    if plotting == "weibull":
        T = (n + 1.0) / m
    elif plotting == "gringorten":
        T = (n - 0.12) / (m - 0.44)
    else:
        raise ValueError("plotting must be 'weibull' or 'gringorten'.")
    if order == "ascending":
        T = T[::-1]
    elif order != "descending":
        raise ValueError("order must be 'ascending' or 'descending'.")
    return T


# --- 0d: NetCDF validation + time helpers (from Xarray_NCtools.py) ----------

_TIME_CANDIDATES: List[str] = ["timemax", "time", "Time", "t", "datetime", "date"]


def _detect_time_coord(
    ds: xr.Dataset, preferred: Optional[str] = None
) -> Optional[str]:
    if preferred and preferred in ds.coords:
        return preferred
    for name in _TIME_CANDIDATES:
        if name in ds.coords:
            return name
    return None


def check_nc_file(
    path,
    required_vars=None,
    required_coords=None,
    check_time=True,
    sample_data=False,
):
    """Lightweight validation of a single SFINCS NetCDF file."""
    issues: List[str] = []
    summary = {"path": str(path)}
    try:
        ds = xr.open_dataset(
            path, decode_cf=True, mask_and_scale=True, engine="netcdf4"
        )
    except Exception as e:
        issues.append(f"OPEN_ERROR: {type(e).__name__}: {e}")
        return False, issues, summary

    summary["dims"] = dict(ds.sizes)
    summary["coords"] = list(ds.coords)
    summary["vars"] = list(ds.data_vars)
    summary["dtypes"] = {v: str(ds[v].dtype) for v in ds.variables}

    for d, n in ds.sizes.items():
        if n <= 0:
            issues.append(f"DIM_EMPTY: {d}={n}")
    for v in required_vars or []:
        if v not in ds.variables:
            issues.append(f"MISSING_VAR: {v}")
    for c in required_coords or []:
        if c not in ds.coords:
            issues.append(f"MISSING_COORD: {c}")

    if check_time:
        tname = _detect_time_coord(ds)
        if tname is not None:
            try:
                t = ds[tname].values
                if np.issubdtype(t.dtype, np.datetime64):
                    if np.any(np.diff(t) < np.timedelta64(0, "ns")):
                        issues.append(f"TIME_NON_MONOTONIC: {tname}")
                else:
                    if np.any(np.diff(t) < 0):
                        issues.append(f"TIME_NON_MONOTONIC: {tname}")
            except Exception as e:
                issues.append(f"TIME_READ_ERROR: {type(e).__name__}: {e}")

    if sample_data:
        try:
            for v in list(ds.data_vars)[:5]:
                slc = {d: 0 for d in ds[v].dims}
                _ = ds[v].isel(**slc).values
        except Exception as e:
            issues.append(f"DATA_SAMPLE_ERROR: {type(e).__name__}: {e}")

    ds.close()
    return len(issues) == 0, issues, summary


def batch_check_nc_files(
    files,
    required_vars=None,
    required_coords=None,
    check_time=True,
    sample_data=False,
):
    """Run check_nc_file across many files; return (good_files, report)."""
    report = []
    good_files = []
    for p in files:
        ok, issues, _ = check_nc_file(
            p,
            required_vars=required_vars,
            required_coords=required_coords,
            check_time=check_time,
            sample_data=sample_data,
        )
        report.append({"path": str(p), "ok": ok, "issues": issues})
        if ok:
            good_files.append(p)
    return good_files, report


def ensure_unique_sorted_time(
    ds: xr.Dataset, time_name: Optional[str] = None, keep: str = "first"
) -> xr.Dataset:
    """Drop duplicate timestamps and sort along the time coord."""
    tname = _detect_time_coord(ds, preferred=time_name)
    if tname is None:
        return ds
    pdt = pd.to_datetime(ds[tname].values)
    dup = pd.Series(pdt).duplicated(keep=keep).to_numpy()
    if dup.any():
        ds = ds.isel({tname: ~dup})
    return ds.sortby(tname)


# --- 0e: Water-year mode trimming (from part1.preprocess_) ------------------


def trim_to_mode_year(
    ds: xr.Dataset,
    time_name: Optional[str] = None,
    tie_break: Literal["earliest", "latest"] = "earliest",
) -> xr.Dataset:
    """Keep only timestamps falling in the most common calendar year."""
    tname = _detect_time_coord(ds, preferred=time_name)
    if tname is None:
        return ds
    if tname not in ds.coords:
        try:
            ds = ds.set_coords(tname)
        except Exception:
            return ds

    years = ds[tname].dt.year
    yvals = years.values
    if yvals.size == 0:
        return ds
    uniq, occ = np.unique(yvals, return_counts=True)
    candidates = uniq[occ == occ.max()]
    mode_year = int(candidates.max() if tie_break == "latest" else candidates.min())
    return ds.where(years == mode_year, drop=True)


# =============================================================================
# SECTION 1: Water-year aggregation
# =============================================================================

# def _open_sfincs_map(
#     sfincs_dir: Path,
#     vars_keep: Sequence[str],
#     time_dim: str,
# ) -> xr.Dataset:
#     """Open one SFINCS run's sfincs_map.nc, trim to its mode year, keep vars."""
#     nc_path = Path(sfincs_dir) / "sfincs_map.nc"
#     ds = xr.open_dataset(nc_path, decode_cf=True, mask_and_scale=True, engine="netcdf4")
#     keep = [v for v in vars_keep if v in ds]
#     if keep:
#         ds = ds[keep]
#     ds = trim_to_mode_year(ds, time_name=time_dim)
#     ds = ensure_unique_sorted_time(ds, time_name=time_dim)
#     return ds


def _open_sfincs_map(
    sfincs_dir: Path,
    vars_keep: Sequence[str],
    time_dim: str,
) -> xr.Dataset:
    """
    Open one SFINCS run's sfincs_map.nc, sanitize/convert the time coordinate,
    trim to its mode year, and keep the requested variables.
    """

    nc_path = Path(sfincs_dir) / "sfincs_map.nc"

    # 1) Open WITHOUT time decoding to avoid Overflow/OutOfBounds during CF decoding
    ds = xr.open_dataset(
        nc_path,
        decode_cf=True,  # still decode CF for other vars
        mask_and_scale=True,  # respect scales for data vars
        engine="netcdf4",
        decode_times=False,  # IMPORTANT: prevent immediate time decoding
    )

    # 2) Sanitize and decode the time coordinate named by `time_dim`
    if time_dim in ds:
        t = ds[time_dim]

        # --- 2a) Build a mask for bad/fill values ---------------------------------
        # Common NetCDF float32 fill value used by several tools:
        COMMON_FILL = np.float32(9.96921e36)

        # Read declared fill/missing markers if present
        declared_fill = t.attrs.get("_FillValue", t.attrs.get("missing_value", None))

        # Work with float for safety; ints can overflow on comparators
        t_float = t.astype("float64")

        # Define a generous threshold: 1e12 seconds ≈ 31,688 years
        # Adjust if you want tighter bounds
        max_seconds = 1e12

        mask = ~np.isfinite(t_float)
        mask |= t_float == COMMON_FILL
        if declared_fill is not None:
            # If declared_fill is an array, compare elementwise; if scalar, broadcast
            mask |= t_float == np.asarray(declared_fill, dtype="float64")

        # Mask extreme magnitudes (likely corrupted/fill)
        mask |= (t_float > max_seconds) | (t_float < -max_seconds)

        # Apply mask; set bad entries to NaN so we can decode safely
        t_clean = xr.where(~mask, t_float, np.nan)

        # --- 2b) Decode to datetime -----------------------------------------------
        units = t.attrs.get("units", None)
        calendar = t.attrs.get("calendar", "standard")  # default

        if not units or "since" not in units:
            # Fallback: treat values as "seconds since 1970-01-01" if units missing
            origin = np.datetime64("1970-01-01T00:00:00")
            t_dt = origin + t_clean.astype("timedelta64[s]")
        else:
            # Parse CF units: "<time_units> since <YYYY-MM-DD HH:MM:SS>"
            # Normalize to ISO "YYYY-MM-DDTHH:MM:SS"
            try:
                ref = units.split("since", 1)[1].strip()
                # Allow "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD"
                if " " in ref and "T" not in ref:
                    ref = ref.replace(" ", "T")
                origin = np.datetime64(ref)
            except Exception:
                # Conservative fallback if units string is odd
                origin = np.datetime64("1970-01-01T00:00:00")

            # Choose numpy vs cftime path
            if calendar in ("standard", "gregorian", "proleptic_gregorian", None):
                # Numpy datetime64 (seconds resolution)
                t_dt = origin + t_clean.astype("timedelta64[s]")
            else:
                # Non-standard calendar: use cftime
                import cftime

                t_dt = xr.apply_ufunc(
                    lambda v: np.array(
                        cftime.num2date(
                            v.astype("float64"),
                            units,
                            calendar,
                            only_use_cftime_datetimes=True,
                        )
                    ),
                    t_clean,
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[object],  # object -> cftime datetime
                )

        # --- 2c) Attach decoded coord, tidy attrs ---------------------------------
        ds = ds.assign_coords({time_dim: t_dt})
        for k in ("units", "calendar"):
            ds[time_dim].attrs.pop(k, None)
        ds[time_dim].attrs["standard_name"] = "time"
        ds[time_dim].attrs["long_name"] = "time"

        # --- 2d) Drop rows where time is invalid (NaT/null) -----------------------
        # For numpy datetime64, np.isnat works; for cftime, use pandas null check
        try:
            if np.issubdtype(ds[time_dim].dtype, np.datetime64):
                valid = ~np.isnat(ds[time_dim].values)
            else:
                # cftime/object dtype
                import pandas as pd

                valid = ~pd.isnull(ds[time_dim].values)
            if valid.ndim == 1 and valid.size == ds.sizes[time_dim]:
                ds = ds.sel({time_dim: valid})
        except Exception:
            # If anything goes wrong, keep the dataset; downstream functions may handle
            pass

    # 3) Keep requested variables (xarray will retain needed coords)
    keep = [v for v in vars_keep if v in ds]
    if keep:
        ds = ds[keep]

    # 4) Your existing pipeline: trim to mode year & ensure unique/sorted time
    ds = trim_to_mode_year(ds, time_name=time_dim)
    ds = ensure_unique_sorted_time(ds, time_name=time_dim)

    return ds


def aggregate_water_year_maxima(
    sfincs_dirs: Sequence[Path],
    extra_vars: Sequence[str] = (),
    time_dim: str = "timemax",
    face_dim: str = "nmesh2d_face",
    output_fn: Optional[Path] = None,
    keep_timeseries: bool = False,
    aggregation_mode: Literal["annual_max", "all_maxima"] = "annual_max",
) -> xr.Dataset:
    """Aggregate SFINCS NetCDF outputs across N water years.

    Two modes:

    - ``"annual_max"`` (default): for each year and each cell, pick the
      single argmax along ``time_dim`` and emit `zsmax_ann` + `<var>_at_peak`
      with dims ``(year, face_dim)``. Cells that have zero valid timesteps
      become NaN (never crash).
    - ``"all_maxima"``: skip the argmax step entirely. Each year's
      ``sfincs_map.nc`` is trimmed to its mode year, then concatenated along
      ``time_dim``. Output dims are ``(time_dim, face_dim)`` with the
      original variable names (``zsmax``, ``qmax``, ...). Suitable for POT
      EVA; incompatible with Weibull / GEV.

    Parameters
    ----------
    sfincs_dirs
        One directory per water year; each must contain ``sfincs_map.nc``.
    extra_vars
        Additional variables (e.g. ``qmax``, ``tmax``) to extract.
    keep_timeseries
        Only meaningful in ``"annual_max"`` mode. If True, persist the full
        concatenated series to a sidecar NC (`<output_fn>.timeseries.nc`),
        needed when POT EVA is run on annual-max output. Ignored in
        ``"all_maxima"`` mode (the main file already is the timeseries).
    output_fn
        Optional NetCDF persistence path. If the file exists, it is opened
        and returned without recomputing.

    Returns
    -------
    xr.Dataset with `attrs["aggregation_mode"]` recording the mode and
    `attrs["n_years"]` recording the input count (used by POT EVA).
    """
    if output_fn is not None and Path(output_fn).exists():
        return xr.open_dataset(output_fn)

    vars_keep = ["zsmax", *extra_vars]
    yearly_datasets: List[xr.Dataset] = []
    full_series: List[xr.Dataset] = []

    SENTINEL = np.float32(-1.0e30)

    for sd in sfincs_dirs:
        sd = Path(sd)
        print(f"  aggregating: {sd}")
        ds = _open_sfincs_map(sd, vars_keep, time_dim)
        if "zsmax" not in ds:
            raise RuntimeError(f"`zsmax` missing in {sd / 'sfincs_map.nc'}")

        if aggregation_mode == "all_maxima":
            # Keep all sub-yearly timemax slices; no per-cell argmax.
            full_series.append(ds.load())
            ds.close()
            continue

        if keep_timeseries:
            full_series.append(ds.copy())

        zs = ds["zsmax"]
        zs_valid = zs.where(np.isfinite(zs) & (zs > -100))
        all_nan = zs_valid.isnull().all(dim=time_dim)
        # Sentinel-fill keeps argmax safe for all-NaN cells (permanently
        # dry quadtree faces); the result is masked back to NaN below.
        i_peak = zs_valid.fillna(SENTINEL).argmax(dim=time_dim, skipna=False)
        zsmax_ann = zs_valid.isel({time_dim: i_peak}).where(~all_nan)
        t_peak = ds[time_dim].isel({time_dim: i_peak})

        year_label = int(pd.to_datetime(ds[time_dim].values).year[0])

        out = xr.Dataset(
            {"zsmax_ann": zsmax_ann.drop_vars(time_dim, errors="ignore")},
            coords={"t_peak": t_peak.drop_vars(time_dim, errors="ignore")},
        )
        for v in extra_vars:
            if v in ds:
                vv = ds[v].isel({time_dim: i_peak}).where(~all_nan)
                out[f"{v}_at_peak"] = vv.drop_vars(time_dim, errors="ignore")
        out = out.expand_dims(year=[year_label])
        yearly_datasets.append(out)
        ds.close()

    n_years = len(sfincs_dirs)

    if aggregation_mode == "all_maxima":
        if not full_series:
            raise RuntimeError("No SFINCS runs were aggregated.")
        ds_agg = xr.concat(full_series, dim=time_dim).sortby(time_dim)
        ds_agg = ensure_unique_sorted_time(ds_agg, time_name=time_dim)
        ds_agg.attrs["aggregation_mode"] = "all_maxima"
        ds_agg.attrs["n_years"] = n_years
        if output_fn is not None:
            Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
            ds_agg.to_netcdf(output_fn)
        return ds_agg

    ds_agg = xr.concat(yearly_datasets, dim="year").sortby("year")
    ds_agg.attrs["aggregation_mode"] = "annual_max"
    ds_agg.attrs["n_years"] = n_years

    if output_fn is not None:
        Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
        ds_agg.to_netcdf(output_fn)

    if keep_timeseries and output_fn is not None and full_series:
        sidecar = Path(output_fn).with_suffix(".timeseries.nc")
        ds_full = xr.concat(full_series, dim=time_dim).sortby(time_dim)
        ds_full = ensure_unique_sorted_time(ds_full, time_name=time_dim)
        ds_full.to_netcdf(sidecar)
        ds_agg.attrs["_full_timeseries_fn"] = str(sidecar)

    return ds_agg


# =============================================================================
# SECTION 2: Extreme value analysis (Weibull / GEV / POT)
# =============================================================================


def eva_weibull(
    da_annual_max: xr.DataArray,
    return_periods: Sequence[float],
    year_dim: str = "year",
) -> xr.DataArray:
    """Empirical Weibull plotting-position EVA per cell.

    Sort the n annual maxima descending; assign T_k = (n+1)/k; linearly
    interpolate the requested return periods in log-T space. Cells with
    fewer than 2 finite years receive NaN.
    """
    rp_target = np.asarray(return_periods, dtype=float)

    def _per_cell(maxes: np.ndarray) -> np.ndarray:
        finite = maxes[np.isfinite(maxes)]
        n = finite.size
        if n < 2:
            return np.full(rp_target.size, np.nan, dtype=np.float64)
        sorted_desc = np.sort(finite)[::-1]
        T_emp = (n + 1.0) / np.arange(1, n + 1, dtype=float)
        log_T_emp = np.log(T_emp[::-1])
        vals_for_T = sorted_desc[::-1]
        return np.interp(
            np.log(rp_target),
            log_T_emp,
            vals_for_T,
            left=vals_for_T[0],
            right=vals_for_T[-1],
        )

    out = xr.apply_ufunc(
        _per_cell,
        da_annual_max,
        input_core_dims=[[year_dim]],
        output_core_dims=[["rp"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"rp": rp_target.size}},
    )
    out = out.assign_coords(rp=("rp", rp_target))
    out.name = "zsmax_rp"
    return out


def eva_gev(
    da_annual_max: xr.DataArray,
    return_periods: Sequence[float],
    year_dim: str = "year",
    min_years: int = 5,
) -> xr.DataArray:
    """Per-cell GEV fit -> non-exceedance quantiles at the target RPs."""
    rp_target = np.asarray(return_periods, dtype=float)
    p = 1.0 - 1.0 / rp_target

    def _per_cell(maxes: np.ndarray) -> np.ndarray:
        finite = maxes[np.isfinite(maxes)]
        if finite.size < min_years:
            return np.full(rp_target.size, np.nan, dtype=np.float64)
        try:
            shape, loc, scale = genextreme.fit(finite)
            return genextreme.ppf(p, shape, loc=loc, scale=scale)
        except Exception:
            return np.full(rp_target.size, np.nan, dtype=np.float64)

    out = xr.apply_ufunc(
        _per_cell,
        da_annual_max,
        input_core_dims=[[year_dim]],
        output_core_dims=[["rp"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"rp": rp_target.size}},
    )
    out = out.assign_coords(rp=("rp", rp_target))
    out.name = "zsmax_rp"
    return out


def eva_pot(
    da_full_timeseries: xr.DataArray,
    return_periods: Sequence[float],
    n_years: int,
    target_per_year: int = 5,
    decluster_window: str = "72h",
    time_dim: str = "timemax",
    face_dim: str = "nmesh2d_face",
) -> xr.DataArray:
    """Per-cell POT extraction + log-T interpolation to target RPs.

    Selects a threshold yielding ~`target_per_year * n_years` declustered
    peaks, then assigns empirical return periods T_k = n_years*(k_peaks+1)/k
    and interpolates in log-T space to `return_periods`.
    """
    rp_target = np.asarray(return_periods, dtype=float)
    target_total = max(1, int(target_per_year * n_years))

    faces = da_full_timeseries[face_dim].values
    out_arr = np.full((faces.size, rp_target.size), np.nan, dtype=np.float64)

    for i, face in enumerate(faces):
        series = da_full_timeseries.sel({face_dim: face})
        vals = series.values
        finite = vals[np.isfinite(vals) & (vals > -100)]
        if finite.size < 5:
            continue
        try:
            th = pot_threshold_set_num_xr(
                series,
                r=decluster_window,
                num_exce=target_total,
                time_dim=time_dim,
                strategy="closest",
            )
            extremes = get_extremes_pot_xr(
                series,
                th,
                r=decluster_window,
                time_dim=time_dim,
                num_exce=target_total,
            )
        except Exception:
            continue
        v = np.sort(extremes.values)[::-1]
        k = v.size
        if k < 2:
            continue
        T_emp = n_years * (k + 1.0) / np.arange(1, k + 1, dtype=float)
        log_T_emp = np.log(T_emp[::-1])
        vals_for_T = v[::-1]
        out_arr[i] = np.interp(
            np.log(rp_target),
            log_T_emp,
            vals_for_T,
            left=vals_for_T[0],
            right=vals_for_T[-1],
        )

    return xr.DataArray(
        out_arr,
        dims=(face_dim, "rp"),
        coords={face_dim: faces, "rp": rp_target},
        name="zsmax_rp",
    )


def eva_apply(
    ds_annual: xr.Dataset,
    method: Literal["weibull", "gev", "pot"],
    return_periods: Sequence[float],
    ds_full_timeseries: Optional[xr.Dataset] = None,
    face_dim: str = "nmesh2d_face",
    time_dim: str = "timemax",
    pot_target_per_year: int = 5,
    pot_decluster: str = "72h",
    gev_min_years: int = 5,
) -> xr.Dataset:
    """Dispatch EVA over annual maxima (or full series for POT).

    The aggregation mode of `ds_annual` (stamped at build time) controls
    which methods are valid:

      - "annual_max": all three EVA methods supported. Weibull / GEV use
        `ds_annual["zsmax_ann"]`; POT requires `ds_full_timeseries`.
      - "all_maxima": only POT is supported. The dataset itself acts as
        the timeseries; `ds_full_timeseries` is ignored if passed.
    """
    method = method.lower()
    agg_mode = ds_annual.attrs.get("aggregation_mode", "annual_max")

    if agg_mode == "all_maxima":
        if method != "pot":
            raise ValueError(
                f"aggregation_mode='all_maxima' is incompatible with "
                f"eva_method={method!r}: no annual maxima are available. "
                "Use 'annual_max' for Weibull/GEV, or switch eva_method to 'pot'."
            )
        if "zsmax" not in ds_annual:
            raise ValueError("`zsmax` missing in all-maxima dataset.")
        n_years = int(ds_annual.attrs.get("n_years", 0))
        if n_years < 2:
            raise ValueError(
                f"POT EVA requires at least 2 water years; got n_years={n_years}."
            )
        zsmax_rp = eva_pot(
            ds_annual["zsmax"],
            return_periods,
            n_years=n_years,
            target_per_year=pot_target_per_year,
            decluster_window=pot_decluster,
            time_dim=time_dim,
            face_dim=face_dim,
        )
    elif method == "weibull":
        zsmax_rp = eva_weibull(ds_annual["zsmax_ann"], return_periods)
    elif method == "gev":
        zsmax_rp = eva_gev(
            ds_annual["zsmax_ann"], return_periods, min_years=gev_min_years
        )
    elif method == "pot":
        if ds_full_timeseries is None or "zsmax" not in ds_full_timeseries:
            raise ValueError("POT EVA requires `ds_full_timeseries` with `zsmax`.")
        n_years = int(ds_annual.attrs.get("n_years", ds_annual.sizes.get("year", 0)))
        if n_years < 2:
            raise ValueError("POT EVA requires at least 2 water years of data.")
        zsmax_rp = eva_pot(
            ds_full_timeseries["zsmax"],
            return_periods,
            n_years=n_years,
            target_per_year=pot_target_per_year,
            decluster_window=pot_decluster,
            time_dim=time_dim,
            face_dim=face_dim,
        )
    else:
        raise ValueError(f"Unknown EVA method {method!r}; use weibull/gev/pot.")

    out = xr.Dataset({"zsmax_rp": zsmax_rp})
    out.attrs["eva_method"] = method
    out.attrs["aggregation_mode"] = agg_mode
    return out


# =============================================================================
# SECTION 3: Event-matched extras + quadtree-to-DEM mapping
# =============================================================================


def extras_at_rp_via_rank(
    ds_annual: xr.Dataset,
    return_periods: Sequence[float],
    extras: Sequence[str],
    year_dim: str = "year",
) -> xr.Dataset:
    """For Weibull/GEV: pick each extra at the year contributing the
    rank-matched annual max. Empirical Weibull rank k = (n+1)/RP, rounded
    to the nearest integer and clamped to [1, n]."""
    if not extras:
        return xr.Dataset()

    rp_target = np.asarray(return_periods, dtype=float)
    n_rp = rp_target.size

    zs = ds_annual["zsmax_ann"]
    zs_t = zs.transpose(..., year_dim)
    n_years = zs_t.sizes[year_dim]
    k_real = np.clip((n_years + 1.0) / rp_target, 1.0, n_years)
    k_int = np.round(k_real).astype(int)

    out = xr.Dataset()
    for v in extras:
        key = f"{v}_at_peak"
        if key not in ds_annual:
            continue
        da_extra = ds_annual[key].transpose(..., year_dim)
        zs_vals = zs_t.values
        ex_vals = da_extra.values
        order = np.argsort(-zs_vals, axis=-1, kind="stable")
        ex_sorted = np.take_along_axis(ex_vals, order, axis=-1)
        rank_cols = (k_int - 1).reshape((1,) * (ex_sorted.ndim - 1) + (n_rp,))
        rank_cols_b = np.broadcast_to(rank_cols, ex_sorted.shape[:-1] + (n_rp,))
        picked = np.take_along_axis(ex_sorted, rank_cols_b, axis=-1)
        non_year_dims = [d for d in da_extra.dims if d != year_dim]
        coords = {d: da_extra[d] for d in non_year_dims}
        coords["rp"] = rp_target
        out[f"{v}_rp"] = xr.DataArray(
            picked,
            dims=tuple(non_year_dims) + ("rp",),
            coords=coords,
            name=f"{v}_rp",
        )
    return out


def map_quadtree_to_dem_nearest(
    da_face: xr.DataArray,
    indices_fn: Path,
    out_fn: Path,
    depth_mask_fn: Optional[Path] = None,
    hmin: float = 0.02,
) -> None:
    """Nearest-neighbour map a per-face DataArray onto the DEM grid via an
    index COG. Optionally mask out pixels where depth <= hmin."""
    indices_fn = Path(indices_fn)
    out_fn = Path(out_fn)
    out_fn.parent.mkdir(parents=True, exist_ok=True)

    values = da_face.values.astype(np.float32)

    with rasterio.open(indices_fn) as idx_src:
        idx_nodata = int(idx_src.nodata)
        meta = idx_src.meta.copy()

    meta.update(
        count=1,
        dtype="float32",
        nodata=float("nan"),
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
        predictor=2,
        BIGTIFF="YES",
    )

    if depth_mask_fn is not None:
        depth_mask_fn = Path(depth_mask_fn)
    with (
        rasterio.open(out_fn, "w", **meta) as dst,
        rasterio.open(indices_fn) as idx_src,
    ):
        dep_src = rasterio.open(depth_mask_fn) if depth_mask_fn is not None else None
        try:
            for _, window in dst.block_windows(1):
                idx_block = idx_src.read(1, window=window)
                if dep_src is not None:
                    dep_block = dep_src.read(1, window=window)
                    wet = np.isfinite(dep_block) & (dep_block > hmin)
                else:
                    wet = np.ones_like(idx_block, dtype=bool)
                valid = (idx_block != idx_nodata) & wet
                out_block = np.full(idx_block.shape, np.nan, dtype=np.float32)
                if np.any(valid):
                    out_block[valid] = values[idx_block[valid].astype(np.intp)]
                dst.write(out_block, 1, window=window)
        finally:
            if dep_src is not None:
                dep_src.close()

def smooth_raster_gaussian_blockwise(
    in_fn: Path,
    out_fn: Path,
    smooth_size: float,
    truncate: float = None,
) -> None:
    """
    Blockwise NaN‑preserving Gaussian smoothing with halo padding.

    This function applies Gaussian smoothing to a raster using a blockwise
    (tiled) streaming approach. Unlike whole‑array convolution, which requires
    loading the entire raster into memory, this method expands each block read
    by adding a surrounding padded “halo” region. The halo ensures that the
    Gaussian filter has the necessary neighborhood context to avoid seam
    artifacts at tile boundaries.

    The smoothing is NaN‑aware: NaNs in the input raster are excluded from
    influencing neighboring pixels, and NaNs are preserved in the result.

    Parameters
    ----------
    in_fn : Path
        Path to the input raster file. Must be readable by rasterio and contain
        a single-band floating‑point dataset (e.g., float32).  
        If `in_fn` and `out_fn` refer to the same resolved path, the function
        performs **safe in‑place smoothing** by writing results to a temporary
        file and then atomically replacing the original.

    out_fn : Path
        Output raster file path.  
        If different from `in_fn`, the smoothed raster is written directly here.  
        If equal to `in_fn`, a temporary raster is created and then moved over
        the original file to guarantee correctness and avoid partial overwrites.

    smooth_size : float
        Standard deviation (sigma) of the Gaussian kernel passed to
        `scipy.ndimage.gaussian_filter`.  
        Larger values produce stronger, more spatially extensive smoothing.

    truncate : float, optional
        Gaussian kernel truncation radius, expressed in multiples of `sigma`.  
        The kernel is effectively limited to:
            radius = truncate * smooth_size  
        Default is `2 * smooth_size` (a common cutoff balancing accuracy and
        performance).  
        This value determines the **halo size** required around each block.

    Notes
    -----
    • Halo size is computed as:  
         halo = int(truncate * smooth_size)

    • The function uses rasterio's native block windows for streaming I/O.  
      Each block is read with extra pixels on all sides (the halo), smoothed,
      and cropped back before writing.

    • Output is written as float32 with NaN nodata.

    • Raster tags are updated to document smoothing parameters.
    """

    in_fn = Path(in_fn)
    out_fn = Path(out_fn)

    # Determine if user wants in-place smoothing
    in_place = in_fn.resolve() == out_fn.resolve()

    # If in-place: create a temporary output file
    if in_place:
        temp_dir = tempfile.TemporaryDirectory()
        tmp_out = Path(temp_dir.name) / "smoothed.tif"
        actual_out = tmp_out
    else:
        out_fn.parent.mkdir(parents=True, exist_ok=True)
        actual_out = out_fn

    if truncate is None:
        truncate = 2 * smooth_size

    halo = int(truncate * smooth_size)

    with rasterio.open(in_fn) as src:
        meta = src.meta.copy()
        height, width = src.height, src.width

        meta.update(
            dtype="float32",
            nodata=np.nan,
            count=1,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
            BIGTIFF="YES",
        )

        with rasterio.open(actual_out, "w", **meta) as dst:

            for _, win in src.block_windows(1):

                # Build padded window
                row_off = max(win.row_off - halo, 0)
                col_off = max(win.col_off - halo, 0)
                row_end = min(win.row_off + win.height + halo, height)
                col_end = min(win.col_off + win.width + halo, width)

                pad_win = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=col_end - col_off,
                    height=row_end - row_off,
                )

                pad_block = src.read(1, window=pad_win).astype("float32")

                # NaN-handling logic
                ind_nan = np.isnan(pad_block)
                V = pad_block.copy()
                V[ind_nan] = 0.0

                W = np.ones_like(V, dtype="float32")
                W[ind_nan] = 0.0

                VV = gaussian_filter(V, sigma=smooth_size, truncate=truncate)
                WW = gaussian_filter(W, sigma=smooth_size, truncate=truncate)

                out_pad = np.full_like(VV, np.nan, dtype="float32")
                mask = WW > 1e-10
                out_pad[mask] = VV[mask] / WW[mask]
                out_pad[ind_nan] = np.nan

                # Crop to block
                row0 = win.row_off - row_off
                row1 = row0 + win.height
                col0 = win.col_off - col_off
                col1 = col0 + win.width

                out_block = out_pad[row0:row1, col0:col1]

                dst.write(out_block, 1, window=win)

            # Provenance
            dst.update_tags(
                smoothing="gaussian_blockwise",
                smoothing_sigma=str(smooth_size),
                smoothing_truncate=str(truncate),
                smoothing_halo=str(halo),
                smoothing_note="Blockwise NaN-preserving Gaussian smoothing with halo"
            )

    # Finalize in-place operation
    if in_place:
        temp_dir.cleanup()  # not yet—wait!
        # Replace original raster
        shutil.move(str(actual_out), str(in_fn))
        # Workaround: we created a TemporaryDirectory, delete structure but not smoothed file
        # But since we moved the file out, the tempdir is empty:
        shutil.rmtree(temp_dir.name, ignore_errors=True)


# =============================================================================
# SECTION 4: Hazard binning + I/O
# =============================================================================


def stamp_provenance(fn: Path, **tags) -> None:
    """Merge `tags` into the GeoTIFF metadata of `fn` (open in r+ mode).

    Values are coerced to `str`. Existing tags (e.g. those `bin_raster`
    writes) are preserved; only the keys passed here are added/overwritten.
    """
    fn = Path(fn)
    if not fn.exists():
        return
    str_tags = {k: str(v) for k, v in tags.items() if v is not None}
    if not str_tags:
        return
    with rasterio.open(fn, "r+") as dst:
        dst.update_tags(**str_tags)


def bin_raster(
    in_fn: Path,
    bin_edges: Sequence[float],
    labels: Sequence[str],
    out_fn: Path,
) -> None:
    """Block-wise hazard-category binning of a float raster -> uint8 GeoTIFF.

    Bin convention (np.digitize, right=False):
      0      below first edge (treated as no-hazard / dry)
      1..N   bin index for values in [edges[i-1], edges[i]) ; last bin -> +inf
      255    nodata (non-finite source pixel)

    `labels` (length N) describe bins 1..N and are written as TIFF tags.
    """
    in_fn = Path(in_fn)
    out_fn = Path(out_fn)
    out_fn.parent.mkdir(parents=True, exist_ok=True)

    edges = np.asarray(bin_edges, dtype=np.float32)
    if edges.size != len(labels):
        raise ValueError(
            f"bin_edges has {edges.size} entries but labels has {len(labels)}; "
            "expected one label per edge (lower bound of each bin)."
        )

    with rasterio.open(in_fn) as src:
        meta = src.meta.copy()
        meta.update(
            dtype="uint8",
            nodata=255,
            count=1,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
            BIGTIFF="YES",
        )
        with rasterio.open(out_fn, "w", **meta) as dst:
            for _, window in src.block_windows(1):
                block = src.read(1, window=window)
                bins = np.digitize(block, edges, right=False).astype(np.uint8)
                bins[~np.isfinite(block)] = 255
                dst.write(bins, 1, window=window)
            dst.update_tags(
                bin_edges=",".join(f"{e:g}" for e in edges),
                bin_labels=",".join(labels),
                nodata_label="nodata",
                bin_0_label="below_first_edge",
            )


def bin_depth_with_overlays(
    hmax_masked_fn: Path,
    connection_fn: Path,
    dem_fn: Path,
    bin_edges: Sequence[float],
    bin_labels: Sequence[str],
    out_fn: Path,
    mhhw_elevation: Optional[float] = None,
    below_mhhw_label: str = "Below MHHW",
    floodprone_label: str = "Flood-prone Low-Lying",
) -> None:
    """Categorical depth raster with MHHW + flood-prone overlays.

    Block-by-block compositing of three rasters into one uint8 GeoTIFF.

    Code mapping (all on the DEM grid):

    ===========  =====================================================
    Code         Source / meaning
    ===========  =====================================================
    0            dry / no flooding
    1            Below MHHW (only set when mhhw_elevation is not None
                 AND dem < mhhw_elevation; takes precedence over all
                 other categories so tidally-submerged channels are
                 not labelled as flood hazard)
    2..N+1       depth bins (N = len(bin_edges); `bin_labels[i]`
                 corresponds to code i+2, ascending in depth)
    N+2          Flood-prone Low-Lying (`connection == 2`, i.e.
                 standing water removed by Step 5 because it was
                 disconnected from the boundary)
    255          nodata (non-finite DEM)
    ===========  =====================================================

    Tags written for downstream legend rendering: `bin_edges`,
    `bin_labels`, `below_mhhw_code`, `floodprone_code`, plus a label
    for each code so QGIS / gdalinfo show the mapping.
    """
    hmax_masked_fn = Path(hmax_masked_fn)
    connection_fn = Path(connection_fn)
    dem_fn = Path(dem_fn)
    out_fn = Path(out_fn)
    out_fn.parent.mkdir(parents=True, exist_ok=True)

    edges = np.asarray(bin_edges, dtype=np.float32)
    if edges.size != len(bin_labels):
        raise ValueError(
            f"bin_edges has {edges.size} entries but bin_labels has "
            f"{len(bin_labels)}; expected one label per bin (lower edge)."
        )
    n_depth_bins = edges.size
    below_mhhw_code = 1
    depth_code_offset = 2  # depth bins -> 2 .. n_depth_bins+1
    floodprone_code = (
        n_depth_bins + depth_code_offset
    )  # next code after the deepest bin

    if floodprone_code >= 255:
        raise ValueError(
            f"Too many depth bins ({n_depth_bins}); floodprone_code "
            f"would be {floodprone_code} >= 255 (nodata)."
        )

    with rasterio.open(hmax_masked_fn) as src:
        meta = src.meta.copy()
    meta.update(
        count=1,
        dtype="uint8",
        nodata=255,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
        BIGTIFF="YES",
    )

    with (
        rasterio.open(out_fn, "w", **meta) as dst,
        rasterio.open(hmax_masked_fn) as hsrc,
        rasterio.open(connection_fn) as csrc,
        rasterio.open(dem_fn) as dsrc,
    ):
        for _, window in dst.block_windows(1):
            h = hsrc.read(1, window=window).astype(np.float32)
            c = csrc.read(1, window=window)
            d = dsrc.read(1, window=window).astype(np.float32)

            out = np.zeros(h.shape, dtype=np.uint8)  # 0 = dry / default

            # nodata: non-finite DEM (outside model domain)
            dem_nodata = ~np.isfinite(d)
            out[dem_nodata] = 255

            # below MHHW (only where the DEM is finite); wins over all else
            if mhhw_elevation is not None:
                below = (~dem_nodata) & (d < float(mhhw_elevation))
                out[below] = below_mhhw_code
            else:
                below = np.zeros(h.shape, dtype=bool)

            # depth bins for pixels NOT below MHHW and with finite depth > 0
            wet = (~below) & (~dem_nodata) & np.isfinite(h) & (h > 0.0)
            if np.any(wet):
                bins = np.digitize(h, edges, right=False).astype(np.int16)
                bins[~wet] = 0
                # np.digitize returns 0 for values below the smallest edge;
                # shift wet bins (1..N) into the [2..N+1] depth-code range
                shifted = np.where(bins >= 1, bins + (depth_code_offset - 1), 0).astype(
                    np.uint8
                )
                out[wet] = shifted[wet]

            # flood-prone low-lying: connection == 2, but only where NOT
            # below MHHW and NOT already a real depth bin
            disconnected = (
                (~below) & (~dem_nodata) & (c == 2) & ((out == 0) | (out == 255))
            )
            out[disconnected] = floodprone_code

            dst.write(out, 1, window=window)

        # Tags for downstream legend rendering
        tags = {
            "bin_edges": ",".join(f"{e:g}" for e in edges),
            "bin_labels": ",".join(bin_labels),
            "code_0_label": "dry",
            "below_mhhw_code": str(below_mhhw_code),
            f"code_{below_mhhw_code}_label": below_mhhw_label,
            "floodprone_code": str(floodprone_code),
            f"code_{floodprone_code}_label": floodprone_label,
            "nodata_label": "nodata",
        }
        for i, lbl in enumerate(bin_labels):
            tags[f"code_{i + depth_code_offset}_label"] = lbl
        if mhhw_elevation is not None:
            tags["mhhw_elevation_m"] = f"{mhhw_elevation:g}"
        dst.update_tags(**tags)


# =============================================================================
# SECTION 5: Shapefile
# =============================================================================


def raster_to_shape_rasterio(raster_file, vector_file):

    with rasterio.open(raster_file) as src:
        image = src.read(1)
        transform = src.transform
        results = shapes(image, transform=transform)

        geoms = []
        for geom, value in results:
            geoms.append({"geometry": shape(geom), "properties": {"ID": int(value)}})
        gdf = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)

    gdf.to_file(vector_file)


def raster_to_polygons(
    raster_file: str,
    vector_file: str,
    connectivity: int = 8,
    min_pixels: int | None = None,
    dissolve: bool = False,
    driver: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Polygonize a (typically categorical) raster to a vector dataset.

    - connectivity: 4 or 8 (queen/rook)
    - min_pixels: if set, removes patches smaller than this (salt-and-pepper cleanup)
    - dissolve: merge polygons by 'ID' after polygonization
    - driver: 'GPKG', 'GeoJSON', 'ESRI Shapefile', etc. If None, inferred from file extension.
    """
    with rasterio.open(raster_file) as src:
        band = src.read(1)  # first band
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        # Build mask to exclude NoData
        if np.issubdtype(band.dtype, np.floating):
            mask = ~np.isnan(band) if nodata is None else band != nodata
        else:
            mask = (band != nodata) if nodata is not None else None

        # Optional cleanup to remove tiny patches
        if min_pixels:
            band = sieve(band, size=min_pixels, connectivity=connectivity)

        # Polygonize
        results = shapes(
            band, mask=mask, transform=transform, connectivity=connectivity
        )

        # Build features: keep integer class values as 'ID'
        feats = []
        for geom, value in results:
            if value is None:
                continue
            # Try to cast to int for categorical classes; fallback to original
            try:
                value_out = int(value)
            except Exception:
                value_out = value
            feats.append({"geometry": shape(geom), "properties": {"ID": value_out}})

    gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)

    # Fix invalid geometries if any (self-intersections, etc.)
    if not gdf.empty:
        gdf = gdf[gdf.geometry.notna()]
        gdf = gdf.set_geometry(gdf.geometry.buffer(0))

        # Split MultiPolygons into single-part polygons
        gdf = gdf.explode(index_parts=False, ignore_index=True)

    # Optionally merge polygons by shared class
    if dissolve and not gdf.empty:
        gdf = gdf.dissolve(by="ID", as_index=False)

    # Choose driver from extension if not provided
    if driver is None:
        ext = os.path.splitext(vector_file)[1].lower()
        driver = {
            ".gpkg": "GPKG",
            ".geojson": "GeoJSON",
            ".json": "GeoJSON",
            ".shp": "ESRI Shapefile",
        }.get(ext, None)

    # Write out
    if driver:
        gdf.to_file(vector_file, driver=driver)
    else:
        gdf.to_file(vector_file)

    return gdf


def export_connectivity_regions(
    raster_file: str,
    connected_shp: str,
    disconnected_shp: str,
    connectivity: int = 8,
    min_pixels: int | None = None,
    fix_invalid: bool = True,
    simplify_tolerance: float | None = None,
    driver: str = "ESRI Shapefile",
):
    """
    Polygonize a binary raster (1 = connected, 2 = disconnected) and write two shapefiles
    containing multiple polygons (NO dissolve). Each feature represents one contiguous region.

    Parameters
    ----------
    raster_file : str
        Path to input raster with values 1 (connected) and 2 (disconnected). May include NoData.
    connected_shp : str
        Output shapefile path for connected regions (ID=1).
    disconnected_shp : str
        Output shapefile path for disconnected regions (ID=2).
    connectivity : int, default=8
        Pixel connectivity used during polygonization (4 = rook, 8 = queen).
    min_pixels : int | None, default=None
        If set, removes patches smaller than this number of pixels before polygonization.
    fix_invalid : bool, default=True
        If True, repairs invalid geometries with a zero-width buffer.
    simplify_tolerance : float | None, default=None
        If set (in CRS units), simplifies geometries (preserve_topology=True) to reduce vertices.
        Use cautiously in geographic CRS; best in projected CRS.
    driver : str, default="ESRI Shapefile"
        GDAL driver for outputs. Alternatives: "GPKG", "GeoJSON", etc.

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        (connected_gdf, disconnected_gdf), each with multiple polygons and attributes:
        - ID: 1 or 2 (class)
        - region_id: sequential identifier per shapefile
        - area_m2: optional, if CRS is projected
    """
    # --- Read raster ---
    with rasterio.open(raster_file) as src:
        band = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        # Build mask to exclude NoData from polygonization
        if nodata is None:
            if np.issubdtype(band.dtype, np.floating):
                mask = ~np.isnan(band)
            else:
                mask = None  # polygonize all values
        else:
            mask = band != nodata

        # Optional cleanup: remove small patches prior to polygonization
        if min_pixels:
            band = sieve(band, size=min_pixels, connectivity=connectivity)

        # Polygonize once; split features into two lists
        feats_conn, feats_disc = [], []
        for geom, val in shapes(
            band, mask=mask, transform=transform, connectivity=connectivity
        ):
            if val == 1:
                feats_conn.append({"geometry": shape(geom), "properties": {"ID": 1}})
            elif val == 2:
                feats_disc.append({"geometry": shape(geom), "properties": {"ID": 2}})
            # ignore any other values

    # Build GeoDataFrames
    gdf_conn = gpd.GeoDataFrame.from_features(feats_conn, crs=crs)
    gdf_disc = gpd.GeoDataFrame.from_features(feats_disc, crs=crs)

    # --- Prep helper ---
    def _prep(gdf: gpd.GeoDataFrame, id_value: int) -> gpd.GeoDataFrame:
        if gdf.empty:
            return gdf
        gdf = gdf[gdf.geometry.notna()]
        if fix_invalid:
            gdf = gdf.set_geometry(gdf.geometry.buffer(0))  # repair self-intersections
        # Split MultiPolygons into single-part features (each part is treated as a region)
        gdf = gdf.explode(index_parts=False, ignore_index=True)
        # Optional simplify to reduce vertex count
        if (simplify_tolerance is not None) and (simplify_tolerance > 0):
            gdf = gdf.set_geometry(
                gdf.geometry.simplify(simplify_tolerance, preserve_topology=True)
            )
        # Assign sequential region IDs per shapefile
        gdf["region_id"] = np.arange(1, len(gdf) + 1, dtype=int)
        gdf["ID"] = id_value

        # Optional area in square meters if CRS is projected
        try:
            if gdf.crs and getattr(gdf.crs, "is_projected", False):
                gdf["area_m2"] = gdf.geometry.area
        except Exception:
            pass

        return gdf

    gdf_conn = _prep(gdf_conn, 1)
    gdf_disc = _prep(gdf_disc, 2)

    # --- Write outputs ---
    for out_path, gdf in [(connected_shp, gdf_conn), (disconnected_shp, gdf_disc)]:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        if not gdf.empty:
            gdf.to_file(out_path, driver=driver)
        else:
            print(
                f"Info: No polygons written for {out_path} (no regions of class {gdf['ID'].iloc[0] if not gdf.empty else 'N/A'})."
            )

    return gdf_conn, gdf_disc

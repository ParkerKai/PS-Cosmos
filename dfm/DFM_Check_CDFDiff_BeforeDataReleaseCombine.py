from pathlib import Path
import os
import math
import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional, List, Dict, Any, Tuple
import cftime

# ---------- Constants ----------
TOL = 1e-8  # unified numerical tolerance for step comparisons & monotonic checks
DT_TOL = np.timedelta64(0, "ns")  # tolerant non-decreasing (allow equal for duplicates)

# =============================== #
# Helpers & Normalization
# =============================== #


def _parse_expected_step_days(expected_freq: str) -> float:
    """
    Convert a fixed-duration string (parseable by pandas.to_timedelta) to days (float).
    Examples: 'D', 'H', '15min', '10min', '2D'.
    NOTE: Month-based aliases like 'MS' are not fixed durations and are unsupported here.
    """
    try:
        td = pd.to_timedelta(expected_freq)
    except Exception as e:
        raise ValueError(
            f"expected_freq '{expected_freq}' must be a fixed duration parseable by pandas.to_timedelta "
            f"(e.g., 'D','H','15min','10min'). Month-based aliases like 'MS' are unsupported."
        ) from e
    return td / pd.Timedelta(days=1)


def _mode_value(a: np.ndarray) -> Optional:
    """Return the mode of array a (float) or None if empty."""
    if a.size == 0:
        return None
    vals, counts = np.unique(a, return_counts=True)
    return float(vals[np.argmax(counts)])


def normalize_labels(values: np.ndarray, *, strip: bool = True, casefold: bool = False) -> np.ndarray:
    """
    Normalize station/cmip6 labels to comparable strings.
    - Converts to pandas Index -> astype(str)
    - Optional strip whitespace
    - Optional casefold (Unicode-lower for robust matching)
    """
    if values is None:
        return np.array([], dtype=object)
    idx = pd.Index(values).astype(str)
    if strip:
        idx = idx.map(lambda s: s.strip())
    if casefold:
        idx = idx.map(lambda s: s.casefold())
    return np.asarray(idx)


def _to_datetime64(times: np.ndarray) -> np.ndarray:
    """
    Return an array of dtype datetime64[ns].
    Raises if cftime objects are found (non-standard calendars).
    """
    if times is None or len(times) == 0:
        return np.array([], dtype="datetime64[ns]")

    t0 = times[0]
    if "cftime" in type(t0).__module__:
        raise TypeError(
            "cftime detected in time coordinate. Please ensure decode_cf=True "
            "with a standard calendar or convert to a standard calendar before checks."
        )

    # Normalize via DatetimeIndex to simplify tz handling
    ts = pd.DatetimeIndex(pd.to_datetime(times))
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_numpy(dtype="datetime64[ns]")


def _infer_prevailing_step(diffs: np.ndarray) -> Optional[np.timedelta64]:
    """
    Given an array of timedelta64 diffs, return the most common positive delta (prevailing step).
    Returns None if there are no positive diffs.

    Robustness: to reduce micro-variance, round to nearest second before mode.
    """
    if diffs.size == 0:
        return None
    pos = diffs[diffs > np.timedelta64(0, "ns")]
    if pos.size == 0:
        return None

    # Round to second to avoid nanosecond-scale jitter
    pos_td = pd.to_timedelta(pos).round("s").to_numpy()
    unique_vals, counts = np.unique(pos_td, return_counts=True)
    return unique_vals[np.argmax(counts)]


def _fmt_td(td: Optional[np.timedelta64]) -> str:
    if td is None:
        return "None"
    return str(pd.Timedelta(td))


def _expected_step_days(expected_freq: str) -> float:
    """
    Only used in CF-numeric scan that operates in units of 'days since ...'.
    Supported fixed durations: 'D' (1 day) and 'H' (1/24 day).
    """
    ef = expected_freq.upper()
    if ef == "D":
        return 1.0
    elif ef == "H":
        return 1.0 / 24.0
    else:
        raise ValueError("expected_freq must be 'D' or 'H' for CF-numeric scans.")


def _to_num_days(
    times: np.ndarray,
    calendar: str = "standard",
    units: Optional[str] = None,
) -> np.ndarray:
    """
    Convert an array of datetime-like OR numeric CF times to float64 days since 1970-01-01.
    """
    if times is None or len(times) == 0:
        return np.array([], dtype=np.float64)

    epoch_units = "days since 1970-01-01 00:00:00"
    a = np.asarray(times)

    # Numeric CF coordinate (e.g., decode_cf=False)
    if np.issubdtype(a.dtype, np.number):
        if not units:
            raise ValueError(
                "Numeric time coordinate provided without CF 'units'. "
                "Either set decode_cf=True or pass units=the time coord units."
            )
        dt_list = cftime.num2date(list(a), units=units, calendar=calendar)
        nums = cftime.date2num(dt_list, units=epoch_units, calendar=calendar)
        return np.asarray(nums, dtype=np.float64)

    # cftime path
    t0 = times[0]
    is_cftime = "cftime" in type(t0).__module__
    if is_cftime:
        nums = cftime.date2num(list(times), units=epoch_units, calendar=calendar)
        return np.asarray(nums, dtype=np.float64)

    # pandas/numpy datetime-like path
    ts = pd.DatetimeIndex(pd.to_datetime(times))
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)

    delta_ns = (
        (ts - pd.Timestamp("1970-01-01"))
        .to_numpy(dtype="timedelta64[ns]")
        .astype(np.int64)
    )
    return delta_ns.astype(np.float64) / (86400.0 * 1e9)


def _to_printable(dt):
    if dt is None:
        return pd.NaT
    return dt if "cftime" in type(dt).__module__ else pd.Timestamp(dt)


# =============================== #
# Time index utilities (fill-ins)
# =============================== #

def ensure_unique_sorted_time(ds: xr.Dataset, time_dim: str = "time") -> xr.Dataset:
    """
    Sort dataset by time and drop duplicate timestamps (keep first occurrence).
    Returns a new dataset.
    """
    if time_dim not in ds.coords and time_dim not in ds.dims:
        raise KeyError(f"ensure_unique_sorted_time: missing '{time_dim}' in dataset")

    ds = ds.sortby(time_dim)
    t = ds[time_dim].values
    if t.size == 0:
        return ds

    # Use numpy unique on datetime64
    _, unique_idx = np.unique(t, return_index=True)
    unique_idx = np.sort(unique_idx)
    if unique_idx.size != t.size:
        print(f"[INFO] Dropping {t.size - unique_idx.size} duplicate time steps")
        ds = ds.isel({time_dim: unique_idx})
    return ds


def assert_regular_time(
    ds: xr.Dataset,
    time_dim: str = "time",
    expected_freq: Optional[str] = None,
    label: str = "dataset",
    allow_single_step: bool = True,
) -> None:
    """
    Assert that the time axis is monotonically non-decreasing and regularly sampled.
    If expected_freq is provided, it must be a duration parseable by pandas.to_timedelta
    (e.g., 'D', 'H', '15min', '2D'). Monthly aliases like 'MS' are NOT supported here.
    Otherwise, enforce that all positive steps equal the prevailing step.
    """
    if time_dim not in ds:
        raise KeyError(f"{label}: missing time coordinate '{time_dim}'")

    t = _to_datetime64(ds[time_dim].values)
    if t.size <= 1:
        if not allow_single_step:
            raise AssertionError(f"{label}: insufficient time steps for regularity check")
        return

    diffs = np.diff(t)
    if not np.all(diffs >= DT_TOL):
        neg = int(np.count_nonzero(diffs < np.timedelta64(0, "ns")))
        raise AssertionError(f"{label}: time not monotonic — {neg} backward steps")

    pos = diffs[diffs > np.timedelta64(0, "ns")]
    if pos.size == 0:
        # all equal timestamps (unlikely after ensure_unique_sorted_time)
        raise AssertionError(f"{label}: time diffs are all zero")

    prevailing = _infer_prevailing_step(diffs)

    if expected_freq is not None:
        # Flexible durations, not offsets like 'MS'
        try:
            expected_td = pd.to_timedelta(expected_freq).to_numpy()
        except Exception:
            raise ValueError(
                f"{label}: expected_freq must be a duration string parseable by pandas.to_timedelta "
                f"(e.g., 'D','H','15min'); monthly aliases like 'MS' are not supported here."
            )

        if not np.all(pos == expected_td):
            arr = np.sort(pos)
            mid = arr[arr.size // 2]
            raise AssertionError(
                f"{label}: irregular sampling — expected {pd.Timedelta(expected_td)}, "
                f"found min/med/max: {_fmt_td(pos.min())}/{_fmt_td(mid)}/{_fmt_td(pos.max())}"
            )
    else:
        if not np.all(pos == prevailing):
            arr = np.sort(pos)
            mid = arr[arr.size // 2]
            raise AssertionError(
                f"{label}: irregular sampling — prevailing={_fmt_td(prevailing)}; "
                f"min/med/max: {_fmt_td(pos.min())}/{_fmt_td(mid)}/{_fmt_td(pos.max())}"
            )


# =============================== #
# NEW: Coordinate (station/cmip6) consistency scanner
# =============================== #

def scan_coordinate_consistency(
    data_dir: str,
    file_glob: str = "*.nc",
    station_dim_name: str = "station",
    cmip6_dim_name: str = "cmip6",
    decode_cf: bool = True,
    engine: Optional[str] = None,
    write_csv_dir: Optional[str] = None,
    normalize_casefold: bool = False,
) -> Dict[str, Any]:
    """
    Scan station/cmip6 coordinates across files:
      - presence of dims/coords
      - duplicates within file
      - dtype/encoding differences
      - missing/extra labels vs reference (first file)
      - reorder vs reference
      - canonical intersection/union (normalized)

    Returns a summary dict and optionally writes CSVs.
    """
    files = sorted(Path(data_dir).glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No files match {file_glob} in {data_dir}")

    per_file = []
    station_sets = []
    cmip6_sets = []
    station_ref_raw = None
    cmip6_ref_raw = None

    for f in files:
        open_kwargs = dict(decode_cf=decode_cf)
        if engine:
            open_kwargs["engine"] = engine

        try:
            with xr.open_dataset(f, **open_kwargs) as ds:
                rec = {"file": f.name}

                # Station
                if station_dim_name in ds.dims or station_dim_name in ds.coords:
                    st = ds[station_dim_name].values
                    st_raw = pd.Index(st)
                    st_norm = normalize_labels(st_raw, strip=True, casefold=normalize_casefold)
                    rec["station_n"] = st_norm.size
                    rec["station_dtype"] = str(st_raw.dtype)
                    rec["station_dups"] = int(st_norm.size - pd.Index(st_norm).nunique())
                    station_sets.append(pd.Index(st_norm))

                    if station_ref_raw is None:
                        station_ref_raw = st_norm
                    else:
                        # Compare vs reference order and membership
                        ref_idx = pd.Index(station_ref_raw)
                        cur_idx = pd.Index(st_norm)
                        rec["station_missing_vs_ref"] = list(ref_idx.difference(cur_idx))
                        rec["station_extra_vs_ref"] = list(cur_idx.difference(ref_idx))
                        rec["station_reordered_vs_ref"] = not ref_idx.equals(cur_idx)
                else:
                    rec["station_n"] = 0
                    rec["station_dtype"] = None
                    rec["station_dups"] = 0
                    rec["station_missing_vs_ref"] = []
                    rec["station_extra_vs_ref"] = []
                    rec["station_reordered_vs_ref"] = False

                # CMIP6
                if cmip6_dim_name in ds.dims or cmip6_dim_name in ds.coords:
                    c6 = ds[cmip6_dim_name].values
                    c6_raw = pd.Index(c6)
                    c6_norm = normalize_labels(c6_raw, strip=True, casefold=normalize_casefold)
                    rec["cmip6_n"] = c6_norm.size
                    rec["cmip6_dtype"] = str(c6_raw.dtype)
                    rec["cmip6_dups"] = int(c6_norm.size - pd.Index(c6_norm).nunique())
                    cmip6_sets.append(pd.Index(c6_norm))

                    if cmip6_ref_raw is None:
                        cmip6_ref_raw = c6_norm
                    else:
                        ref_idx = pd.Index(cmip6_ref_raw)
                        cur_idx = pd.Index(c6_norm)
                        rec["cmip6_missing_vs_ref"] = list(ref_idx.difference(cur_idx))
                        rec["cmip6_extra_vs_ref"] = list(cur_idx.difference(ref_idx))
                        rec["cmip6_reordered_vs_ref"] = not ref_idx.equals(cur_idx)
                else:
                    rec["cmip6_n"] = 0
                    rec["cmip6_dtype"] = None
                    rec["cmip6_dups"] = 0
                    rec["cmip6_missing_vs_ref"] = []
                    rec["cmip6_extra_vs_ref"] = []
                    rec["cmip6_reordered_vs_ref"] = False

                per_file.append(rec)

        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")

    # Canonical sets
    station_union = pd.Index([]) if not station_sets else station_sets[0]
    cmip6_union = pd.Index([]) if not cmip6_sets else cmip6_sets[0]
    for i in range(1, len(station_sets)):
        station_union = station_union.union(station_sets[i])
    for i in range(1, len(cmip6_sets)):
        cmip6_union = cmip6_union.union(cmip6_sets[i])

    station_intersection = pd.Index([]) if not station_sets else station_sets[0]
    cmip6_intersection = pd.Index([]) if not cmip6_sets else cmip6_sets[0]
    for i in range(1, len(station_sets)):
        station_intersection = station_intersection.intersection(station_sets[i])
    for i in range(1, len(cmip6_sets)):
        cmip6_intersection = cmip6_intersection.intersection(cmip6_sets[i])

    summary = {
        "per_file": per_file,
        "station_union": list(station_union),
        "station_intersection": list(station_intersection),
        "cmip6_union": list(cmip6_union),
        "cmip6_intersection": list(cmip6_intersection),
        "recommendation": (
            "Use INTERSECTION to avoid NaNs during concat (exact join), "
            "or UNION only if you intend to keep missing values for added/removed stations."
        ),
    }

    if write_csv_dir:
        out_dir = Path(write_csv_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_file).to_csv(out_dir / "coord_scan_per_file.csv", index=False)
        pd.DataFrame({"station_union": summary["station_union"]}).to_csv(out_dir / "station_union.csv", index=False)
        pd.DataFrame({"station_intersection": summary["station_intersection"]}).to_csv(out_dir / "station_intersection.csv", index=False)
        pd.DataFrame({"cmip6_union": summary["cmip6_union"]}).to_csv(out_dir / "cmip6_union.csv", index=False)
        pd.DataFrame({"cmip6_intersection": summary["cmip6_intersection"]}).to_csv(out_dir / "cmip6_intersection.csv", index=False)
        print(f"  ↳ wrote coordinate scans to: {out_dir}")

    # Console warnings
    for rec in per_file:
        warn_bits = []
        if rec.get("station_dups", 0) > 0:
            warn_bits.append(f"station_dups={rec['station_dups']}")
        if rec.get("cmip6_dups", 0) > 0:
            warn_bits.append(f"cmip6_dups={rec['cmip6_dups']}")
        if rec.get("station_missing_vs_ref"):
            warn_bits.append(f"station_missing_vs_ref={len(rec['station_missing_vs_ref'])}")
        if rec.get("station_extra_vs_ref"):
            warn_bits.append(f"station_extra_vs_ref={len(rec['station_extra_vs_ref'])}")
        if rec.get("cmip6_missing_vs_ref"):
            warn_bits.append(f"cmip6_missing_vs_ref={len(rec['cmip6_missing_vs_ref'])}")
        if rec.get("cmip6_extra_vs_ref"):
            warn_bits.append(f"cmip6_extra_vs_ref={len(rec['cmip6_extra_vs_ref'])}")
        if rec.get("station_reordered_vs_ref"):
            warn_bits.append("station_reordered")
        if rec.get("cmip6_reordered_vs_ref"):
            warn_bits.append("cmip6_reordered")

        if warn_bits:
            print(f"[WARN] {rec['file']}: " + ", ".join(warn_bits))
        else:
            print(f"[OK]   {rec['file']}: station/cmip6 coords consistent with reference")

    return summary


# =============================== #
# NEW: Preflight concat risk diagnosis
# =============================== #

def diagnose_concat_risks(
    data_dir: str,
    file_glob: str = "*.nc",
    time_dim_name: str = "time",
    station_dim_name: str = "station",
    cmip6_dim_name: str = "cmip6",
    decode_cf: bool = True,
    engine: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Predict whether xarray.concat would cause reindexing (and thus NaNs) by checking:
      - Station/cmip6 coordinate equality across files (after normalization)
      - Time adjacency/overlaps at boundaries (uses start/end and prevailing step)
    """
    files = sorted(Path(data_dir).glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No files match {file_glob} in {data_dir}")

    open_kwargs = dict(decode_cf=decode_cf)
    if engine:
        open_kwargs["engine"] = engine

    recs = []
    per_file_steps = []

    for f in files:
        with xr.open_dataset(f, **open_kwargs) as ds:
            # time markers
            if time_dim_name in ds.dims or time_dim_name in ds.coords:
                t = _to_datetime64(ds[time_dim_name].values)
                start = t[0] if t.size else None
                end = t[-1] if t.size else None
                step = _infer_prevailing_step(np.diff(t)) if t.size > 1 else None
            else:
                start = None
                end = None
                step = None

            # coords normalized
            st_norm = normalize_labels(
                ds[station_dim_name].values if (station_dim_name in ds) else np.array([]),
                strip=True, casefold=False
            )
            c6_norm = normalize_labels(
                ds[cmip6_dim_name].values if (cmip6_dim_name in ds) else np.array([]),
                strip=True, casefold=False
            )

            recs.append({
                "file": f.name,
                "start": start,
                "end": end,
                "station_norm": pd.Index(st_norm),
                "cmip6_norm": pd.Index(c6_norm),
            })
            if step is not None:
                per_file_steps.append(step)

    # Sort by start time for boundary check
    recs = sorted(recs, key=lambda r: (pd.Timestamp(r["start"]) if r["start"] else pd.Timestamp.min, r["file"]))

    # Exact-equality checks for station/cmip6
    station_equal_all = all(recs[0]["station_norm"].equals(r["station_norm"]) for r in recs)
    cmip6_equal_all = all(recs[0]["cmip6_norm"].equals(r["cmip6_norm"]) for r in recs)

    # Typical prevailing step across files (mode)
    typical_step = None
    if per_file_steps:
        uniq, cnt = np.unique(np.array(per_file_steps), return_counts=True)
        typical_step = uniq[np.argmax(cnt)]

    # Boundaries
    boundary_flags = []
    for i in range(len(recs) - 1):
        curr, nxt = recs[i], recs[i+1]
        if curr["end"] is None or nxt["start"] is None:
            relation = "unknown"
            delta = None
        else:
            delta = nxt["start"] - curr["end"]
            if delta <= np.timedelta64(0, "ns"):
                relation = "overlap_or_duplicate"
            elif typical_step is not None and delta == typical_step:
                relation = "adjacent"
            else:
                relation = "gap"
        boundary_flags.append({
            "file_curr": curr["file"], "file_next": nxt["file"],
            "end_curr": _to_printable(curr["end"]), "start_next": _to_printable(nxt["start"]),
            "delta": _fmt_td(delta) if delta is not None else "None",
            "relation": relation
        })

    risk = {
        "will_reindex_stations": not station_equal_all,
        "will_reindex_cmip6": not cmip6_equal_all,
        "boundary_flags": boundary_flags,
        "recommendation": (
            "If 'will_reindex_*' is True, subset to a canonical INTERSECTION and use xr.concat(..., join='exact')."
        )
    }

    if risk["will_reindex_stations"] or risk["will_reindex_cmip6"]:
        print("[WARN] Concat would reindex non-time coords -> NaNs likely. See 'diagnose_concat_risks' output.")
    else:
        print("[OK]   Non-time coordinates are identical across files; concat should not introduce NaNs.")

    return risk


# =============================== #
# Your existing scanners (minor additions)
# =============================== #

def scan_time_consistency_infile(
    data_dir: str,
    file_glob: str = "*.nc",
    time_dim_name: str = "time",
    station_dim_name: str = "station",
    cmip6_dim_name: str = "cmip6",
    decode_cf: bool = True,
    mask_and_scale: bool = False,
    engine: Optional[str] = None,
    write_csv_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Check time consistency within each file using native numpy.datetime64.
    """
    files = sorted(Path(data_dir).glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No files match {file_glob} in {data_dir}")

    per_file: List[Dict[str, Any]] = []
    station_ref = None
    cmip6_ref = None
    coord_mismatch_files: List[str] = []

    for f in files:
        open_kwargs = dict(decode_cf=decode_cf, mask_and_scale=mask_and_scale)
        if engine:
            open_kwargs["engine"] = engine

        try:
            with xr.open_dataset(f, **open_kwargs) as ds:
                if time_dim_name not in ds.dims and time_dim_name not in ds.coords:
                    print(f"[SKIP] {f.name}: missing time dim/coord '{time_dim_name}'")
                    continue

                t_raw = ds[time_dim_name].values
                t = _to_datetime64(t_raw)
                n = t.size

                start_dt = t[0] if n > 0 else None
                end_dt = t[-1] if n > 0 else None

                diffs = np.diff(t) if n > 1 else np.array([], dtype="timedelta64[ns]")

                is_monotonic = bool(np.all(diffs >= DT_TOL)) if diffs.size else True
                duplicates_within = int(n) - int(len(pd.Index(t).unique())) if n > 0 else 0
                backward_steps = int(np.count_nonzero(diffs < np.timedelta64(0, "ns")))

                prevailing = _infer_prevailing_step(diffs)
                irregular_steps = (
                    int(np.count_nonzero((diffs > np.timedelta64(0, "ns")) & (diffs != prevailing)))
                    if prevailing is not None
                    else 0
                )

                pos = diffs[diffs > np.timedelta64(0, "ns")]
                min_step = _fmt_td(pos.min() if pos.size else None)
                med_step = _fmt_td(np.sort(pos)[pos.size // 2] if pos.size else None)
                max_step = _fmt_td(pos.max() if pos.size else None)

                per_file.append(
                    {
                        "file": f.name,
                        "n_steps": n,
                        "monotonic": is_monotonic,
                        "duplicates_within": duplicates_within,
                        "backward_steps": backward_steps,
                        "prevailing_step": _fmt_td(prevailing),
                        "irregular_steps": irregular_steps,
                        "min_step": min_step,
                        "median_step": med_step,
                        "max_step": max_step,
                        "start": pd.Timestamp(start_dt) if start_dt is not None else pd.NaT,
                        "end": pd.Timestamp(end_dt) if end_dt is not None else pd.NaT,
                    }
                )

                # Coordinate consistency (station/cmip6) — normalized to reduce false mismatches
                if station_dim_name in ds.dims or station_dim_name in ds.coords:
                    stations_norm = normalize_labels(ds[station_dim_name].values)
                    if station_ref is None:
                        station_ref = stations_norm
                    else:
                        if (len(stations_norm) != len(station_ref)) or not np.array_equal(stations_norm, station_ref):
                            coord_mismatch_files.append(f.name)

                if cmip6_dim_name in ds.dims or cmip6_dim_name in ds.coords:
                    cmip6_norm = normalize_labels(ds[cmip6_dim_name].values)
                    if cmip6_ref is None:
                        cmip6_ref = cmip6_norm
                    else:
                        if (len(cmip6_norm) != len(cmip6_ref)) or not np.array_equal(cmip6_norm, cmip6_ref):
                            coord_mismatch_files.append(f.name)

        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")

    summary = {
        "per_file": per_file,
        "coord_mismatch_files": sorted(set(coord_mismatch_files)),
    }

    if write_csv_dir:
        out_dir = Path(write_csv_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_file).to_csv(out_dir / "time_scan_per_file.csv", index=False)
        if summary["coord_mismatch_files"]:
            pd.DataFrame({"file": summary["coord_mismatch_files"]}).to_csv(
                out_dir / "coord_mismatch_files.csv", index=False
            )
        print(f"  ↳ wrote time scan: {out_dir / 'time_scan_per_file.csv'}")
        if summary["coord_mismatch_files"]:
            print(f"  ↳ wrote coord mismatch list: {out_dir / 'coord_mismatch_files.csv'}")

    for rec in per_file:
        if ((rec["duplicates_within"] > 0)
            or (rec["backward_steps"] > 0)
            or (rec["irregular_steps"] > 0)
            or (not rec["monotonic"])):
            print(
                f"[WARN] {rec['file']}: time issues — "
                f"monotonic={rec['monotonic']}, dup={rec['duplicates_within']}, "
                f"backward={rec['backward_steps']}, irregular={rec['irregular_steps']}; "
                f"n={rec['n_steps']}, prevailing_step={rec['prevailing_step']}, "
                f"min/med/max={rec['min_step']}/{rec['median_step']}/{rec['max_step']}"
            )
        else:
            print(f"[OK]   {rec['file']}: time looks clean (n={rec['n_steps']}, prevailing_step={rec['prevailing_step']})")

    if summary["coord_mismatch_files"]:
        print(
            f"[WARN] {len(summary['coord_mismatch_files'])} files have coordinate mismatches (station/cmip6). "
            f"Concatenation may reindex and introduce NaNs.\n"
            f"  Files: {', '.join(summary['coord_mismatch_files'][:5])} ..."
        )

    return summary


def concat_time_and_global_checks(
    data_dir: str,
    file_glob: str = "*.nc",
    time_dim_name: str = "time",
    decode_cf: bool = True,
    mask_and_scale: bool = False,
    engine: Optional[str] = None,
    write_csv_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Concatenate native datetime64 time axes from all files, then check global issues.
    """
    files = sorted(Path(data_dir).glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No files match {file_glob} in {data_dir}")

    records: List[Dict[str, Any]] = []

    for f in files:
        open_kwargs = dict(decode_cf=decode_cf, mask_and_scale=mask_and_scale)
        if engine:
            open_kwargs["engine"] = engine

        try:
            with xr.open_dataset(f, **open_kwargs) as ds:
                if time_dim_name not in ds.dims and time_dim_name not in ds.coords:
                    print(f"[SKIP] {f.name}: missing time dim/coord '{time_dim_name}'")
                    continue

                t = _to_datetime64(ds[time_dim_name].values)
                if t.size == 0:
                    print(f"[SKIP] {f.name}: empty time coordinate")
                    continue

                records.append(
                    {
                        "file": f.name,
                        "start": t[0],
                        "end": t[-1],
                        "time_vals": t,
                    }
                )

        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")

    if not records:
        raise RuntimeError("No usable time coordinates were found.")

    records_sorted = sorted(records, key=lambda r: (r["start"], r["file"]))

    all_times = np.concatenate([r["time_vals"] for r in records_sorted])
    n_total = all_times.size

    diffs = np.diff(all_times) if n_total > 1 else np.array([], dtype="timedelta64[ns]")
    is_monotonic_global = bool(np.all(diffs >= DT_TOL)) if diffs.size else True
    duplicates_total = int(n_total) - int(len(pd.Index(all_times).unique()))
    backward_total = int(np.count_nonzero(diffs < np.timedelta64(0, "ns")))
    prevailing = _infer_prevailing_step(diffs)
    irregular_total = (
        int(np.count_nonzero((diffs > np.timedelta64(0, "ns")) & (diffs != prevailing)))
        if prevailing is not None
        else 0
    )

    boundary_rows = []
    for i in range(len(records_sorted) - 1):
        curr = records_sorted[i]
        nxt = records_sorted[i + 1]
        delta = nxt["start"] - curr["end"]  # timedelta64
        relation = (
            "overlap_or_duplicate" if delta <= np.timedelta64(0, "ns")
            else "adjacent" if (prevailing is not None and delta == prevailing)
            else "non_adjacent_gap"
        )
        boundary_rows.append(
            {
                "file_curr": curr["file"],
                "file_next": nxt["file"],
                "end_curr": pd.Timestamp(curr["end"]),
                "start_next": pd.Timestamp(nxt["start"]),
                "delta": _fmt_td(delta),
                "relation": relation,
            }
        )

    summary = {
        "status": "ok",
        "files_order": [r["file"] for r in records_sorted],
        "n_total_steps": n_total,
        "start": pd.Timestamp(records_sorted[0]["start"]),
        "end": pd.Timestamp(records_sorted[-1]["end"]),
        "monotonic_global": is_monotonic_global,
        "duplicates_total": duplicates_total,
        "backward_total": backward_total,
        "prevailing_step_global": _fmt_td(prevailing),
        "irregular_total": irregular_total,
        "boundary_deltas": boundary_rows,
    }

    if write_csv_dir:
        out_dir = Path(write_csv_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(boundary_rows).to_csv(out_dir / "time_concat_boundaries.csv", index=False)
        pd.DataFrame([summary]).drop(columns=["boundary_deltas"]).to_csv(
            out_dir / "time_concat_global_summary.csv", index=False
        )
        print(f"  ↳ wrote concatenation boundaries: {out_dir / 'time_concat_boundaries.csv'}")
        print(f"  ↳ wrote global summary: {out_dir / 'time_concat_global_summary.csv'}")

    print("\n[GLOBAL] Concatenated time axis check:")
    print(f"  Files concatenated (sorted): {len(records_sorted)}")
    print(f"  Total steps: {n_total} | Start: {summary['start']} | End: {summary['end']}")
    print(
        f"  Monotonic={is_monotonic_global}, Duplicates={duplicates_total}, "
        f"Backward={backward_total}, Irregular={irregular_total}, "
        f"Prevailing step≈{summary['prevailing_step_global']}"
    )

    if (
        (duplicates_total == 0)
        and (backward_total == 0)
        and (irregular_total == 0)
        and is_monotonic_global
    ):
        print("[OK]   Concatenation yields a clean, consistent series with a single prevailing step.")
    else:
        print("[WARN] Concatenation has issues. See metrics above and boundaries CSV (if written).")

    return summary

def scan_time_consistency(
    data_dir: str,
    file_glob: str = "*.nc",
    time_dim_name: str = "time",
    station_dim_name: str = "station",
    cmip6_dim_name: str = "cmip6",
    expected_freq: str = "infer",         # <— NEW: 'infer' or any fixed duration like '10min', 'H', 'D'
    decode_cf: bool = True,
    mask_and_scale: bool = False,
    engine: Optional[str] = None,
    write_csv_dir: Optional[str] = None,
    print_top_boundary_issues: int = 10,
    diff_round_decimals: int = 12,        # rounding for CF-numeric diffs to reduce jitter
) -> Dict[str, Any]:
    """
    CF-numeric time consistency check within each file and across files, with support for
    sub-daily sampling (e.g., '10min', 'H') or auto-inference ('infer').

    Behavior:
      - Within each file, converts time to float-days since epoch and checks monotonicity.
      - Compares per-step diffs to the expected step (either provided or inferred) using tolerance.
      - Reports counts of 'gaps_within' (diff > expected) and 'overlaps_within' (diff < expected).
        Note: 'overlaps_within' here means "understep vs expected", not duplicate timestamps.
      - Across files, checks adjacency using the expected step: gap vs overlap.

    Limitations:
      - Month-based frequencies like 'MS' are not fixed durations; use datetime64 path instead.
    """
    files = sorted(Path(data_dir).glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No files match {file_glob} in {data_dir}")

    per_file: List[Dict[str, Any]] = []
    all_boundaries: List[Tuple[str, float, float, str, Any, Any]] = []
    coord_mismatch_files: List[str] = []
    units_set = set()
    calendars_set = set()

    # For inference: collect positive diffs (in days) across all files
    all_pos_diffs_days: List[float] = []

    # Temporary references to detect coord mismatches
    station_ref = None
    cmip6_ref = None

    for f in files:
        open_kwargs = dict(decode_cf=decode_cf, mask_and_scale=mask_and_scale)
        if engine:
            open_kwargs["engine"] = engine

        try:
            with xr.open_dataset(f, **open_kwargs) as ds:
                if time_dim_name not in ds.dims and time_dim_name not in ds.coords:
                    print(f"[SKIP] {f.name}: missing time dim/coord '{time_dim_name}'")
                    continue

                time_var = ds[time_dim_name]
                t = time_var.values
                cal = time_var.attrs.get("calendar", "standard")
                units = time_var.attrs.get("units", None)
                calendars_set.add(cal)
                if units:
                    units_set.add(units)

                # Convert to float-days (robust across CF numeric & cftime)
                nums = _to_num_days(t, calendar=cal, units=units)
                n = len(nums)

                start_dt = t[0] if n > 0 else None
                end_dt = t[-1] if n > 0 else None
                start_num = nums[0] if n > 0 else np.nan
                end_num = nums[-1] if n > 0 else np.nan

                # Diffs in days; rounded to reduce floating jitter
                diffs = np.diff(np.round(nums, diff_round_decimals)) if n > 1 else np.array([], dtype=float)
                is_monotonic = bool(np.all(diffs >= -TOL)) if diffs.size else True

                # Duplicate numeric steps within the file (after rounding)
                dup_within = int(n) - int(len(pd.Index(np.round(nums, diff_round_decimals)).unique())) if n > 0 else 0

                # For inference: collect positive diffs in days
                pos_diffs = diffs[diffs > 0.0]
                if pos_diffs.size:
                    all_pos_diffs_days.extend(pos_diffs.tolist())

                # Temporarily set expected step to 1 day for span-based estimate;
                # we'll recompute expected_count later once final expected step is known.
                expected_count_placeholder = None

                per_file.append(
                    {
                        "file": f.name,
                        "n_steps": n,
                        "expected_count_from_span": expected_count_placeholder,  # set later
                        "monotonic": is_monotonic,
                        "duplicates_within": dup_within,
                        "gaps_within": None,          # set later
                        "overlaps_within": None,      # set later (understeps)
                        "calendar": cal,
                        "units": units,
                        "start": _to_printable(start_dt),
                        "end": _to_printable(end_dt),
                        "_start_num": start_num,
                        "_end_num": end_num,
                        "_diffs": diffs,              # cache for second pass
                    }
                )

                # Coordinate equality (normalized)
                if station_dim_name in ds.dims or station_dim_name in ds.coords:
                    stations_norm = normalize_labels(ds[station_dim_name].values)
                    if station_ref is None:
                        station_ref = stations_norm
                    else:
                        if stations_norm.shape != station_ref.shape or not np.array_equal(stations_norm, station_ref):
                            coord_mismatch_files.append(f.name)

                if cmip6_dim_name in ds.dims or cmip6_dim_name in ds.coords:
                    cmip6_norm = normalize_labels(ds[cmip6_dim_name].values)
                    if cmip6_ref is None:
                        cmip6_ref = cmip6_norm
                    else:
                        if cmip6_norm.shape != cmip6_ref.shape or not np.array_equal(cmip6_norm, cmip6_ref):
                            coord_mismatch_files.append(f.name)

        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")

    # --- Determine expected step (days) ---
    if expected_freq == "infer":
        if not all_pos_diffs_days:
            # Fallback if we couldn't infer
            raise RuntimeError("Cannot infer expected step: no positive diffs found across files.")
        # Round diffs to reduce jitter then take the mode
        diffs_rounded = np.round(np.array(all_pos_diffs_days, dtype=float), diff_round_decimals)
        expected_step_days = _mode_value(diffs_rounded)
        expected_freq_str = f"infer → {expected_step_days:.12f} days"
    else:
        expected_step_days = _parse_expected_step_days(expected_freq)
        expected_freq_str = expected_freq

    # --- Second pass: compute gaps/understeps within files & cross-file adjacency ---
    cross_issues: List[Dict[str, Any]] = []
    all_boundaries_sorted = sorted(
        [
            (rec["file"], rec["_start_num"], rec["_end_num"], rec["calendar"], rec["start"], rec["end"])
            for rec in per_file
        ],
        key=lambda r: (np.nan_to_num(r[1], nan=-1e99), r[0])
    )

    # Fill per-file metrics now that we know expected step
    for rec in per_file:
        diffs = rec["_diffs"]
        if diffs.size:
            gaps_mask = diffs > (expected_step_days + TOL)
            under_mask = diffs < (expected_step_days - TOL)  # "overlaps_within" in prior naming
            num_gaps = int(np.count_nonzero(gaps_mask))
            num_under = int(np.count_nonzero(under_mask))
        else:
            num_gaps = 0
            num_under = 0

        # Expected count based on span and expected step
        if rec["n_steps"] > 0 and not np.isnan(rec["_start_num"]) and not np.isnan(rec["_end_num"]):
            span_days = rec["_end_num"] - rec["_start_num"]
            expected_count = int(round(span_days / expected_step_days)) + 1
        else:
            expected_count = 0

        rec["expected_count_from_span"] = expected_count
        rec["gaps_within"] = num_gaps
        rec["overlaps_within"] = num_under
        # Clean cache fields from final dict
        for k in ("_start_num", "_end_num", "_diffs"):
            rec.pop(k, None)

    # Cross-file adjacency checks
    for i in range(len(all_boundaries_sorted) - 1):
        f_curr, s_num, e_num, cal_curr, s_dt, e_dt = all_boundaries_sorted[i]
        f_next, s2_num, e2_num, cal_next, s2_dt, e2_dt = all_boundaries_sorted[i + 1]

        if cal_curr != cal_next:
            cross_issues.append({
                "pair": (f_curr, f_next),
                "issue": "calendar_mismatch",
                "calendar_curr": cal_curr,
                "calendar_next": cal_next
            })

        gap_measure = (s2_num - e_num) - expected_step_days
        if gap_measure > TOL:
            cross_issues.append({
                "pair": (f_curr, f_next),
                "issue": "gap_between_files",
                "gap_days": gap_measure,
                "end_curr": _to_printable(e_dt),
                "start_next": _to_printable(s2_dt)
            })
        elif gap_measure < -TOL:
            cross_issues.append({
                "pair": (f_curr, f_next),
                "issue": "overlap_between_files",
                "overlap_days": -gap_measure,
                "end_curr": _to_printable(e_dt),
                "start_next": _to_printable(s2_dt)
            })

    summary = {
        "per_file": per_file,
        "cross_file_issues": cross_issues,
        "coord_mismatch_files": sorted(set(coord_mismatch_files)),
        "time_units_used": sorted(units_set),
        "time_calendars_used": sorted(calendars_set),
        "expected_step_days": expected_step_days,
        "expected_freq_used": expected_freq_str,
    }

    # --- Optional CSV outputs ---
    if write_csv_dir:
        out_dir = Path(write_csv_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_file).to_csv(out_dir / "time_scan_per_file.csv", index=False)
        pd.DataFrame(cross_issues).to_csv(out_dir / "time_scan_cross_file.csv", index=False)
        meta_df = pd.DataFrame([{
            "expected_step_days": expected_step_days,
            "expected_freq_used": expected_freq_str,
            "time_calendars_used": ";".join(sorted(calendars_set)),
            "time_units_used": ";".join(sorted(units_set)),
        }])
        meta_df.to_csv(out_dir / "time_scan_meta.csv", index=False)
        if summary["coord_mismatch_files"]:
            pd.DataFrame({"file": summary["coord_mismatch_files"]}).to_csv(out_dir / "coord_mismatch_files.csv", index=False)
        print(f"  ↳ wrote time scan: {out_dir / 'time_scan_per_file.csv'}")
        print(f"  ↳ wrote cross-file issues: {out_dir / 'time_scan_cross_file.csv'}")
        print(f"  ↳ wrote meta: {out_dir / 'time_scan_meta.csv'}")
        if summary["coord_mismatch_files"]:
            print(f"  ↳ wrote coord mismatch list: {out_dir / 'coord_mismatch_files.csv'}")

    # --- Console summary like your original ---
    for rec in per_file:
        if ((rec["duplicates_within"] > 0)
            or (rec["gaps_within"] > 0)
            or (rec["overlaps_within"] > 0)
            or (not rec["monotonic"])):
            print(
                f"[WARN] {rec['file']}: time issues — "
                f"monotonic={rec['monotonic']}, dup={rec['duplicates_within']}, "
                f"gaps={rec['gaps_within']}, overlaps={rec['overlaps_within']}; "
                f"n={rec['n_steps']}, expected_from_span={rec['expected_count_from_span']}, "
                f"calendar={rec['calendar']}, units={rec['units']}"
            )
        else:
            print(
                f"[OK]   {rec['file']}: time looks clean "
                f"(n={rec['n_steps']}, calendar={rec['calendar']}, units={rec['units']})"
            )

    if cross_issues:
        print(f"Cross-file adjacency issues (expected_freq={expected_freq_str}; top few):")
        for row in cross_issues[:print_top_boundary_issues]:
            qty = row.get("gap_days") or row.get("overlap_days")
            print(f"  {row['pair'][0]} → {row['pair'][1]}: {row['issue']} ({qty})")
    else:
        print(f"[OK]   Cross-file time adjacency: perfect (no gaps/overlaps; expected_freq={expected_freq_str}).")

    if summary["coord_mismatch_files"]:
        print(
            f"[WARN] {len(summary['coord_mismatch_files'])} files have coordinate mismatches (station/cmip6). "
            f"Concatenation may reindex and introduce NaNs.\n"
            f"  Files: {', '.join(summary['coord_mismatch_files'][:5])} ..."
        )

    # Mixed calendars/units can also cause subtle reindexing issues
    if len(summary["time_calendars_used"]) > 1:
        print(f"[WARN] Mixed time calendars found: {summary['time_calendars_used']}")
    if len(summary["time_units_used"]) > 1:
        print(f"[WARN] Mixed time units found: {summary['time_units_used']}")

    return summary


def scan_files_for_nans(
    data_dir: str,
    var_name: str,
    file_glob: str = "*.nc",
    station_dim_name: str = "station",
    cmip6_dim_name: str = "cmip6",
    time_dim_name: str = "time",
    chunks_time: int = 365,
    print_top: int = 10,
    write_csv_dir: Optional[str] = None,
) -> None:
    """
    Iterate NetCDF files one-by-one and print a summary ONLY when NaNs are detected.
    """
    files = sorted(Path(data_dir).glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No files match {file_glob} in {data_dir}")

    for f in files:
        try:
            with xr.open_dataset(
                f,
                decode_cf=True,
                chunks={time_dim_name: chunks_time} if chunks_time else None,
            ) as ds:
                if var_name not in ds:
                    print(f"[SKIP] {f.name}: variable '{var_name}' not found. Available: {list(ds.data_vars)}")
                    continue

                da = ds[var_name]

                dims_lower_to_orig = {d.lower(): d for d in da.dims}
                names_lower = {
                    "time": time_dim_name.lower(),
                    "station": station_dim_name.lower(),
                    "cmip6": cmip6_dim_name.lower(),
                }
                missing = [k for k, v in names_lower.items() if v not in dims_lower_to_orig]
                if missing:
                    print(f"[SKIP] {f.name}: missing dims {missing} in {da.dims}")
                    continue

                time_dim = dims_lower_to_orig[names_lower["time"]]
                station_dim = dims_lower_to_orig[names_lower["station"]]
                cmip6_dim = dims_lower_to_orig[names_lower["cmip6"]]

                da = da.transpose(time_dim, station_dim, cmip6_dim)
                T = int(da.sizes[time_dim])
                S = int(da.sizes[station_dim])
                C = int(da.sizes[cmip6_dim])
                total_steps_all = T * S * C

                has_nan = bool(da.isnull().any().compute())
                if not has_nan:
                    print(f"[OK]   {f.name}: no NaNs (time={T}, station={S}, cmip6={C})")
                    continue

                M = da.isnull()
                nan_steps = M.sum(dim=time_dim)
                total_nan_steps = int(nan_steps.sum().compute())
                nan_fraction = nan_steps / T

                affected_mask = (nan_steps > 0).compute()
                num_pairs_affected = int(affected_mask.sum().item())

                print(
                    f"[WARN] {f.name}: NaNs detected — "
                    f"{num_pairs_affected}/{S * C} pairs affected; "
                    f"total_nan_steps={total_nan_steps}/{total_steps_all} "
                    f"({total_nan_steps / total_steps_all:.2%})"
                )

                if print_top and num_pairs_affected > 0:
                    nan_fraction_comp = nan_fraction.compute()
                    top_df = (
                        nan_fraction_comp.to_dataset(name="nan_fraction")
                        .to_dataframe()
                        .reset_index()
                        .sort_values("nan_fraction", ascending=False)
                        .head(print_top)
                    )
                    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
                    print("Top pairs by NaN fraction in this file:")
                    print(top_df.to_string(index=False))
                    pd.reset_option("display.float_format")

                if write_csv_dir:
                    out_dir = Path(write_csv_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    nan_steps_comp = nan_steps.compute()
                    nan_fraction_comp = nan_fraction.compute()
                    ds_out = xr.Dataset({"nan_steps": nan_steps_comp, "nan_fraction": nan_fraction_comp})
                    df_out = ds_out.to_dataframe().reset_index()
                    df_out["total_steps"] = T
                    df_out = df_out.sort_values([station_dim, cmip6_dim])
                    out_csv = out_dir / f"{Path(f).stem}_nan_summary.csv"
                    df_out.to_csv(out_csv, index=False)
                    print(f"  ↳ wrote per-file summary: {out_csv}")

        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")


# =============================== #
# NEW: Robust subset + concat
# =============================== #

def _assign_norm_and_align(
    ds: xr.Dataset,
    dim_name: str,
    canon_norm: pd.Index,
    *,
    casefold: bool = False,
) -> xr.Dataset:
    """
    For ds- normalize labels (strip + optional casefold)
      - assign normalized labels back into the coordinate
      - select intersection with `canon_norm`
      - reindex to canonical order (exact match; no fill)

    Returns new dataset subset whose ds[dim_name] aligns to canon_norm and preserves canonical order.
    Raises if no overlap.
    """
    if dim_name not in ds:
        return ds

    # Normalize current labels
    current_vals = ds[dim_name].values
    current_norm = pd.Index(normalize_labels(current_vals, strip=True, casefold=casefold))

    # Assign normalized labels into the coordinate
    ds = ds.assign_coords({dim_name: (ds[dim_name].dims, current_norm.values)})

    # Determine intersection & enforce canonical order
    pick = current_norm.intersection(canon_norm)
    if pick.size == 0:
        raise ValueError(f"No overlap between dataset.{dim_name} and canonical set")

    ds = ds.sel({dim_name: pick.values})
    ds = ds.reindex({dim_name: canon_norm}, method=None, copy=False)

    return ds


def safe_load_and_concat_subset(
    files: List[str],
    station_dim_name: str,
    cmip6_dim_name: str,
    preprocess=None,
    engine: Optional[str] = None,
    label: str = "dataset",
    enforce_intersection: bool = True,
    expected_freq: Optional[str] = None,  # duration parseable by pd.to_timedelta (e.g., 'D','H','15min')
) -> xr.Dataset:
    """
    Memory-aware loader:
      - Normalizes station/cmip6 labels, determines canonical set (intersection by default)
      - Subsets each file to canonical set and enforces identical order
      - Loads only subsets and concatenates along 'time'
      - Uses xr.concat(..., join='exact') when available to avoid silent reindexing
      - Ensures unique sorted time and asserts regular sampling (optional)

    Raises early if coordinates cannot be made identical across files.
    """
    if not files:
        raise FileNotFoundError(f"{label}: no input files found")

    # Discover canonical normalized sets
    station_sets = []
    cmip6_sets = []
    open_kwargs = {}
    if engine:
        open_kwargs["engine"] = engine

    for fp in sorted(files):
        with xr.open_dataset(fp, decode_cf=True, **open_kwargs) as ds:
            if (station_dim_name not in ds) and (cmip6_dim_name not in ds):
                raise KeyError(f"{os.path.basename(fp)}: missing '{station_dim_name}' or '{cmip6_dim_name}'")

            st_norm = pd.Index(
                normalize_labels(ds[station_dim_name].values) if station_dim_name in ds else []
            )
            c6_norm = pd.Index(
                normalize_labels(ds[cmip6_dim_name].values) if cmip6_dim_name in ds else []
            )

            station_sets.append(st_norm)
            cmip6_sets.append(c6_norm)

    # Canonical sets
    station_canon = station_sets[0]
    cmip6_canon = cmip6_sets[0]
    if enforce_intersection:
        for s in station_sets[1:]:
            station_canon = station_canon.intersection(s)
        for c in cmip6_sets[1:]:
            cmip6_canon = cmip6_canon.intersection(c)
    else:
        for s in station_sets[1:]:
            station_canon = station_canon.union(s)
        for c in cmip6_sets[1:]:
            cmip6_canon = cmip6_canon.union(c)

    if station_canon.size == 0:
        raise ValueError(f"{label}: station canonical set empty — cannot concatenate without introducing NaNs.")
    if cmip6_canon.size == 0:
        print(f"[INFO] {label}: empty cmip6 canonical set; proceeding with station-only axis")

    # Build datasets list with consistent ordering
    dsets = []
    for fp in sorted(files):
        with xr.open_dataset(fp, decode_cf=True, **open_kwargs) as ds:
            try:
                if preprocess is not None:
                    ds = preprocess(ds)

                if station_dim_name in ds:
                    ds = _assign_norm_and_align(ds, station_dim_name, station_canon, casefold=False)
                if cmip6_dim_name in ds and cmip6_canon.size > 0:
                    ds = _assign_norm_and_align(ds, cmip6_dim_name, cmip6_canon, casefold=False)

                ds = ds.load()  # materialize ONLY the subset
                dsets.append(ds)
            except Exception:
                # context manager closes on exit; re-raise
                raise

    if not dsets:
        raise ValueError(f"{label}: no data for requested station/cmip6 subset")

    # Concat with exact join when available (xarray >= ~0.20)
    concat_kwargs = dict(dim="time", data_vars="minimal", coords="minimal")
    try:
        ds_cat = xr.concat(dsets, join="exact", **concat_kwargs)
    except Exception:
        # Older xarray without 'join' parameter — fallback; we pre-enforced coordinates anyway.
        ds_cat = xr.concat(dsets, **concat_kwargs)

    ds_cat = ensure_unique_sorted_time(ds_cat, time_dim="time")
    assert_regular_time(ds_cat, time_dim="time", expected_freq=expected_freq, label=label)
    return ds_cat


# =============================== #
# EXAMPLE: Run on your data
# =============================== #

if __name__ == "__main__":
    # Paths (adjust if needed)
    data_dir = r"D:\Kai\DFM\cdf_diff\000"
    file_glob = "ERA5wl_Diff_*.nc"
    write_dir = r"D:\Kai\DFM\cdf_diff\nan_scans"

    # 0) Coordinate consistency scan
    coord_summary = scan_coordinate_consistency(
        data_dir=data_dir,
        file_glob=file_glob,
        station_dim_name="station",
        cmip6_dim_name="cmip6",
        decode_cf=True,
        engine=None,
        write_csv_dir=write_dir,
        normalize_casefold=False,  # set True if case variations exist
    )

    # 1) NaN scan per-file (existing)
    scan_files_for_nans(
        data_dir=data_dir,
        var_name="cmip_diff",
        file_glob=file_glob,
        station_dim_name="station",
        cmip6_dim_name="cmip6",
        time_dim_name="time",
        chunks_time=365,
        print_top=10,
        write_csv_dir=write_dir,
    )

    # 2) Time consistency scan (existing, extended CF-numeric)
    scan_time_consistency(
        data_dir=data_dir,
        file_glob=file_glob,
        time_dim_name="time",
        station_dim_name="station",
        cmip6_dim_name="cmip6",
        decode_cf=True,
        mask_and_scale=False,
        engine=None,
        write_csv_dir=write_dir,
    )

    # 3) Global concatenated time axis (existing)
    concat_time_and_global_checks(
        data_dir=data_dir,
        file_glob=file_glob,
        time_dim_name="time",
        decode_cf=True,
        mask_and_scale=False,
        engine=None,
        write_csv_dir=write_dir,
    )

    # 4) Preflight: predict concat risks
    diagnose_concat_risks(
        data_dir=data_dir,
        file_glob=file_glob,
        time_dim_name="time",
        station_dim_name="station",
        cmip6_dim_name="cmip6",
        decode_cf=True,
        engine=None,
    )

    # 5) Robust subset + concat
    files = [str(p) for p in sorted(Path(data_dir).glob(file_glob))]
    ds_cat = safe_load_and_concat_subset(
        files=files,
        station_dim_name="station",
        cmip6_dim_name="cmip6",
        preprocess=None,        # or your _preprocess
        engine=None,
        label="cmip_diff_concat",
        enforce_intersection=True,   # safest to avoid NaNs
        expected_freq="10min",           # any duration parseable by pd.to_timedelta (e.g., '15min', 'H', '2D')
    )
    print(ds_cat)
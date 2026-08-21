from pathlib import Path
import math
import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional, List, Dict, Any
import cftime

# ---------- Constants ----------
TOL = 1e-8  # unified numerical tolerance for step comparisons & monotonic checks


# ---------- Helpers for native datetime64 ----------

DT_TOL = np.timedelta64(0, "ns")  # tolerant non-decreasing (allow equal for duplicates)


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

    # Convert with pandas (handles python datetime, numpy datetime64, DatetimeIndex)
    ts = pd.to_datetime(times)

    # Normalize any tz-aware values to UTC and drop tz
    try:
        if getattr(ts, "tz", None) is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
    except Exception:
        # elementwise fallback
        ts = pd.to_datetime(
            [
                (x.tz_convert("UTC").tz_localize(None) if getattr(x, "tz", None) else x)
                for x in ts
            ]
        )

    return ts.to_numpy(dtype="datetime64[ns]")


def _infer_prevailing_step(diffs: np.ndarray) -> Optional[np.timedelta64]:
    """
    Given an array of timedelta64 diffs, return the most common positive delta (prevailing step).
    Returns None if there are no positive diffs.
    """
    if diffs.size == 0:
        return None
    pos = diffs[diffs > np.timedelta64(0, "ns")]
    if pos.size == 0:
        return None
    # Use unique with counts to find the mode
    unique_vals, counts = np.unique(pos, return_counts=True)
    return unique_vals[np.argmax(counts)]


def _fmt_td(td: Optional[np.timedelta64]) -> str:
    if td is None:
        return "None"
    return str(pd.Timedelta(td))


def _expected_step_days(expected_freq: str) -> float:
    ef = expected_freq.upper()
    if ef == "D":
        return 1.0
    elif ef == "H":
        return 1.0 / 24.0
    else:
        raise ValueError("expected_freq must be 'D' or 'H'.")


def _to_num_days(
    times: np.ndarray,
    calendar: str = "standard",
    units: Optional[str] = None,
) -> np.ndarray:
    """
    Convert an array of datetime-like OR numeric CF times to float64 days since 1970-01-01.
    - If input is cftime dates: cftime.date2num
    - If input is pandas/numpy datetimes: normalized to UTC, then to epoch days
    - If input is numeric CF times: require CF 'units' to num2date -> date2num
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
        # Convert numeric to cftime datetimes, then back to epoch days
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
    ts = pd.to_datetime(times)

    # If it's a DatetimeIndex or Series, normalize any tz-aware to UTC and drop tz
    try:
        if getattr(ts, "tz", None) is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
    except Exception:
        # Fallback: elementwise normalization
        ts = pd.to_datetime(
            [
                (x.tz_convert("UTC").tz_localize(None) if getattr(x, "tz", None) else x)
                for x in ts
            ]
        )

    # Convert to epoch days
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


# ---------- 1) Per-file time checks (monotonic, duplicates, irregular steps) ----------


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
    Check time consistency *within* each file using native numpy.datetime64- Non-decreasing monotonicity (tolerant)
      - Duplicates
      - Backward steps
      - Irregular steps relative to auto-inferred prevailing step (mode of positive deltas)
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

                # Monotonic non-decreasing (allow duplicates)
                is_monotonic = bool(np.all(diffs >= DT_TOL)) if diffs.size else True

                # Duplicates: repeated timestamps anywhere
                duplicates_within = (
                    int(n) - int(len(pd.Index(t).unique())) if n > 0 else 0
                )

                # Backward steps: negative diffs
                backward_steps = int(np.count_nonzero(diffs < np.timedelta64(0, "ns")))

                # Step stats & irregular steps (vs prevailing)
                prevailing = _infer_prevailing_step(diffs)
                irregular_steps = (
                    int(
                        np.count_nonzero(
                            (diffs > np.timedelta64(0, "ns")) & (diffs != prevailing)
                        )
                    )
                    if prevailing is not None
                    else 0
                )

                # Min/median/max positive step (for reporting)
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
                        "start": pd.Timestamp(start_dt)
                        if start_dt is not None
                        else pd.NaT,
                        "end": pd.Timestamp(end_dt) if end_dt is not None else pd.NaT,
                    }
                )

                # Coordinate consistency (station/cmip6)
                if station_dim_name in ds.dims or station_dim_name in ds.coords:
                    stations = ds[station_dim_name].values
                    if station_ref is None:
                        station_ref = stations
                    else:
                        if stations.shape != station_ref.shape or not np.array_equal(
                            stations, station_ref
                        ):
                            coord_mismatch_files.append(f.name)

                if cmip6_dim_name in ds.dims or cmip6_dim_name in ds.coords:
                    cmip6 = ds[cmip6_dim_name].values
                    if cmip6_ref is None:
                        cmip6_ref = cmip6
                    else:
                        if cmip6.shape != cmip6_ref.shape or not np.array_equal(
                            cmip6, cmip6_ref
                        ):
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
            print(
                f"  ↳ wrote coord mismatch list: {out_dir / 'coord_mismatch_files.csv'}"
            )

    # Console output
    for rec in per_file:
        if (
            (rec["duplicates_within"] > 0)
            or (rec["backward_steps"] > 0)
            or (rec["irregular_steps"] > 0)
            or (not rec["monotonic"])
        ):
            print(
                f"[WARN] {rec['file']}: time issues — "
                f"monotonic={rec['monotonic']}, dup={rec['duplicates_within']}, "
                f"backward={rec['backward_steps']}, irregular={rec['irregular_steps']}; "
                f"n={rec['n_steps']}, prevailing_step={rec['prevailing_step']}, "
                f"min/med/max={rec['min_step']}/{rec['median_step']}/{rec['max_step']}"
            )
        else:
            print(
                f"[OK]   {rec['file']}: time looks clean (n={rec['n_steps']}, prevailing_step={rec['prevailing_step']})"
            )

    if summary["coord_mismatch_files"]:
        print(
            f"[WARN] {len(summary['coord_mismatch_files'])} files have coordinate mismatches (station/cmip6). "
            f"Concatenation may reindex and introduce NaNs.\n"
            f"  Files: {', '.join(summary['coord_mismatch_files'][:5])} ..."
        )

    return summary


# ---------- 2) Global concatenation + duplicates/backward/irregular & boundary deltas ----------


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
    Concatenate native datetime64 time axes from all files, then check:
      - Global non-decreasing monotonicity (tolerant)
      - Duplicates across files
      - Backward steps
      - Irregular steps relative to auto-inferred global prevailing step
      - Boundary deltas between consecutive files (raw timedelta64)
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

    # Sort files by start time
    records_sorted = sorted(records, key=lambda r: (r["start"], r["file"]))

    # Concatenate times
    all_times = np.concatenate([r["time_vals"] for r in records_sorted])
    n_total = all_times.size

    # Global diffs
    diffs = np.diff(all_times) if n_total > 1 else np.array([], dtype="timedelta64[ns]")
    is_monotonic_global = bool(np.all(diffs >= DT_TOL)) if diffs.size else True

    # Duplicates across the full series
    duplicates_total = int(n_total) - int(len(pd.Index(all_times).unique()))

    # Backward steps
    backward_total = int(np.count_nonzero(diffs < np.timedelta64(0, "ns")))

    # Prevailing step & irregulars
    prevailing = _infer_prevailing_step(diffs)
    irregular_total = (
        int(np.count_nonzero((diffs > np.timedelta64(0, "ns")) & (diffs != prevailing)))
        if prevailing is not None
        else 0
    )

    # Boundary deltas between files
    boundary_rows = []
    for i in range(len(records_sorted) - 1):
        curr = records_sorted[i]
        nxt = records_sorted[i + 1]
        delta = nxt["start"] - curr["end"]  # timedelta64
        relation = (
            "overlap_or_duplicate"
            if delta <= np.timedelta64(0, "ns")
            else "adjacent_to_prevailing"
            if (prevailing is not None and delta == prevailing)
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

    # Optional CSV outputs
    if write_csv_dir:
        out_dir = Path(write_csv_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Boundaries file (handy for eyeballing adjacency)
        pd.DataFrame(boundary_rows).to_csv(
            out_dir / "time_concat_boundaries.csv", index=False
        )
        # Global summary
        pd.DataFrame([summary]).drop(columns=["boundary_deltas"]).to_csv(
            out_dir / "time_concat_global_summary.csv", index=False
        )
        print(
            f"  ↳ wrote concatenation boundaries: {out_dir / 'time_concat_boundaries.csv'}"
        )
        print(f"  ↳ wrote global summary: {out_dir / 'time_concat_global_summary.csv'}")

    # Console summary
    print("\n[GLOBAL] Concatenated time axis check:")
    print(f"  Files concatenated (sorted): {len(records_sorted)}")
    print(
        f"  Total steps: {n_total} | Start: {summary['start']} | End: {summary['end']}"
    )
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
        print(
            "[OK]   Concatenation yields a clean, consistent series with a single prevailing step."
        )
    else:
        print(
            "[WARN] Concatenation has issues. See metrics above and boundaries CSV (if written)."
        )

    return summary


def scan_time_consistency(
    data_dir: str,
    file_glob: str = "*.nc",
    time_dim_name: str = "time",
    station_dim_name: str = "station",
    cmip6_dim_name: str = "cmip6",
    expected_freq: str = "D",  # 'D' for daily, 'H' for hourly
    decode_cf: bool = True,
    mask_and_scale: bool = False,  # mask/scale not needed for time-only checks
    engine: Optional[str] = None,
    write_csv_dir: Optional[str] = None,
    print_top_boundary_issues: int = 10,
) -> Dict[str, Any]:
    """
    Checks time consistency within each monthly file and across files.
    Returns a summary dict; optionally writes per-file CSV and prints concise warnings.

    - Within-file: monotonicity (tolerant), duplicates, gaps relative to expected frequency.
    - Cross-file: gap/overlap between consecutive files.
    - Coordinate consistency: checks station/cmip6 coordinate equality across files.
    """
    files = sorted(Path(data_dir).glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No files match {file_glob} in {data_dir}")

    expected_step_days = _expected_step_days(expected_freq)

    per_file = []
    all_boundaries = []  # (file, start_num, end_num, calendar, start_dt, end_dt)
    station_ref = None
    cmip6_ref = None
    coord_mismatch_files = []

    for f in files:
        open_kwargs = dict(decode_cf=decode_cf, mask_and_scale=mask_and_scale)
        if engine:
            open_kwargs["engine"] = engine

        try:
            with xr.open_dataset(f, **open_kwargs) as ds:
                # Ensure time variable exists
                if time_dim_name not in ds.dims and time_dim_name not in ds.coords:
                    print(f"[SKIP] {f.name}: missing time dim/coord '{time_dim_name}'")
                    continue

                # Read time coordinate (small—safe to materialize)
                time_var = ds[time_dim_name]
                t = time_var.values
                cal = time_var.attrs.get("calendar", "standard")
                units = time_var.attrs.get("units", None)
                nums = _to_num_days(t, calendar=cal, units=units)
                n = len(nums)

                # Basic markers
                start_dt = t[0] if n > 0 else None
                end_dt = t[-1] if n > 0 else None
                start_num = nums[0] if n > 0 else np.nan
                end_num = nums[-1] if n > 0 else np.nan

                # Within-file checks (tolerant, non-decreasing)
                diffs = (
                    np.diff(np.round(nums, 10)) if n > 1 else np.array([], dtype=float)
                )
                is_monotonic = bool(np.all(diffs >= -TOL)) if n > 1 else True

                # duplicates: repeated timestamps (tolerant rounding)
                dup_within = (
                    int(n) - int(len(pd.Index(np.round(nums, 10)).unique()))
                    if n > 0
                    else 0
                )

                gaps_mask = diffs > (expected_step_days + TOL)
                overlaps_mask = diffs < (
                    expected_step_days - TOL
                )  # includes zero/negatives
                num_gaps = int(np.count_nonzero(gaps_mask))
                num_overlaps = int(np.count_nonzero(overlaps_mask))

                # Expected count (daily/hourly) from [start, end] inclusive (tolerant)
                if n > 0 and not np.isnan(start_num) and not np.isnan(end_num):
                    span_days = end_num - start_num
                    expected_count = int(round(span_days / expected_step_days)) + 1
                else:
                    expected_count = 0

                per_file.append(
                    {
                        "file": f.name,
                        "n_steps": n,
                        "expected_count_from_span": expected_count,
                        "monotonic": is_monotonic,
                        "duplicates_within": dup_within,
                        "gaps_within": num_gaps,
                        "overlaps_within": num_overlaps,
                        "calendar": cal,
                        "start": _to_printable(start_dt),
                        "end": _to_printable(end_dt),
                    }
                )

                all_boundaries.append(
                    (f.name, start_num, end_num, cal, start_dt, end_dt)
                )

                # Coordinate consistency checks (exact equality to prevent reindexing)
                if station_dim_name in ds.dims or station_dim_name in ds.coords:
                    stations = ds[station_dim_name].values
                    if station_ref is None:
                        station_ref = stations
                    else:
                        if stations.shape != station_ref.shape or not np.array_equal(
                            stations, station_ref
                        ):
                            coord_mismatch_files.append(f.name)

                if cmip6_dim_name in ds.dims or cmip6_dim_name in ds.coords:
                    cmip6 = ds[cmip6_dim_name].values
                    if cmip6_ref is None:
                        cmip6_ref = cmip6
                    else:
                        if cmip6.shape != cmip6_ref.shape or not np.array_equal(
                            cmip6, cmip6_ref
                        ):
                            coord_mismatch_files.append(f.name)

        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")

    # Cross-file boundary checks
    cross_issues = []
    all_boundaries_sorted = sorted(
        all_boundaries, key=lambda r: (np.nan_to_num(r[1], nan=-1e99), r[0])
    )

    for i in range(len(all_boundaries_sorted) - 1):
        f_curr, s_num, e_num, cal_curr, s_dt, e_dt = all_boundaries_sorted[i]
        f_next, s2_num, e2_num, cal_next, s2_dt, e2_dt = all_boundaries_sorted[i + 1]

        # Calendar mismatch
        if cal_curr != cal_next:
            cross_issues.append(
                {
                    "pair": (f_curr, f_next),
                    "issue": "calendar_mismatch",
                    "calendar_curr": cal_curr,
                    "calendar_next": cal_next,
                }
            )

        # Expect end-of-current + step == start-of-next (tolerant)
        gap = (s2_num - e_num) - expected_step_days
        if gap > TOL:
            cross_issues.append(
                {
                    "pair": (f_curr, f_next),
                    "issue": "gap_between_files",
                    "gap_days": gap,
                    "end_curr": _to_printable(e_dt),
                    "start_next": _to_printable(s2_dt),
                }
            )
        elif gap < -TOL:
            cross_issues.append(
                {
                    "pair": (f_curr, f_next),
                    "issue": "overlap_between_files",
                    "overlap_days": -gap,
                    "end_curr": _to_printable(e_dt),
                    "start_next": _to_printable(s2_dt),
                }
            )
        else:
            # perfect adjacency
            pass

    # Optional write-out
    summary = {
        "per_file": per_file,
        "cross_file_issues": cross_issues,
        "coord_mismatch_files": sorted(set(coord_mismatch_files)),
    }

    if write_csv_dir:
        out_dir = Path(write_csv_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        df_per = pd.DataFrame(per_file)
        df_per.to_csv(out_dir / "time_scan_per_file.csv", index=False)

        df_cross = pd.DataFrame(cross_issues)
        df_cross.to_csv(out_dir / "time_scan_cross_file.csv", index=False)

        if summary["coord_mismatch_files"]:
            pd.DataFrame({"file": summary["coord_mismatch_files"]}).to_csv(
                out_dir / "coord_mismatch_files.csv", index=False
            )

        print(f"  ↳ wrote time scan: {out_dir / 'time_scan_per_file.csv'}")
        print(f"  ↳ wrote cross-file issues: {out_dir / 'time_scan_cross_file.csv'}")
        if summary["coord_mismatch_files"]:
            print(
                f"  ↳ wrote coord mismatch list: {out_dir / 'coord_mismatch_files.csv'}"
            )

    # Console summaries
    for rec in per_file:
        if (
            (rec["duplicates_within"] > 0)
            or (rec["gaps_within"] > 0)
            or (rec["overlaps_within"] > 0)
            or (not rec["monotonic"])
        ):
            print(
                f"[WARN] {rec['file']}: time issues — "
                f"monotonic={rec['monotonic']}, dup={rec['duplicates_within']}, "
                f"gaps={rec['gaps_within']}, overlaps={rec['overlaps_within']}; "
                f"n={rec['n_steps']}, expected_from_span={rec['expected_count_from_span']}, "
                f"calendar={rec['calendar']}"
            )
        else:
            print(
                f"[OK]   {rec['file']}: time looks clean (n={rec['n_steps']}, calendar={rec['calendar']})"
            )

    if cross_issues:
        print("Cross-file adjacency issues (top few):")
        for row in cross_issues[:print_top_boundary_issues]:
            qty = row.get("gap_days") or row.get("overlap_days")
            print(f"  {row['pair'][0]} → {row['pair'][1]}: {row['issue']} ({qty})")
    else:
        print("[OK]   Cross-file time adjacency: perfect (no gaps/overlaps).")

    if summary["coord_mismatch_files"]:
        print(
            f"[WARN] {len(summary['coord_mismatch_files'])} files have coordinate mismatches (station/cmip6). "
            f"Concatenation may reindex and introduce NaNs.\n"
            f"  Files: {', '.join(summary['coord_mismatch_files'][:5])} ..."
        )

    return summary


def scan_files_for_nans(
    data_dir: str,
    var_name: str,
    file_glob: str = "*.nc",
    station_dim_name: str = "station",
    cmip6_dim_name: str = "cmip6",
    time_dim_name: str = "time",
    chunks_time: int = 365,  # adjust as needed (e.g., daily -> yearly chunks)
    print_top: int = 10,  # show top-N (station, cmip6) by NaN fraction
    write_csv_dir: Optional[str] = None,  # optional: write per-file CSV summaries
) -> None:
    """
    Iterate NetCDF files one-by-one and print a summary ONLY when NaNs are detected.
    Uses xarray/dask lazy computation—does not load entire files to RAM.
    """
    files = sorted(Path(data_dir).glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No files match {file_glob} in {data_dir}")

    for f in files:
        try:
            with xr.open_dataset(
                f,
                decode_cf=True,  # keep cf decoding true for reliable time handling
                chunks={time_dim_name: chunks_time} if chunks_time else None,
            ) as ds:
                if var_name not in ds:
                    print(
                        f"[SKIP] {f.name}: variable '{var_name}' not found. Available: {list(ds.data_vars)}"
                    )
                    continue

                da = ds[var_name]

                # Robust dimension mapping (lowercase-insensitive)
                dims_lower_to_orig = {d.lower(): d for d in da.dims}
                names_lower = {
                    "time": time_dim_name.lower(),
                    "station": station_dim_name.lower(),
                    "cmip6": cmip6_dim_name.lower(),
                }
                missing = [
                    k for k, v in names_lower.items() if v not in dims_lower_to_orig
                ]
                if missing:
                    print(f"[SKIP] {f.name}: missing dims {missing} in {da.dims}")
                    continue

                time_dim = dims_lower_to_orig[names_lower["time"]]
                station_dim = dims_lower_to_orig[names_lower["station"]]
                cmip6_dim = dims_lower_to_orig[names_lower["cmip6"]]

                # Canonical order
                da = da.transpose(time_dim, station_dim, cmip6_dim)
                T = int(da.sizes[time_dim])
                S = int(da.sizes[station_dim])
                C = int(da.sizes[cmip6_dim])
                total_steps_all = T * S * C

                # Quick detection (lazy): any NaN anywhere?
                has_nan = bool(da.isnull().any().compute())
                if not has_nan:
                    print(
                        f"[OK]   {f.name}: no NaNs (time={T}, station={S}, cmip6={C})"
                    )
                    continue

                # Compute per-(station, cmip6) NaN counts lazily
                M = da.isnull()  # boolean DataArray (lazy)
                nan_steps = M.sum(dim=time_dim)  # (station, cmip6)
                total_nan_steps = int(nan_steps.sum().compute())
                nan_fraction = nan_steps / T  # lazy ratio

                # Pairs affected
                affected_mask = (nan_steps > 0).compute()
                num_pairs_affected = int(affected_mask.sum().item())

                print(
                    f"[WARN] {f.name}: NaNs detected — "
                    f"{num_pairs_affected}/{S * C} pairs affected; "
                    f"total_nan_steps={total_nan_steps}/{total_steps_all} "
                    f"({total_nan_steps / total_steps_all:.2%})"
                )

                # Top-N pairs by NaN fraction (optional)
                if print_top and num_pairs_affected > 0:
                    nan_fraction_comp = nan_fraction.compute()
                    top_df = (
                        nan_fraction_comp.to_dataset(name="nan_fraction")
                        .to_dataframe()
                        .reset_index()
                        .sort_values("nan_fraction", ascending=False)
                        .head(print_top)
                    )
                    # Pretty print (limit to 4 decimals)
                    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
                    print("Top pairs by NaN fraction in this file:")
                    print(top_df.to_string(index=False))
                    pd.reset_option("display.float_format")

                # Optional: write per-file CSV
                if write_csv_dir:
                    out_dir = Path(write_csv_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    nan_steps_comp = nan_steps.compute()
                    nan_fraction_comp = nan_fraction.compute()
                    ds_out = xr.Dataset(
                        {"nan_steps": nan_steps_comp, "nan_fraction": nan_fraction_comp}
                    )
                    df_out = ds_out.to_dataframe().reset_index()
                    df_out["total_steps"] = T
                    df_out = df_out.sort_values([station_dim, cmip6_dim])
                    out_csv = out_dir / f"{Path(f).stem}_nan_summary.csv"
                    df_out.to_csv(out_csv, index=False)
                    print(f"  ↳ wrote per-file summary: {out_csv}")

        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")


if __name__ == "__main__":
    scan_files_for_nans(
        data_dir=r"D:\Kai\DFM\cdf_diff\000",
        var_name="cmip_diff",
        file_glob="ERA5wl_Diff_*.nc",
        station_dim_name="station",
        cmip6_dim_name="cmip6",
        time_dim_name="time",
        chunks_time=365,  # tune for your data
        print_top=10,
        write_csv_dir=r"D:\Kai\DFM\cdf_diff\nan_scans",  # or None
    )

    # 2) Time consistency scan (new)
    scan_time_consistency(
        data_dir=r"D:\Kai\DFM\cdf_diff\000",
        file_glob="ERA5wl_Diff_*.nc",
        time_dim_name="time",
        station_dim_name="station",
        cmip6_dim_name="cmip6",
        decode_cf=True,
        mask_and_scale=False,
        engine=None,
        write_csv_dir=r"D:\Kai\DFM\cdf_diff\nan_scans",
    )

    # 3) Global concatenated time axis + overlaps/gaps/duplicates/sampling
    concat_time_and_global_checks(
        data_dir=r"D:\Kai\DFM\cdf_diff\000",
        file_glob="ERA5wl_Diff_*.nc",
        time_dim_name="time",
        decode_cf=True,
        mask_and_scale=False,
        engine=None,
        write_csv_dir=r"D:\Kai\DFM\cdf_diff\nan_scans",
    )

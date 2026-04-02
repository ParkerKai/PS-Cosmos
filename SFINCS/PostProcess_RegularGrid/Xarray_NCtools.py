import xarray as xr
from typing import Optional
import numpy as np


def check_nc_file(
    path,
    required_vars=None,
    required_coords=None,
    check_time=True,
    sample_data=False,
):
    """
    Returns (ok, issues, summary) for a single NetCDF file.
    - ok: bool
    - issues: list of strings describing problems
    - summary: dict with lightweight metadata (dims, vars, coords, dtypes)
    """
    issues = []
    summary = {"path": path}

    try:
        # Avoid full load; just open the dataset lazily.
        # netcdf4 engine is robust for classic/NETCDF4 files; can switch to 'scipy' if needed.
        ds = xr.open_dataset(
            path, decode_cf=True, mask_and_scale=True, engine="netcdf4"
        )
    except Exception as e:
        issues.append(f"OPEN_ERROR: {type(e).__name__}: {e}")
        return False, issues, summary

    # Collect summary info
    summary["dims"] = dict(ds.sizes)
    summary["coords"] = list(ds.coords)
    summary["vars"] = [v for v in ds.data_vars]
    summary["global_attrs"] = dict(ds.attrs)

    # Dtypes snapshot (no compute)
    dtypes = {}
    for v in ds.variables:
        try:
            dtypes[v] = str(ds[v].dtype)
        except Exception:
            dtypes[v] = "UNKNOWN"
    summary["dtypes"] = dtypes

    # 1) Dimensions non-empty
    for d, n in ds.sizes.items():
        if n <= 0:
            issues.append(f"DIM_EMPTY: dimension {d} has size {n}")

    # 2) Required variables/coords
    if required_vars:
        for v in required_vars:
            if v not in ds.variables:
                issues.append(f"MISSING_VAR: {v}")
    if required_coords:
        for c in required_coords:
            if c not in ds.coords:
                issues.append(f"MISSING_COORD: {c}")

    # 3) Time checks
    if check_time:
        time_name = None
        # Common time names
        for nm in ["time", "t", "Time"]:
            if nm in ds.coords:
                time_name = nm
                break
        if time_name:
            try:
                t = ds[time_name].values  # NumPy array (may be datetime64)
                if np.isnan(t).any():
                    issues.append(f"TIME_NAN: {time_name} contains NaNs")
                # Monotonic increasing check (allow equal for multi-file splits)
                if (
                    np.any(np.diff(t) < np.timedelta64(0, "ns"))
                    if np.issubdtype(t.dtype, np.datetime64)
                    else np.any(np.diff(t) < 0)
                ):
                    issues.append(
                        f"TIME_NON_MONOTONIC: {time_name} not strictly increasing"
                    )
            except Exception as e:
                issues.append(f"TIME_READ_ERROR: {time_name} {type(e).__name__}: {e}")

    # 4) Sample small data slices (optional, can be costly)
    if sample_data:
        try:
            for v in list(ds.data_vars)[:5]:  # sample up to 5 variables
                # Take a tiny slice to trigger decode/scale without loading full
                slc = {d: 0 for d in ds[v].dims}
                _ = ds[v].isel(**slc).values
        except Exception as e:
            issues.append(f"DATA_SAMPLE_ERROR: {type(e).__name__}: {e}")

    # 5) Minimal attribute & encoding sanity
    for v in ds.variables:
        enc = ds[v].encoding
        # Known source of conflicts in open_mfdataset: variable attrs differ across files
        # We just record presence; comparison across files happens later.
        if "_FillValue" in enc and enc["_FillValue"] is None:
            issues.append(f"ENCODING_WARN: {v} has null _FillValue")

    # Clean up file handle
    ds.close()

    ok = len(issues) == 0
    return ok, issues, summary


def batch_check_nc_files(
    files,
    required_vars=None,
    required_coords=None,
    check_time=True,
    sample_data=False,
    compare_coords=(
        "time",
        "lat",
        "lon",
    ),  # coordinates to compare shape/attrs across files
):
    """
    Checks files and returns:
    - good_files: list of valid file paths
    - report: list of per-file records (path, ok, issues)
    - schema_ref: snapshot from first good file (dims, dtypes, coord shapes)
    """
    report = []
    good_files = []
    schema_ref = {"dims": None, "dtypes": None, "coord_shapes": {}}

    # First pass: individual checks
    per_file_summaries = {}
    for p in files:
        ok, issues, summary = check_nc_file(
            p,
            required_vars=required_vars,
            required_coords=required_coords,
            check_time=check_time,
            sample_data=sample_data,
        )
        report.append({"path": p, "ok": ok, "issues": issues})
        per_file_summaries[p] = summary
        if ok:
            good_files.append(p)

    # Establish schema reference from first good file
    if good_files:
        ref = per_file_summaries[good_files[0]]
        schema_ref["dims"] = ref["dims"]
        schema_ref["dtypes"] = ref["dtypes"]
        # Record coordinate shapes
        coord_shapes = {}
        for c in compare_coords:
            if c in ref["dims"]:
                coord_shapes[c] = ref["dims"][c]
            elif c in ref["coords"]:
                # Some coords are 1D not in dims; try len
                coord_shapes[c] = None
        schema_ref["coord_shapes"] = coord_shapes

    # Second pass: cross-file schema/coord consistency
    for p in good_files:
        ref = schema_ref
        summ = per_file_summaries[p]

        # Compare key coordinate dimension sizes (lightweight)
        for c, sz in ref["coord_shapes"].items():
            if c in summ["dims"] and sz is not None:
                if summ["dims"][c] != sz:
                    # Not necessarily fatal (concat may differ), but flag it
                    for r in report:
                        if r["path"] == p:
                            r["issues"].append(
                                f"COORD_SIZE_DIFF: {c}={summ['dims'][c]} (ref {sz})"
                            )
                            break

        # Compare dtypes for variables present in both
        ref_dtypes = ref["dtypes"]
        for v, dt in summ["dtypes"].items():
            if v in ref_dtypes and dt != ref_dtypes[v]:
                for r in report:
                    if r["path"] == p:
                        r["issues"].append(
                            f"DTYPE_DIFF: var {v} dtype {dt} (ref {ref_dtypes[v]})"
                        )
                        break

    # Recompute good_files to exclude anything with issues after cross-check
    final_good = [r["path"] for r in report if r["ok"] and len(r["issues"]) == 0]

    return final_good, report, schema_ref


def _detect_time_coord(
    ds: xr.Dataset, preferred: Optional[str] = None
) -> Optional[str]:
    """Return the name of the time coordinate if found, else None."""

    # Candidate names for the time coordinate; we'll auto-detect the first that exists.
    TIME_CANDIDATES: List[str] = ["time", "Time", "t", "datetime", "date", "timemax"]

    if preferred and preferred in ds.coords:
        return preferred
    for name in TIME_CANDIDATES:
        if name in ds.coords:
            return name
    return None


def ensure_unique_sorted_time(ds, time_name: Optional[str] = None, keep="first"):
    """
    Drop duplicate time stamps and sort by time.
    keep: 'first' or 'last' — which duplicate to keep.
    time_name : str, optional
        Explicit time coordinate name. If None, will auto-detect.
    """
    tname = _detect_time_coord(ds, preferred=time_name)  # noqa: F821
    if tname is None:
        return ds  # No time coordinate present; nothing to trim.

    # Convert time coord to pandas datetime for reliable duplicate detection
    tvals = ds[time_name].values
    # pd.to_datetime handles datetime64/cftime->numpy conversion reasonably; if cftime persists, decode_cf first.
    pdt = pd.to_datetime(tvals)
    dup_mask = pd.Series(pdt).duplicated(keep=keep).to_numpy()  # True where duplicate
    if dup_mask.any():
        ds = ds.isel({time_name: ~dup_mask})
    ds = ds.sortby(time_name)
    return ds

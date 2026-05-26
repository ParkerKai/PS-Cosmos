"""
Unified SFINCS post-processing driver for PS-CoSMoS.

Replaces the previous five-part pipeline (part1..part5_*.py) with a single,
top-to-bottom script. Each step is toggled at the top, and outputs are
checked for existence so re-runs are cheap.

Pipeline:
  1. Water-year aggregation of SFINCS NetCDF outputs -> per-cell annual maxima
     of zsmax (+ extras like qmax/tmax at the peak time per cell per year).
  2. Cell-level extreme value analysis (Weibull / GEV / POT) -> per-RP zsmax
     on the quadtree, plus event-matched extras.
  3. Build / reuse the DEM-pixel -> SFINCS-cell index COG.
  4. Downscale per-RP zsmax -> hmax + zsmax rasters via hydromt-sfincs.
  5. Remove disconnected flooding using sfincs.bnd boundary points.
  6. Map extras (qmax, tmax, ...) onto the DEM grid via the index COG.
  7. Derive velocity from (qmax, hmax) and bin depth + velocity into
     hazard categories.

All generic downscaling logic lives in `hydromt_sfincs.workflows.downscaling`.
PS-CoSMoS-specific logic lives in `ps_cosmos_postprocess_helper.py`.

Run in the hydromt-sfincs-dev environment:
    conda run -n hydromt-sfincs-dev python downscale_sfincs_results.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import xarray as xr
import xugrid as xu

from hydromt_sfincs import SfincsModel
from hydromt_sfincs.workflows import downscaling

import ps_cosmos_postprocess_helper as helper

# Force unbuffered output so VS Code shows prints in real-time
sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# CONFIGURATION
# =============================================================================
# --- inputs ---
# Reference SFINCS model (provides the quadtree grid + boundary points).
# For multi-year aggregation, point `sfincs_dirs` at each water-year run dir.
sfincs_root     = Path(r"p:\CoSMoS_PS\10_Kitsap\20260526_improve_scripts\2026-05-17-Kitsap")
bnd_file        = sfincs_root / "sfincs.bnd"

# Discover one SFINCS run directory per water year.
sfincs_dirs = sorted(
    p for p in sfincs_root.iterdir()
    if p.is_dir() and p.name.isdigit() and len(p.name) == 4
    and (p / "sfincs_map.nc").exists()
)
print(f"Found {len(sfincs_dirs)} water-year runs: "
    f"{sfincs_dirs[0].name}..{sfincs_dirs[-1].name}")

# DEM + index COG
dem_res         = 1                                # DEM resolution [m]
dem_file        = Path(r"p:\CoSMoS_PS\10_Kitsap\20260526_improve_scripts\kitsap_DEM\KitsapCo_CoNED_DEMmods_v1.tif")
indices_fn      = sfincs_root / "downscaled_results_1m" / f"indices_{dem_res}m.tif"

# Output
output_dir      = sfincs_root / "downscaled_results_1m"
domain_stem     = f"_{dem_res}m"

# --- aggregation + EVA ---
# "annual_max" picks one peak per cell per year (suitable for Weibull / GEV /
# POT-with-sidecar). "all_maxima" keeps every sub-yearly timemax slice
# concatenated across years and is POT-only.
aggregation_mode  = "annual_max"                # "annual_max" | "all_maxima"
eva_method        = "weibull"                   # "weibull" | "gev" | "pot"
return_periods    = [1, 2, 5, 10, 20, 50]
extra_vars        = ["qmax"]                    # e.g. ["qmax", "tmax", "tmax_zs"]

# POT-only knobs
pot_target_per_yr = 5
pot_decluster     = "72h"

# GEV-only knob
gev_min_years     = 5

# --- downscaling ---
downscale_method  = "bilinear"                  # "raw" | "constant" | "bilinear"
dilation          = 0.5                         # None to disable; only used for bilinear
hmin              = 0.02                        # wet threshold [m], used everywhere

# --- hazard binning ---
# Depth bins (lower edges, m). The categorical output also adds:
#   - "Below MHHW" wherever DEM < mhhw_elevation (tidally submerged)
#   - "Flood-prone Low-Lying" wherever the Step-5 connection raster == 2
#     (standing water that was disconnected from the boundary)
# See helper.bin_depth_with_overlays for the full code mapping.
depth_bins      = [0.0, 0.3048, 0.9144, 1.524]     # < 1ft / 1-3ft / 3-5ft / > 5ft
depth_labels    = ["< 1 ft", "1-3 ft", "3-5 ft", "> 5 ft"]
mhhw_elevation  = 2.748                            # MHHW [m, NAVD88]; Everett station 9447130, ~2.62 m. Verify against DEM datum.
qmax_bins       = [0.0, 0.3, 0.6, 1.0, 2.0]        # lower edges [m^2/s]
qmax_labels     = ["Low", "Medium", "High", "Very High", "Extreme"]

# --- step toggles ---
run_aggregation = True
run_eva         = True
run_downscale   = True
run_disconnect  = True
run_extras      = True
run_binning     = True

# Index COG block size (only used when (re)building the index COG)
NRMAX           = 2000

# =============================================================================
# Derived paths + sanity
# =============================================================================
output_dir.mkdir(parents=True, exist_ok=True)
if aggregation_mode == "annual_max":
    aggregated_fn = output_dir / "aggregated_annual.nc"
    timeseries_fn = output_dir / "aggregated_annual.timeseries.nc"
    # POT on annual-max mode needs the sidecar full timeseries
    keep_timeseries = (eva_method == "pot")
elif aggregation_mode == "all_maxima":
    aggregated_fn = output_dir / "aggregated_all.nc"
    timeseries_fn = aggregated_fn          # main file IS the timeseries
    keep_timeseries = False
else:
    raise ValueError(
        f"Unknown aggregation_mode={aggregation_mode!r}; "
        "use 'annual_max' or 'all_maxima'."
    )
eva_fn = output_dir / f"eva_RP_{eva_method}.nc"

# Provenance token appended to every output filename so runs with different
# (eva_method, aggregation_mode) coexist in the same output_dir without
# colliding. e.g. "weibull_annual", "gev_annual", "pot_all".
AGG_TOKEN = {"annual_max": "annual", "all_maxima": "all"}[aggregation_mode]
provenance_tag = f"{eva_method}_{AGG_TOKEN}"


def rp_tag(rp: float) -> str:
    return f"RP{int(round(rp)):03d}"


def hmax_path(rp: float) -> Path:
    return output_dir / f"hmax_{rp_tag(rp)}_{domain_stem}_{provenance_tag}.tif"


def zsmax_path(rp: float) -> Path:
    return output_dir / f"zsmax_{rp_tag(rp)}_{domain_stem}_{provenance_tag}.tif"


def hmax_masked_path(rp: float) -> Path:
    p = hmax_path(rp)
    return p.parent / (p.stem + "_masked" + p.suffix)


def zsmax_masked_path(rp: float) -> Path:
    p = zsmax_path(rp)
    return p.parent / (p.stem + "_masked" + p.suffix)


def connection_path(rp: float) -> Path:
    return output_dir / f"connection_{rp_tag(rp)}_{domain_stem}_{provenance_tag}.tif"


def extra_path(var: str, rp: float) -> Path:
    return output_dir / f"{var}_{rp_tag(rp)}_{domain_stem}_{provenance_tag}.tif"


def depth_bins_path(rp: float) -> Path:
    return output_dir / f"depth_bins_{rp_tag(rp)}_{domain_stem}_{provenance_tag}.tif"


def qmax_bins_path(rp: float) -> Path:
    return output_dir / f"qmax_bins_{rp_tag(rp)}_{domain_stem}_{provenance_tag}.tif"


# =============================================================================
# Step 1: Water-year aggregation
# =============================================================================
print(f"\n{'='*60}\nStep 1: water-year aggregation ({aggregation_mode})\n{'='*60}")
if run_aggregation:
    t0 = time.time()
    # In annual_max mode we still need to (re)write the sidecar timeseries
    # when POT is selected; in all_maxima mode the main file IS the timeseries.
    needs_sidecar = (
        aggregation_mode == "annual_max"
        and keep_timeseries
        and not timeseries_fn.exists()
    )
    if aggregated_fn.exists() and not needs_sidecar:
        print(f"  skip: {aggregated_fn.name} exists")
    else:
        helper.aggregate_water_year_maxima(
            sfincs_dirs=sfincs_dirs,
            extra_vars=extra_vars,
            output_fn=aggregated_fn,
            keep_timeseries=keep_timeseries,
            aggregation_mode=aggregation_mode,
        )
        print(f"  wrote: {aggregated_fn.name} ({time.time()-t0:.1f}s)")
else:
    print("  skipped by toggle")

ds_annual = xr.open_dataset(aggregated_fn)
ds_full = None
if eva_method == "pot":
    # In all_maxima mode, ds_annual IS the timeseries (timeseries_fn == aggregated_fn);
    # eva_apply will read it from the dataset directly, so ds_full can stay None.
    # In annual_max mode, the sidecar must exist.
    if aggregation_mode == "annual_max":
        if not timeseries_fn.exists():
            raise FileNotFoundError(
                f"POT EVA requested but {timeseries_fn} is missing. "
                "Re-run with run_aggregation=True so the full series is persisted."
            )
        ds_full = xr.open_dataset(timeseries_fn)

# Universal provenance tags stamped onto every output raster.
n_years_stamped = int(ds_annual.attrs.get("n_years", len(sfincs_dirs)))
base_tags = {
    "aggregation_mode": aggregation_mode,
    "eva_method": eva_method,
    "n_years": n_years_stamped,
    "hmin": hmin,
    "domain_stem": domain_stem,
    "produced_by": "downscale_sfincs_results.py",
}
if sfincs_dirs:
    base_tags["water_year_first"] = sfincs_dirs[0].name
    base_tags["water_year_last"]  = sfincs_dirs[-1].name


# =============================================================================
# Step 2: Extreme value analysis (Weibull / GEV / POT)
# =============================================================================
print(f"\n{'='*60}\nStep 2: EVA ({eva_method})\n{'='*60}")
if run_eva:
    t0 = time.time()
    if eva_fn.exists():
        print(f"  skip: {eva_fn.name} exists")
    else:
        ds_eva = helper.eva_apply(
            ds_annual,
            method=eva_method,
            return_periods=return_periods,
            ds_full_timeseries=ds_full,
            pot_target_per_year=pot_target_per_yr,
            pot_decluster=pot_decluster,
            gev_min_years=gev_min_years,
        )
        # event-matched extras (Weibull/GEV); POT extras are not implemented in v1
        if eva_method in ("weibull", "gev") and extra_vars:
            ds_extras_rp = helper.extras_at_rp_via_rank(
                ds_annual, return_periods=return_periods, extras=extra_vars,
            )
            for v in ds_extras_rp.data_vars:
                ds_eva[v] = ds_extras_rp[v]
        ds_eva.to_netcdf(eva_fn)
        print(f"  wrote: {eva_fn.name} ({time.time()-t0:.1f}s)")
else:
    print("  skipped by toggle")

ds_eva = xr.open_dataset(eva_fn)


# =============================================================================
# Step 3: Load SFINCS model + build (or reuse) index COG
# =============================================================================
print(f"\n{'='*60}\nStep 3: SFINCS model + index COG\n{'='*60}")
mod = SfincsModel(str(sfincs_root), mode="r")
print(f"  grid type: {mod.grid_type}")

if not indices_fn.exists():
    print(f"  building index COG: {indices_fn}")
    indices_fn.parent.mkdir(parents=True, exist_ok=True)
    downscaling.make_index_cog(mod, indices_fn, dem_file, nrmax=NRMAX)
    print("  index COG created")
else:
    print(f"  reuse: {indices_fn}")

# Build a UgridDataArray template for the quadtree topology (cell-center
# coords). The EVA outputs are plain xr (no ugrid sidecar after NetCDF
# round-trip), so bilinear downscaling would otherwise fail with `x
# dimension not found`. We re-attach the topology in Step 4 via
# template.copy(data=...).
# Note: in this multi-year layout `sfincs_map.nc` lives inside each year
# subdirectory, not at the SFINCS root - so we can't use mod.output.read().
# Open the first year's map file directly via xugrid instead.
if run_downscale:
    template_src = Path(sfincs_dirs[0]) / "sfincs_map.nc"
    print(f"  building topology template from {template_src}")
    _template_ds = xu.open_dataset(template_src)
    zsmax_template = _template_ds["zsmax"].isel(timemax=0).drop_vars(
        "timemax", errors="ignore"
    )


# =============================================================================
# Step 4: Downscale per-RP zsmax -> hmax + zsmax rasters
# =============================================================================
print(f"\n{'='*60}\nStep 4: downscale per RP ({downscale_method})\n{'='*60}")
if run_downscale:
    assert ds_eva["zsmax_rp"].sizes["nmesh2d_face"] == zsmax_template.sizes["nmesh2d_face"], (
        "face count mismatch between EVA dataset and SFINCS model; "
        "the aggregated NetCDF was likely built from a different quadtree."
    )
    for rp in return_periods:
        h_fn = hmax_path(rp)
        z_fn = zsmax_path(rp)
        if h_fn.exists() and z_fn.exists():
            print(f"  skip {rp_tag(rp)}: outputs exist")
            continue
        t0 = time.time()
        # Re-attach quadtree topology by copying values into the template.
        eva_face = ds_eva["zsmax_rp"].sel(rp=rp)
        zsmax_uda = zsmax_template.copy(
            data=eva_face.values.astype(zsmax_template.dtype)
        )
        zsmax_uda.name = "zsmax"
        kw = dict(
            zsmax       = zsmax_uda,
            dep         = str(dem_file),
            method      = downscale_method,
            hmin        = hmin,
            indices     = str(indices_fn),
            floodmap_fn = str(h_fn),
            zsmap_fn    = str(z_fn),
        )
        if downscale_method == "bilinear" and dilation:
            kw["dilation"] = dilation
        downscaling.downscale_floodmap(**kw)
        step_tags = {
            "return_period": int(round(rp)),
            "downscale_method": downscale_method,
            "dilation": dilation if downscale_method == "bilinear" else "",
        }
        helper.stamp_provenance(h_fn, **base_tags, **step_tags, variable="hmax")
        helper.stamp_provenance(z_fn, **base_tags, **step_tags, variable="zsmax")
        print(f"  {rp_tag(rp)}: wrote hmax + zsmax ({time.time()-t0:.1f}s)")
else:
    print("  skipped by toggle")


# =============================================================================
# Step 5: Remove disconnected flooding per RP
# =============================================================================
print(f"\n{'='*60}\nStep 5: disconnected-flooding removal\n{'='*60}")
if run_disconnect:
    for rp in return_periods:
        h_fn = hmax_path(rp)
        z_fn = zsmax_path(rp)
        h_m  = hmax_masked_path(rp)
        z_m  = zsmax_masked_path(rp)
        c_fn = connection_path(rp)
        if not h_fn.exists():
            print(f"  skip {rp_tag(rp)}: hmax not found")
            continue
        if h_m.exists() and z_m.exists() and c_fn.exists():
            print(f"  skip {rp_tag(rp)}: masked outputs exist")
            continue
        t0 = time.time()
        downscaling.remove_disconnected_flooding(
            depth_fn      = str(h_fn),
            bnd_fn        = str(bnd_file),
            hmin          = hmin,
            connection_fn = str(c_fn),
            output_fns    = {str(h_fn): str(h_m), str(z_fn): str(z_m)},
        )
        step5_tags = {
            "return_period": int(round(rp)),
            "produced_by_step": "remove_disconnected_flooding",
            "bnd_file": Path(bnd_file).name,
        }
        helper.stamp_provenance(h_m,  **base_tags, **step5_tags, variable="hmax_masked")
        helper.stamp_provenance(z_m,  **base_tags, **step5_tags, variable="zsmax_masked")
        helper.stamp_provenance(c_fn, **base_tags, **step5_tags, variable="connection_mask")
        print(f"  {rp_tag(rp)}: connection + masked rasters ({time.time()-t0:.1f}s)")
else:
    print("  skipped by toggle")


# =============================================================================
# Step 6: Map extras (qmax, tmax, ...) onto the DEM grid
# =============================================================================
print(f"\n{'='*60}\nStep 6: extras nearest-neighbour mapping\n{'='*60}")
if run_extras and extra_vars:
    for rp in return_periods:
        depth_for_mask = hmax_masked_path(rp) if hmax_masked_path(rp).exists() else hmax_path(rp)
        if not depth_for_mask.exists():
            print(f"  skip {rp_tag(rp)}: no hmax raster")
            continue
        for v in extra_vars:
            out_fn = extra_path(v, rp)
            if out_fn.exists():
                print(f"  skip {rp_tag(rp)}/{v}: exists")
                continue
            key = f"{v}_rp"
            if key not in ds_eva:
                print(f"  skip {rp_tag(rp)}/{v}: {key} not in EVA dataset")
                continue
            t0 = time.time()
            helper.map_quadtree_to_dem_nearest(
                da_face=ds_eva[key].sel(rp=rp),
                indices_fn=indices_fn,
                out_fn=out_fn,
                depth_mask_fn=depth_for_mask,
                hmin=hmin,
            )
            helper.stamp_provenance(
                out_fn, **base_tags,
                return_period=int(round(rp)),
                variable=v,
                source="quadtree_nearest_via_index_cog",
            )
            print(f"  {rp_tag(rp)}/{v}: wrote {out_fn.name} ({time.time()-t0:.1f}s)")
else:
    print("  skipped by toggle / no extras configured")


# =============================================================================
# Step 7: Hazard binning per RP (depth + qmax)
# =============================================================================
print(f"\n{'='*60}\nStep 7: hazard binning (depth + qmax)\n{'='*60}")
if run_binning:
    for rp in return_periods:
        h_for_bin = hmax_masked_path(rp) if hmax_masked_path(rp).exists() else hmax_path(rp)
        if not h_for_bin.exists():
            print(f"  skip {rp_tag(rp)}: no hmax raster")
            continue

        # depth bins (composite: Below MHHW + 4 depth bins + Flood-prone Low-Lying)
        d_bins_fn = depth_bins_path(rp)
        c_fn = connection_path(rp)
        if d_bins_fn.exists():
            print(f"  skip {rp_tag(rp)}/depth_bins: exists")
        elif not c_fn.exists():
            print(f"  skip {rp_tag(rp)}/depth_bins: connection raster not found")
        else:
            t0 = time.time()
            helper.bin_depth_with_overlays(
                hmax_masked_fn=h_for_bin,
                connection_fn=c_fn,
                dem_fn=dem_file,
                bin_edges=depth_bins,
                bin_labels=depth_labels,
                out_fn=d_bins_fn,
                mhhw_elevation=mhhw_elevation,
            )
            helper.stamp_provenance(
                d_bins_fn, **base_tags,
                return_period=int(round(rp)),
                variable="depth_bins_composite",
                source_depth=Path(h_for_bin).name,
                source_connection=Path(c_fn).name,
                source_dem=Path(dem_file).name,
                mhhw_elevation=mhhw_elevation if mhhw_elevation is not None else "",
            )
            print(f"  {rp_tag(rp)}/depth_bins: {d_bins_fn.name} ({time.time()-t0:.1f}s)")

        # qmax bins (depth-velocity product, m^2/s) - bin the raster from Step 6
        q_fn = extra_path("qmax", rp)
        if not q_fn.exists():
            print(f"  skip {rp_tag(rp)}/qmax_bins: qmax raster not available")
            continue
        q_bins_fn = qmax_bins_path(rp)
        if q_bins_fn.exists():
            print(f"  skip {rp_tag(rp)}/qmax_bins: exists")
        else:
            t0 = time.time()
            helper.bin_raster(q_fn, qmax_bins, qmax_labels, q_bins_fn)
            helper.stamp_provenance(
                q_bins_fn, **base_tags,
                return_period=int(round(rp)),
                variable="qmax_bins",
                source=Path(q_fn).name,
                units="m2/s",
            )
            print(f"  {rp_tag(rp)}/qmax_bins: {q_bins_fn.name} ({time.time()-t0:.1f}s)")
else:
    print("  skipped by toggle")


print(f"\n{'='*60}\nAll steps complete: outputs in {output_dir}\n{'='*60}")

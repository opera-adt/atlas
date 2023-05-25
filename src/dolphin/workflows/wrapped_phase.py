from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from dolphin import io, ps, stack, utils
from dolphin._background import NvidiaMemoryWatcher
from dolphin._log import get_log, log_runtime
from dolphin.interferogram import Network

from . import _utils, sequential, single
from .config import Workflow


@log_runtime
def run(cfg: Workflow, debug: bool = False) -> tuple[list[Path], Path, Path]:
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    cfg : Workflow
        [`Workflow`][dolphin.workflows.config.Workflow] object with workflow parameters
    debug : bool, optional
        Enable debug logging, by default False.

    Returns
    -------
    list[Path]
        list of Paths to virtual interferograms created.
    Path
        Path the final compressed SLC file created.
    Path
        Path to temporal correlation file created.
        In the case of a single phase linking step, this is the one tcorr file.
        In the case of sequential phase linking, this is the average tcorr file.
    """
    logger = get_log(debug=debug)

    input_file_list = cfg.cslc_file_list
    if not input_file_list:
        raise ValueError("No input files found")

    # #############################################
    # Make a VRT pointing to the input SLC files
    # #############################################
    subdataset = cfg.input_options.subdataset
    vrt_stack = stack.VRTStack(
        input_file_list,
        subdataset=subdataset,
        outfile=cfg.scratch_directory / "slc_stack.vrt",
    )

    # Make the nodata mask from the polygons, if we're using OPERA CSLCs
    try:
        nodata_mask_file = cfg.scratch_directory / "nodata_mask.tif"
        _utils.make_nodata_mask(
            vrt_stack.file_list, out_file=nodata_mask_file, buffer_pixels=2000
        )
    except Exception as e:
        logger.warning(f"Could not make nodata mask: {e}")
        nodata_mask_file = None

    # ###############
    # PS selection
    # ###############
    ps_output = cfg.ps_options._output_file
    if ps_output.exists():
        logger.info(f"Skipping making existing PS file {ps_output}")
    else:
        logger.info(f"Creating persistent scatterer file {ps_output}")
        try:
            existing_amp: Optional[Path] = cfg.amplitude_mean_files[0]
            existing_disp: Optional[Path] = cfg.amplitude_dispersion_files[0]
        except IndexError:
            existing_amp = existing_disp = None

        ps.create_ps(
            slc_vrt_file=vrt_stack.outfile,
            output_file=ps_output,
            output_amp_mean_file=cfg.ps_options._amp_mean_file,
            output_amp_dispersion_file=cfg.ps_options._amp_dispersion_file,
            amp_dispersion_threshold=cfg.ps_options.amp_dispersion_threshold,
            existing_amp_dispersion_file=existing_disp,
            existing_amp_mean_file=existing_amp,
            block_size_gb=cfg.worker_settings.block_size_gb,
        )

    # TODO: Need a good way to store the nslc attribute in the PS file...
    # If we pre-compute it from some big stack, we need to use that for SHP
    # finding, not use the size of `slc_vrt_file`

    # #########################
    # phase linking/EVD step
    # #########################
    pl_path = cfg.phase_linking._directory

    phase_linked_slcs = list(pl_path.glob("2*.tif"))
    if len(phase_linked_slcs) > 0:
        logger.info(f"Skipping EVD step, {len(phase_linked_slcs)} files already exist")
        comp_slc_file = next(pl_path.glob("compressed*tif"))
        tcorr_file = next(pl_path.glob("tcorr*tif"))
    else:
        logger.info(f"Running sequential EMI step in {pl_path}")
        if utils.gpu_is_available():  # Track the GPU mem usage if we're using it
            watcher = NvidiaMemoryWatcher(log_file=pl_path / "nvidia_memory.log")
        else:
            watcher = None

        if cfg.workflow_name == "single":
            phase_linked_slcs, comp_slc_file, tcorr_file = (
                single.run_wrapped_phase_single(
                    slc_vrt_file=vrt_stack.outfile,
                    output_folder=pl_path,
                    half_window=cfg.phase_linking.half_window.dict(),
                    strides=cfg.output_options.strides,
                    reference_idx=0,
                    beta=cfg.phase_linking.beta,
                    mask_file=nodata_mask_file,
                    ps_mask_file=ps_output,
                    amp_mean_file=cfg.ps_options._amp_mean_file,
                    amp_dispersion_file=cfg.ps_options._amp_dispersion_file,
                    shp_method=cfg.phase_linking.shp_method,
                    shp_alpha=cfg.phase_linking.shp_alpha,
                    shp_nslc=None,
                    max_bytes=cfg.worker_settings.block_size_gb * 1e9,
                    n_workers=cfg.worker_settings.n_workers,
                    gpu_enabled=cfg.worker_settings.gpu_enabled,
                )
            )
        else:
            phase_linked_slcs, comp_slcs, tcorr_file = (
                sequential.run_wrapped_phase_sequential(
                    slc_vrt_file=vrt_stack.outfile,
                    output_folder=pl_path,
                    half_window=cfg.phase_linking.half_window.dict(),
                    strides=cfg.output_options.strides,
                    beta=cfg.phase_linking.beta,
                    ministack_size=cfg.phase_linking.ministack_size,
                    mask_file=nodata_mask_file,
                    ps_mask_file=ps_output,
                    amp_mean_file=cfg.ps_options._amp_mean_file,
                    amp_dispersion_file=cfg.ps_options._amp_dispersion_file,
                    shp_method=cfg.phase_linking.shp_method,
                    shp_alpha=cfg.phase_linking.shp_alpha,
                    shp_nslc=None,
                    max_bytes=cfg.worker_settings.block_size_gb * 1e9,
                    n_workers=cfg.worker_settings.n_workers,
                    gpu_enabled=cfg.worker_settings.gpu_enabled,
                )
            )
            comp_slc_file = comp_slcs[-1]

        if watcher:
            watcher.notify_finished()

    # ###################################################
    # Form interferograms from estimated wrapped phase
    # ###################################################
    ifg_dir = cfg.interferogram_network._directory
    existing_ifgs = list(ifg_dir.glob("*.int.*"))
    if len(existing_ifgs) > 0:
        logger.info(f"Skipping interferogram step, {len(existing_ifgs)} exists")
    else:
        logger.info(
            f"Creating virtual interferograms from {len(phase_linked_slcs)} files"
        )
        # if Path(vrt_stack.file_list[0]).name.startswith("compressed"):
        if cfg.workflow_name == "single":
            # With a single update, whether or not we have compressed SLCs,
            # the phase linking results will be referenced to the first date.
            # TODO: how to handle the multiple interferogram case? We'll still
            # want to make a network.
            if cfg.interferogram_network.indexes is None:
                raise NotImplementedError(
                    "Only currently supporting manual interferogram network indexes for"
                    " single update."
                )
            idxs = cfg.interferogram_network.indexes
            if len(idxs) > 1:
                raise NotImplementedError(
                    "Multiple interferograms are not supported with single update"
                )
            if idxs[0] != (0, -1):
                raise NotImplementedError(
                    "Only currently supporting Interferogram network indexes (0, -1)"
                    " for single update"
                )
            ref_idx, sec_idx = idxs[0]
            file1, file2 = vrt_stack.file_list[ref_idx], vrt_stack.file_list[sec_idx]
            date1, date2 = utils.get_dates(file1)[0], utils.get_dates(file2)[0]
            # We're just copying, so get the extension of the file to copy
            to_copy = phase_linked_slcs[sec_idx]
            suffix = utils.full_suffix(to_copy)
            ifg_name = ifg_dir / (io._format_date_pair(date1, date2) + suffix)
            shutil.copyfile(to_copy, ifg_name)
            ifg_file_list = [ifg_name]  # return just the one as the "network"
        else:
            network = Network(
                slc_list=phase_linked_slcs,
                reference_idx=cfg.interferogram_network.reference_idx,
                max_bandwidth=cfg.interferogram_network.max_bandwidth,
                max_temporal_baseline=cfg.interferogram_network.max_temporal_baseline,
                indexes=cfg.interferogram_network.indexes,
                outdir=ifg_dir,
            )
            if len(network) == 0:
                raise ValueError("No interferograms were created")
            ifg_file_list = [ifg.path for ifg in network.ifg_list]

    return ifg_file_list, comp_slc_file, tcorr_file

from pathlib import Path
from typing import List, Tuple

from dolphin import ps, stack
from dolphin._background import NvidiaMemoryWatcher
from dolphin._log import get_log, log_runtime
from dolphin.interferogram import Network, VRTInterferogram
from dolphin.utils import gpu_is_available

from . import sequential, single
from .config import Workflow


@log_runtime
def run(
    cfg: Workflow, debug: bool = False
) -> Tuple[List[VRTInterferogram], Path, Path]:
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    cfg : Workflow
        [`Workflow`][dolphin.workflows.config.Workflow] object with workflow parameters
    debug : bool, optional
        Enable debug logging, by default False.

    Returns
    -------
    List[VRTInterferogram]
        List of virtual interferograms created.
    Path
        Path the final compressed SLC file created.
    Path
        Path to temporal correlation file created.
        In the case of a single phase linking step, this is the one tcorr file.
        In the case of sequential phase linking, this is the average tcorr file.
    """
    logger = get_log(debug=debug)
    scratch_dir = cfg.outputs.scratch_directory

    input_file_list = cfg.cslc_file_list
    if not input_file_list:
        raise ValueError("No input files found")

    # #############################################
    # 1. Make a VRT pointing to the input SLC files
    # #############################################
    subdataset = cfg.input_meta.subdataset
    vrt_path = scratch_dir / "slc_stack.vrt"
    if vrt_path.exists():
        vrt_stack = stack.VRTStack.from_vrt_file(vrt_path)
    else:
        vrt_stack = stack.VRTStack(
            input_file_list,
            subdataset=subdataset,
            outfile=scratch_dir / "slc_stack.vrt",
        )

    # ###############
    # 2. PS selection
    # ###############
    ps_output = cfg.ps_options._output_file
    if ps_output.exists():
        logger.info(f"Skipping making existing PS file {ps_output}")
    else:
        logger.info(f"Creating persistent scatterer file {ps_output}")
        ps.create_ps(
            slc_vrt_file=vrt_stack.outfile,
            output_file=ps_output,
            amp_mean_file=cfg.ps_options._amp_mean_file,
            amp_dispersion_file=cfg.ps_options._amp_dispersion_file,
            amp_dispersion_threshold=cfg.ps_options.amp_dispersion_threshold,
            block_size_gb=cfg.worker_settings.block_size_gb,
        )

    # #########################
    # 3. phase linking/EVD step
    # #########################
    pl_path = cfg.phase_linking._directory

    watcher = NvidiaMemoryWatcher() if gpu_is_available() else None
    phase_linked_slcs = list(pl_path.glob("2*.tif"))
    if len(phase_linked_slcs) > 0:
        logger.info(f"Skipping EVD step, {len(phase_linked_slcs)} files already exist")
        comp_slc_file = next(pl_path.glob("compressed*tif"))
        tcorr_file = next(pl_path.glob("tcorr*tif"))
    else:
        logger.info(f"Running sequential EMI step in {pl_path}")
        if cfg.workflow_name == "single":
            phase_linked_slcs, comp_slc_file, tcorr_file = single.run_evd_single(
                slc_vrt_file=vrt_stack.outfile,
                output_folder=pl_path,
                half_window=cfg.phase_linking.half_window.dict(),
                strides=cfg.outputs.strides,
                reference_idx=0,
                # mask_file=cfg.inputs.mask_file,
                ps_mask_file=ps_output,
                max_bytes=cfg.worker_settings.block_size_gb * 1e9,
                n_workers=cfg.worker_settings.n_workers,
                gpu_enabled=cfg.worker_settings.gpu_enabled,
                beta=cfg.phase_linking.beta,
            )
        else:
            phase_linked_slcs, comp_slcs, tcorr_file = sequential.run_evd_sequential(
                slc_vrt_file=vrt_stack.outfile,
                output_folder=pl_path,
                half_window=cfg.phase_linking.half_window.dict(),
                strides=cfg.outputs.strides,
                ministack_size=cfg.phase_linking.ministack_size,
                # mask_file=cfg.inputs.mask_file,
                ps_mask_file=ps_output,
                max_bytes=cfg.worker_settings.block_size_gb * 1e9,
                n_workers=cfg.worker_settings.n_workers,
                gpu_enabled=cfg.worker_settings.gpu_enabled,
                beta=cfg.phase_linking.beta,
            )
            comp_slc_file = comp_slcs[-1]

    if watcher:
        watcher.notify_finished()

    # ###################################################
    # 4. Form interferograms from estimated wrapped phase
    # ###################################################
    ifg_dir = cfg.interferogram_network._directory
    existing_ifgs = list(ifg_dir.glob("*.int.vrt"))
    if len(existing_ifgs) > 0:
        logger.info(f"Skipping interferogram step, {len(existing_ifgs)} exists")
    else:
        # phase_linked_slcs = sorted(pl_path.glob("20*.tif"))
        logger.info(
            f"Creating virtual interferograms from {len(phase_linked_slcs)} files"
        )
        if Path(vrt_stack.file_list[0]).name.startswith("compressed"):
            # The first file is a compressed SLC, so we want to use that as the reference
            # TODO: will this fail with anything by single-reference/manual-index?
            slc_list = [vrt_stack.file_list[0]] + phase_linked_slcs
        else:
            slc_list = phase_linked_slcs

        network = Network(
            slc_list=slc_list,
            reference_idx=cfg.interferogram_network.reference_idx,
            max_bandwidth=cfg.interferogram_network.max_bandwidth,
            max_temporal_baseline=cfg.interferogram_network.max_temporal_baseline,
            indexes=cfg.interferogram_network.indexes,
            outdir=ifg_dir,
        )
        if len(network) == 0:
            raise ValueError("No interferograms were created")

    return network.ifg_list, comp_slc_file, tcorr_file

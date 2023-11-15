import numpy.testing as npt
import pytest

from dolphin import _readers, io, stack
from dolphin.phase_link import mle, simulate
from dolphin.utils import gpu_is_available
from dolphin.workflows import sequential

GPU_AVAILABLE = gpu_is_available()
simulate._seed(1234)

# 'Grid size 49 will likely result in GPU under-utilization due to low occupancy.'
pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaPerformanceWarning"
)


@pytest.mark.parametrize("gpu_enabled", [True, False])
def test_sequential_gtiff(tmp_path, slc_file_list, gpu_enabled):
    """Run through the sequential estimation with a GeoTIFF stack."""
    vrt_file = tmp_path / "slc_stack.vrt"
    vrt_stack = _readers.VRTStack(slc_file_list, outfile=vrt_file)
    _, rows, cols = vrt_stack.shape

    half_window = {"x": cols // 2, "y": rows // 2}
    strides = {"x": 1, "y": 1}
    output_folder = tmp_path / "sequential"
    ms_size = 10
    ms_planner = stack.MiniStackPlanner(
        file_list=slc_file_list,
        dates=vrt_stack.dates,
        is_compressed=[False] * len(slc_file_list),
        output_folder=output_folder,
    )

    sequential.run_wrapped_phase_sequential(
        slc_vrt_file=vrt_file,
        ministack_planner=ms_planner,
        ministack_size=ms_size,
        half_window=half_window,
        strides=strides,
        ps_mask_file=None,
        amp_mean_file=None,
        amp_dispersion_file=None,
        shp_method="rect",
        shp_alpha=None,
        shp_nslc=None,
        n_workers=4,
        gpu_enabled=gpu_enabled,
    )

    assert len(list(output_folder.glob("2*.slc.tif"))) == vrt_stack.shape[0]

    # TODO: This probably won't work with just random data
    pytest.skip("This test is not working with random data.")
    # Get the MLE estimates from the entire stack output.
    slc_stack = vrt_stack.read_stack()
    mle_est, _, _ = mle.run_mle(
        slc_stack,
        half_window=half_window,
        strides=strides,
        gpu_enabled=gpu_enabled,
    )

    # Check that the sequential output matches the MLE estimates.
    for idx, out_file in enumerate(sorted(output_folder.glob("2*.slc.tif"))):
        layer = io.load_gdal(out_file)
        expected = mle_est[idx]
        npt.assert_allclose(layer, expected, atol=1e-3)


# @pytest.mark.parametrize(
#     "half_window", [{"x": 1, "y": 1}, {"x": 2, "y": 2}, {"x": 3, "y": 2}]
# )
# @pytest.mark.parametrize(
#     "strides", [{"x": 1, "y": 1}, {"x": 1, "y": 2}, {"x": 2, "y": 2}, {"x": 3, "y": 2}]
# )
# Input is only (5, 10) so we can't use a larger window.
@pytest.mark.parametrize(
    "half_window, strides",
    [
        ({"x": 1, "y": 1}, {"x": 1, "y": 1}),
        ({"x": 2, "y": 1}, {"x": 2, "y": 3}),
        ({"x": 3, "y": 1}, {"x": 3, "y": 1}),
    ],
)
def test_sequential_nc(tmp_path, slc_file_list_nc, half_window, strides):
    """Check various strides/windows/ministacks with a NetCDF input stack."""
    vrt_file = tmp_path / "slc_stack.vrt"
    v = _readers.VRTStack(slc_file_list_nc, outfile=vrt_file, subdataset="data")
    ms_planner = stack.MiniStackPlanner(
        file_list=slc_file_list_nc,
        dates=v.dates,
        is_compressed=[False] * len(slc_file_list_nc),
        output_folder=tmp_path / "sequential",
    )

    sequential.run_wrapped_phase_sequential(
        slc_vrt_file=vrt_file,
        ministack_planner=ms_planner,
        ministack_size=10,
        half_window=half_window,
        strides=strides,
        ps_mask_file=None,
        amp_mean_file=None,
        amp_dispersion_file=None,
        shp_method="rect",
        shp_alpha=None,
        shp_nslc=None,
        n_workers=4,
        gpu_enabled=False,
    )


@pytest.mark.parametrize("ministack_size", [5, 9, 20])
def test_sequential_ministack_sizes(tmp_path, slc_file_list_nc, ministack_size):
    """Check various strides/windows/ministacks with a NetCDF input stack."""
    vrt_file = tmp_path / "slc_stack.vrt"
    # Make it not a round number to test
    vrt_stack = _readers.VRTStack(
        slc_file_list_nc[:21], outfile=vrt_file, subdataset="data"
    )
    _, rows, cols = vrt_stack.shape
    ms_planner = stack.MiniStackPlanner(
        file_list=vrt_stack.file_list,
        dates=vrt_stack.dates,
        is_compressed=[False] * len(vrt_stack.file_list),
        output_folder=tmp_path / "sequential",
    )

    # Record the warning, check after if it's thrown
    sequential.run_wrapped_phase_sequential(
        slc_vrt_file=vrt_file,
        ministack_planner=ms_planner,
        ministack_size=ministack_size,
        half_window={"x": cols // 2, "y": rows // 2},
        strides={"x": 1, "y": 1},
        ps_mask_file=None,
        amp_mean_file=None,
        amp_dispersion_file=None,
        shp_method="rect",
        shp_alpha=None,
        shp_nslc=None,
        n_workers=4,
        gpu_enabled=False,
    )

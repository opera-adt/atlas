"""Module for handling blocked/decimated input and output."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional

from dolphin._types import HalfWindow, Strides
from dolphin.utils import compute_out_shape

# 1. iterate without overlap over output, decimated array
#   - Start with an offset, shrink the size, both by `half_window//strides`
#   - since we want every one wot be full, just skipping the incomplete blocks
# 2. map the current output pixels to their input locations (using strides)
# 3. pad the input block (using half window)

__all__ = [
    "BlockIndices",
    "StridedBlockManager",
    "iter_blocks",
]


logger = logging.getLogger("dolphin")


@dataclass(frozen=True)
class BlockIndices:
    """Class holding slices for 2D array access."""

    row_start: int
    row_stop: Optional[int]  # Can be None if we want slice(0, None)
    col_start: int
    col_stop: Optional[int]

    @classmethod
    def from_slices(cls, row_slice: slice, col_slice: slice) -> BlockIndices:
        return cls(
            row_start=row_slice.start,
            row_stop=row_slice.stop,
            col_start=col_slice.start,
            col_stop=col_slice.stop,
        )

    @property
    def row_slice(self) -> slice:
        return slice(self.row_start, self.row_stop)

    @property
    def col_slice(self) -> slice:
        return slice(self.col_start, self.col_stop)

    # Allows user to unpack like (row_slice, col_slice) = block_index
    def __iter__(self):
        return iter((self.row_slice, self.col_slice))


def iter_blocks(
    arr_shape: tuple[int, int],
    block_shape: tuple[int, int],
    overlaps: tuple[int, int] = (0, 0),
    start_offsets: tuple[int, int] = (0, 0),
    end_margin: tuple[int, int] = (0, 0),
) -> Iterator[BlockIndices]:
    """Create a generator to get indexes for accessing blocks of a raster.

    Parameters
    ----------
    arr_shape : tuple[int, int]
        (num_rows, num_cols), full size of array to access
    block_shape : tuple[int, int]
        (height, width), size of blocks to load
    overlaps : tuple[int, int], default = (0, 0)
        (row_overlap, col_overlap), number of pixels to re-include from
        the previous block after sliding
    start_offsets : tuple[int, int], default = (0, 0)
        Offsets from top left to start reading from
    end_margin : tuple[int, int], default = (0, 0)
        Margin to avoid at the bottom/right of array

    Yields
    ------
    BlockIndices
        Iterator of BlockIndices, which can be unpacked into
        (slice(row_start, row_stop), slice(col_start, col_stop))

    Examples
    --------
        >>> list(iter_blocks((180, 250), (100, 100)))
        [BlockIndices(row_start=0, row_stop=100, col_start=0, col_stop=100), \
BlockIndices(row_start=0, row_stop=100, col_start=100, col_stop=200), \
BlockIndices(row_start=0, row_stop=100, col_start=200, col_stop=250), \
BlockIndices(row_start=100, row_stop=180, col_start=0, col_stop=100), \
BlockIndices(row_start=100, row_stop=180, col_start=100, col_stop=200), \
BlockIndices(row_start=100, row_stop=180, col_start=200, col_stop=250)]
        >>> list(map(tuple, iter_blocks((180, 250), (100, 100), overlaps=(10, 10))))
        [(slice(0, 100, None), slice(0, 100, None)), (slice(0, 100, None), \
slice(90, 190, None)), (slice(0, 100, None), slice(180, 250, None)), \
(slice(90, 180, None), slice(0, 100, None)), (slice(90, 180, None), \
slice(90, 190, None)), (slice(90, 180, None), slice(180, 250, None))]

    """
    total_rows, total_cols = arr_shape
    height, width = block_shape
    row_overlap, col_overlap = overlaps
    start_row_offset, start_col_offset = start_offsets
    last_row = total_rows - end_margin[0]
    last_col = total_cols - end_margin[1]

    if height is None:
        height = total_rows
    if width is None:
        width = total_cols

    # Check we're not moving backwards with the overlap:
    if row_overlap >= height and height != total_rows:
        msg = f"{row_overlap = } must be less than block height {height}"
        raise ValueError(msg)
    if col_overlap >= width and width != total_cols:
        msg = f"{col_overlap = } must be less than block width {width}"
        raise ValueError(msg)

    # Set up the iterating indices
    cur_row = start_row_offset
    cur_col = start_col_offset
    while cur_row < last_row:
        while cur_col < last_col:
            row_stop = min(cur_row + height, last_row)
            col_stop = min(cur_col + width, last_col)
            # yield (slice(cur_row, row_stop), slice(cur_col, col_stop))
            yield BlockIndices(cur_row, row_stop, cur_col, col_stop)

            cur_col += width
            if cur_col < last_col:  # dont bring back if already at edge
                cur_col -= col_overlap

        cur_row += height
        if cur_row < last_row:
            cur_row -= row_overlap
        cur_col = start_col_offset  # reset back to the starting offset


def get_full_res_trim(
    in_no_pad_block: BlockIndices, in_block: BlockIndices
) -> BlockIndices:
    in_no_pad_rows, in_no_pad_cols = in_no_pad_block
    in_rows, in_cols = in_block
    # If the stops are the same, return None instead of slice(0, 0)
    col_stop = (in_no_pad_cols.stop - in_cols.stop) or None
    row_stop = (in_no_pad_rows.stop - in_rows.stop) or None
    trim_full_col = slice(in_no_pad_cols.start - in_cols.start, col_stop)
    trim_full_row = slice(in_no_pad_rows.start - in_rows.start, row_stop)
    return BlockIndices.from_slices(trim_full_row, trim_full_col)


def get_slice_length(s: slice, data_size: int = 1_000_000):
    """Get the size of a slice of data.

    Uses `slice.indices` to avoid making any dummy data.
    Assumes that 1. data is larger than slice, and 2. size is
    less than `data_size`.
    """
    return len(range(*s.indices(data_size)))


def pad_block(in_block: BlockIndices, margins: tuple[int, int]) -> BlockIndices:
    """Pad `in_block` by the (row_margin, col_margin) pixels in `margins`.

    Will clip the row/column slice starts to be >= 0.

    Parameters
    ----------
    in_block : BlockIndices
        Slices original array block.
    margins : dict[int, int]
        Number of pixels to extend `in_block` in the (row, col) directions.
        The margins are subtracted from the beginning and added to the end,
        so the block size grows by (2 * row_margin, 2 * col_margin)

    Returns
    -------
    BlockIndices
        Output slices for larger block

    Raises
    ------
    ValueError
        If the block is too small to be padded by the given margins
        (leads to start < 0)

    """
    r_margin, c_margin = margins
    r_slice, c_slice = in_block
    if r_slice.start - r_margin < 0:
        msg = f"{r_slice = }, but {r_margin = }"
        raise ValueError(msg)
    if c_slice.start - c_margin < 0:
        msg = f"{c_slice = }, but {c_margin = }"
        raise ValueError(msg)
    return BlockIndices(
        # max(r_slice.start - r_margin, 0),
        r_slice.start - r_margin,
        r_slice.stop + r_margin,
        # max(c_slice.start - c_margin, 0),
        c_slice.start - c_margin,
        c_slice.stop + c_margin,
    )


def dilate_block(in_block: BlockIndices, strides: Strides) -> BlockIndices:
    """Grow slices in `BlockIndices` to fit a larger array.

    This is to undo the "stride"/decimation index changes, so
     we can go from smaller, strided array indices to the original.

    Assumes in in_block is the smaller one which has been made
    by taking `strides` from the larger block.

    Parameters
    ----------
    in_block : BlockIndices
        Slices for an smaller, strided array.
    strides : Strides or tuple[int, int]
        Decimation factor in y and c which was used for `in_block`

    Returns
    -------
    BlockIndices
        Output slices for larger array

    """
    row_strides, col_strides = strides
    row_start = in_block.row_start * row_strides
    col_start = in_block.col_start * col_strides
    row_stop = None if in_block.row_stop is None else in_block.row_stop * row_strides
    col_stop = None if in_block.col_stop is None else in_block.col_stop * col_strides
    return BlockIndices(row_start, row_stop, col_start, col_stop)


@dataclass
class StridedBlockManager:
    """Class to handle slicing/trimming overlapping blocks with strides."""

    arr_shape: tuple[int, int]
    """(row, col) of the full-res 2D image"""
    block_shape: tuple[int, int]
    """(row, col) size of each input block to operate on at one time"""
    strides: Strides = field(default_factory=lambda: Strides(1, 1))
    """Decimation/downsampling factor in y/row and x/column direction"""
    half_window: HalfWindow = field(default_factory=lambda: HalfWindow(0, 0))
    """For multi-looking iterations, size of the (full-res) half window
    in y/row and x/column direction.
    Used to find `overlaps` between blocks and `start_offset`/`end_margin` for
    `iter_blocks`."""

    def iter_blocks(
        self,
    ) -> Iterator[
        tuple[BlockIndices, BlockIndices, BlockIndices, BlockIndices, BlockIndices]
    ]:
        """Iterate over the input/output block indices.

        Yields
        ------
        output_block : BlockIndices
            The current slices for the output raster
        out_trim : BlockIndices
            Slices to use on a processed output block to remove nodata border pixels.
            These may be relative (e.g. slice(1, -1)), not absolute like `output_block`.
        input_block : BlockIndices
            Slices used to load the full-res input data
        input_no_padding : BlockIndices
            Slices which point to the position within the full-res data without padding
        input_trim : BlockIndices
            Slices to use on full-res input block to remove padding from half-window.

        """
        out_trim = self.get_trimming_block()
        for out_block in self.iter_outputs():
            input_no_padding = dilate_block(out_block, strides=self.strides)
            input_block = pad_block(input_no_padding, margins=self.input_padding_shape)
            input_trim = get_full_res_trim(input_no_padding, input_block)
            yield (out_block, out_trim, input_block, input_no_padding, input_trim)

    def _get_out_nodata_size(self, direction: str) -> int:
        half_win = getattr(self.half_window, direction)
        stride = getattr(self.strides, direction)
        return round(half_win / stride)

    @property
    def output_shape(self) -> tuple[int, int]:
        return compute_out_shape(self.arr_shape, self.strides)

    @property
    def out_block_shape(self) -> tuple[int, int]:
        return compute_out_shape(self.block_shape, self.strides)

    @property
    def input_padding_shape(self) -> tuple[int, int]:
        """Amount of extra padding the input blocks need.

        Depends on the window size, and the strides.
        """
        return (
            self.strides.y * self._get_out_nodata_size("y"),
            self.strides.x * self._get_out_nodata_size("x"),
        )

    @property
    def output_margin(self) -> tuple[int, int]:
        """The output margins that we ignore while iterating.

        Depends on the half window and strides.

        The `half_window` is in full-res (input) coordinates (which would be the
        amount to skip with no striding), so the output margin size is smaller
        """
        return (self._get_out_nodata_size("y"), self._get_out_nodata_size("x"))

    def iter_outputs(self) -> Iterator[BlockIndices]:
        yield from iter_blocks(
            arr_shape=self.output_shape,
            block_shape=self.out_block_shape,
            overlaps=(0, 0),  # We're not overlapping in the *output* grid
            start_offsets=self.output_margin,
            end_margin=self.output_margin,
        )

    def get_trimming_block(self) -> BlockIndices:
        """Compute the slices which trim output nodata values.

        When the BlockIndex gets dilated (using `strides`) and padded (using
        `half_window`), the result will have nodata around the edges.
        The size of the nodata pixels in the full-res block is just
            (half_window['y'], half_window['x'])
        In the output (strided) coordinates, the number of nodata pixels is
        shrunk by how many strides are taken.

        Note that this is independent of which block we're on; the number of
        nodata pixels on the border is always the same.
        """
        row_nodata = self._get_out_nodata_size("y")
        col_nodata = self._get_out_nodata_size("x")
        # Extra check if we have no trimming to do: use slice(0, None)
        row_end = -row_nodata if row_nodata > 0 else None
        col_end = -col_nodata if col_nodata > 0 else None
        return BlockIndices(row_nodata, row_end, col_nodata, col_end)

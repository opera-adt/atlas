from __future__ import annotations

import functools
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("dolphin")

__all__ = [
    "atomic_output",
]


def atomic_output(
    output_arg: str = "output_file",
    is_dir: bool = False,
    use_tmp: bool = False,
    overwrite: bool = False,
) -> Callable:
    """Use a temporary file/directory for the `output_arg` until the function finishes.

    Decorator is used on a function which writes to an output file/directory in blocks.
    If the function were interrupted, the file/directory would be partially complete.

    This decorator replaces the final output name with a temp file/dir, and then
    renames the temp file/dir to the final name after the function finishes.

    Note that when `is_dir=True`, `output_arg` can be a directory (if multiple files
    are being written to). In this case, the entire directory is temporary, and
    renamed after the function finishes.

    Parameters
    ----------
    output_arg : str, optional
        The name of the argument to replace, by default 'output_file'
    is_dir : bool, default = False
        If `True`, the output argument is a directory, not a file
    use_tmp : bool, default = False
        If `False`, uses the parent directory of the desired output, with
        a random suffix added to the name to distinguish from actual output.
        If `True`, uses the `/tmp` directory (or wherever the default is
        for the `tempfile` module).
    overwrite : bool, default = False
        Overwrite an existing file.
        If `False` raises `FileExistsError` if the file already exists.

    Returns
    -------
    Callable
        The decorated function

    Raises
    ------
    FileExistsError
        if the file for `output_arg` already exists (if out_dir=False), or
        if the directory at `output_arg` exists and is non-empty.

    Notes
    -----
    The output at `output_arg` *must not* exist already, or the decorator will error
    (though if `is_dir=True`, it is allowed to be an empty directory).
    The function being decorated *must* be called with keyword args for `output_arg`.

    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extract the output file path
            if kwargs.get(output_arg):
                final_out_name = kwargs[output_arg]
            else:
                msg = (
                    f"Argument {output_arg} not passed to function {func.__name__}:"
                    f" {kwargs}"
                )
                raise FileExistsError(msg)

            final_path = Path(final_out_name)
            # Make sure the desired final output doesn't already exist
            _raise_if_exists(final_path, is_dir=is_dir, overwrite=overwrite)
            # None means that tempfile will use /tmp
            tmp_dir = final_path.parent if not use_tmp else None

            # Make the tempfile start the same as the desired output
            prefix = final_path.name
            if is_dir:
                # Create a temporary directory
                temp_path = tempfile.mkdtemp(dir=tmp_dir, prefix=prefix)
            else:
                # Create a temporary file
                suffix = final_path.suffix
                _, temp_path = tempfile.mkstemp(
                    dir=tmp_dir, prefix=prefix, suffix=suffix
                )
            logger.debug("Writing to temp file %s instead of %s", temp_path, final_path)

            try:
                # Replace the output file path with the temp file
                # It would be like this if we only allows keyword:
                kwargs[output_arg] = temp_path
                # Execute the original function
                result = func(*args, **kwargs)
                # Move the temp file to the final location
                logger.debug("Moving %s to %s", temp_path, final_path)
                shutil.move(temp_path, final_path)

                return result
            finally:
                logger.debug("Cleaning up temp file %s", temp_path)
                # Different cleanup is needed
                if is_dir:
                    shutil.rmtree(temp_path, ignore_errors=True)
                else:
                    Path(temp_path).unlink(missing_ok=True)

        return wrapper

    return decorator


def _raise_if_exists(final_path: Path, is_dir: bool, overwrite: bool):
    msg = f"{final_path} already exists"
    if final_path.exists():
        logger.debug(f"{final_path} already exists")
        if overwrite:
            if final_path.is_dir():
                shutil.rmtree(final_path)
            else:
                final_path.unlink()
            return

        if is_dir and final_path.is_dir():
            # We can work with an empty directory
            try:
                final_path.rmdir()
            except OSError as e:
                err_msg = str(e)
                if "Directory not empty" in err_msg:
                    raise FileExistsError(msg) from e
                else:
                    # Some other error we don't know
                    raise
        else:
            raise FileExistsError(msg)

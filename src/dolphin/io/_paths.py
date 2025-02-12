from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from typing import Union
from urllib.parse import ParseResult, urlparse

from dolphin._types import GeneralPath

__all__ = ["S3Path"]


logger = logging.getLogger("dolphin")


class S3Path(GeneralPath):
    """A convenience class to handle paths on S3.

    This class relies on `pathlib.Path` for operations using `urllib` to parse the url.

    If passing a url with a trailing slash, the slash will be preserved
    when converting back to string.

    Note that pure path manipulation functions do *not* require `boto3`,
    but functions which interact with S3 (e.g. `exists()`, `.read_text()`) do.

    Attributes
    ----------
    bucket : str
        Name of bucket in the url
    path : pathlib.Path
        The URL path after s3://<bucket>/
    key : str
        Alias of `path` converted to a string

    Examples
    --------
    >>> from dolphin.io import S3Path
    >>> s3_path = S3Path("s3://bucket/path/to/file.txt")
    >>> str(s3_path)
    's3://bucket/path/to/file.txt'
    >>> s3_path.parent
    S3Path("s3://bucket/path/to/")
    >>> str(s3_path.parent)
    's3://bucket/path/to/'

    """

    def __init__(self, s3_url: Union[str, "S3Path"], unsigned: bool = False):
        """Create an S3Path.

        Parameters
        ----------
        s3_url : str or S3Path
            The S3 url to parse.
        unsigned : bool, optional
            If True, disable signing requests to S3.

        """
        # Names come from the urllib.parse.ParseResult
        if isinstance(s3_url, S3Path):
            self._scheme: str = s3_url._scheme
            self._netloc: str = s3_url._netloc
            self.bucket: str = s3_url.bucket
            self.path: Path = s3_url.path
            self._trailing_slash: str = s3_url._trailing_slash
        else:
            parsed: ParseResult = urlparse(s3_url)
            self._scheme = parsed.scheme
            self._netloc = self.bucket = parsed.netloc
            self._parsed = parsed
            self.path = Path(parsed.path)
            self._trailing_slash = "/" if s3_url.endswith("/") else ""

        if self._scheme != "s3":
            raise ValueError(f"{s3_url} is not an S3 url")

        self._unsigned = unsigned

    @classmethod
    def from_bucket_key(cls, bucket: str, key: str):
        """Create a `S3Path` from the bucket name and key/prefix.

        Matches API of some Boto3 functions which use this format.

        Parameters
        ----------
        bucket : str
            Name of S3 bucket.
        key : str
            S3 url of path after the bucket.

        """
        return cls(f"s3://{bucket}/{key}")

    def get_path(self):
        # For S3 paths, we need to add the double slash and netloc back to the front
        return f"{self._scheme}://{self._netloc}{self.path.as_posix()}{self._trailing_slash}"

    @property
    def key(self) -> str:
        """Name of key/prefix within the bucket with leading slash removed."""
        return f"{str(self.path.as_posix()).lstrip('/')}{self._trailing_slash}"

    @property
    def parent(self):
        parent_path = self.path.parent
        # Since this is a parent, it will will always end in a slash
        if self._scheme == "s3":
            # For S3 paths, we need to add the scheme and netloc back to the front
            return S3Path(f"{self._scheme}://{self._netloc}{parent_path.as_posix()}/")
        else:
            # For local paths, we can just convert the path to a string
            return S3Path(str(parent_path) + "/")

    @property
    def suffix(self):
        return self.path.suffix

    def resolve(self) -> S3Path:
        """Resolve the path to an absolute path- S3 paths are always absolute."""
        return self

    def _get_client(self):
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        if self._unsigned:
            return boto3.client("s3", config=Config(signature_version=UNSIGNED))
        else:
            return boto3.client("s3")

    def exists(self) -> bool:
        """Whether this path exists on S3."""
        client = self._get_client()
        resp = client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self.key,
            MaxKeys=1,
        )
        return resp.get("KeyCount") == 1

    def read_text(self) -> str:
        """Download/read the S3 file as text."""
        return self._download_as_bytes().decode()

    def read_bytes(self) -> bytes:
        """Download/read the S3 file as bytes."""
        return self._download_as_bytes()

    def _download_as_bytes(self) -> bytes:
        """Download file to a `BytesIO` buffer to read as bytes."""
        from io import BytesIO

        client = self._get_client()

        bio = BytesIO()
        client.download_fileobj(self.bucket, self.key, bio)
        bio.seek(0)
        out = bio.read()
        bio.close()
        return out

    def __truediv__(self, other):
        new = copy.deepcopy(self)
        new.path = self.path / other
        new._trailing_slash = "/" if str(other).endswith("/") else ""
        return new

    def __eq__(self, other):
        if isinstance(other, S3Path):
            return self.get_path() == other.get_path()
        elif isinstance(other, str):
            return self.get_path() == other
        else:
            return False

    def __repr__(self):
        return f'S3Path("{self.get_path()}")'

    def __str__(self):
        return self.get_path()

    def to_gdal(self):
        """Convert this S3Path to a GDAL URL."""
        return f"/vsis3/{self.bucket}/{self.key}"


def fix_s3_url(url):
    """Fix an S3 URL that has been altered by pathlib.

    Will replace s3:/my-bucket/... with s3://my-bucket/...
    """
    return re.sub(r"s3:/((?!/).*)", r"s3://\1", str(url))

"""Low-level sysfs/debugfs file access."""

from __future__ import annotations

import os
from pathlib import Path

from ._errors import DevfreqError, DevfreqNotFoundError, DevfreqPermissionError


def under_root(root: str, path: str) -> str:
    """
    Place an absolute sysfs path under a filesystem root.

    root is "/" on a real system; tests point it at a fake sysfs tree.
    """
    if root in ("", "/"):
        return path
    return str(Path(root) / path.lstrip("/"))


class _SysfsNode:
    """
    Read/write wrapper for a single sysfs file.
    All I/O errors are translated to DevfreqError subclasses here so
    that controllers never need try/except around reads or writes.
    """

    def __init__(self, path: str) -> None:
        self._path = path

    @property
    def path(self) -> str:
        return self._path

    def exists(self) -> bool:
        try:
            os.stat(self._path)
        except (FileNotFoundError, NotADirectoryError):
            return False
        except PermissionError as exc:
            raise DevfreqPermissionError(
                f"Accessing {self._path} failed: run as root (sudo)"
            ) from exc
        return True

    def read(self) -> str:
        """Return the stripped file contents, or "" if the file does not exist."""
        try:
            return Path(self._path).read_text().strip()
        except (FileNotFoundError, NotADirectoryError):
            return ""
        except PermissionError as exc:
            raise DevfreqPermissionError(
                f"Reading {self._path} failed: run as root (sudo)"
            ) from exc

    def read_int(self) -> int:
        text = self.read()
        if not text:
            raise DevfreqNotFoundError(f"Sysfs path not found or empty: {self._path}")
        return int(text)

    def read_list(self) -> list[str]:
        return self.read().split()

    def write(self, value: str | int) -> None:
        try:
            Path(self._path).write_text(str(value) + "\n")
        except PermissionError as exc:
            raise DevfreqPermissionError(
                f"Writing to {self._path} failed: run as root (sudo)"
            ) from exc
        except FileNotFoundError as exc:
            raise DevfreqNotFoundError(
                f"Sysfs path not found: {self._path}"
            ) from exc
        except OSError as exc:
            # e.g. EINVAL when the kernel rejects the value
            raise DevfreqError(
                f"Kernel rejected writing {value!r} to {self._path}: {exc.strerror}"
            ) from exc

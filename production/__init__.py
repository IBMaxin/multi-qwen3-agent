"""Package init module."""

from __future__ import annotations

import sys

# Enforce Python 3.10.x only
REQUIRED_MAJOR = 3
REQUIRED_MINOR = 10
if (
    sys.version_info.major,
    sys.version_info.minor,
) != (
    REQUIRED_MAJOR,
    REQUIRED_MINOR,
):  # pragma: no cover
    detected = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    msg = (
        "Python 3.10.x is required for this project. "
        f"Detected {detected}. Please activate .venv310 or install Python 3.10."
    )
    raise RuntimeError(msg)

__version__: str = "1.0.0"

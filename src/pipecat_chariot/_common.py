#
# Copyright (c) 2026, Chariot AI
#
# SPDX-License-Identifier: MIT
#

"""Pieces shared by the Chariot speech-to-text and text-to-speech services."""

import platform
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from pipecat import version as pipecat_version

CHARIOT_BASE_URL = "wss://api.chariot.in"

# The server closes with this code when the workspace's credit balance is
# exhausted, both at connect time and mid-session.
CLOSE_CODE_NO_CREDITS = 4402

try:
    __version__ = _package_version("pipecat-chariot")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0"


def user_agent() -> str:
    """The ``User-Agent`` sent on every connection.

    Names this package, Pipecat and Python with their versions, so Chariot can
    tell which integration releases are in use.
    """
    python = ".".join(platform.python_version_tuple()[:2])
    return f"pipecat-chariot/{__version__} pipecat/{pipecat_version()} python/{python}"


def close_code(error: BaseException) -> int | None:
    """The close code the server sent, if the error is a closed connection."""
    rcvd = getattr(error, "rcvd", None)
    return getattr(rcvd, "code", None)


def close_reason(error: BaseException) -> str:
    """The close reason the server sent, or an empty string."""
    rcvd = getattr(error, "rcvd", None)
    return getattr(rcvd, "reason", "") or ""

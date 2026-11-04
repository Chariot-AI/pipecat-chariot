#
# Copyright (c) 2026, Chariot AI
#
# SPDX-License-Identifier: MIT
#

"""Chariot speech-to-text and text-to-speech for Pipecat."""

from pipecat_chariot._common import __version__
from pipecat_chariot.stt import (
    CHARIOT_DEFAULT_STT_MODEL,
    CHARIOT_STT_LANGUAGES,
    CHARIOT_STT_SAMPLE_RATES,
    ChariotSTTService,
    ChariotSTTSettings,
)
from pipecat_chariot.tts import (
    CHARIOT_DEFAULT_TTS_MODEL,
    CHARIOT_DEFAULT_VOICE,
    CHARIOT_SAMPLE_RATE,
    ChariotTTSService,
    ChariotTTSSettings,
)

__all__ = [
    "CHARIOT_DEFAULT_STT_MODEL",
    "CHARIOT_DEFAULT_TTS_MODEL",
    "CHARIOT_DEFAULT_VOICE",
    "CHARIOT_SAMPLE_RATE",
    "CHARIOT_STT_LANGUAGES",
    "CHARIOT_STT_SAMPLE_RATES",
    "ChariotSTTService",
    "ChariotSTTSettings",
    "ChariotTTSService",
    "ChariotTTSSettings",
    "__version__",
]

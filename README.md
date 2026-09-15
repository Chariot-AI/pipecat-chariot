# pipecat-chariot

Chariot text-to-speech for [Pipecat](https://github.com/pipecat-ai/pipecat).

`ChariotTTSService` is a drop-in Pipecat `TTSService` backed by [Chariot](https://chariot.in), which serves lifelike voices for Indian languages and English. Audio streams back as 16-bit mono PCM while the server is still generating, so playback starts before synthesis finishes.

This integration is built and maintained by Chariot, the company providing the service.

## Pipecat compatibility

Tested with Pipecat v1.10.0. The package requires `pipecat-ai>=1.7,<2` and Python 3.11 or newer, which is Pipecat's own floor.

## Installation

```bash
uv add pipecat-chariot
```

or

```bash
pip install pipecat-chariot
```

## Prerequisites

1. Create an account at [platform.chariot.in](https://platform.chariot.in). New accounts include free credits.
2. Copy an API key from the console.
3. Optionally pick a voice from `GET https://api.chariot.in/v1/voices`. Without one the service uses a Chariot stock voice, so the example below runs on the API key alone.

```bash
export CHARIOT_API_KEY="your-api-key"
export CHARIOT_VOICE_ID="your-voice-uuid"   # optional
```

## Usage

```python
import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat_chariot import ChariotTTSService

tts = ChariotTTSService(
    api_key=os.getenv("CHARIOT_API_KEY"),
    settings=ChariotTTSService.Settings(
        voice=os.getenv("CHARIOT_VOICE_ID"),
    ),
)

pipeline = Pipeline(
    [
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ]
)
```

## Configuration

### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | required | Chariot API key. Falls back to `CHARIOT_API_KEY`. |
| `voice_id` | `str \| None` | stock voice | Voice UUID from `GET /v1/voices`. `settings.voice` takes precedence. |
| `base_url` | `str` | `wss://api.chariot.in` | Override for a non-production endpoint. |
| `sample_rate` | `int \| None` | `44100` | Output rate in Hz. Chariot streams 44.1 kHz; a mismatch logs a warning. |
| `model` | `str \| None` | account default | Chariot TTS model. `settings.model` takes precedence. |
| `speed` | `float \| None` | voice as recorded | Speaking rate, one of the speeds the voice supports. `settings.speed` takes precedence. |
| `optimize_streaming_latency` | `int` | `0` | Latency optimization level, 0 to 4. Higher trades quality passes for speed. |

### Settings

Pass through `ChariotTTSService.Settings(...)`. Updatable at runtime with `TTSUpdateSettingsFrame`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `voice` | `str` | stock voice | Voice UUID. |
| `model` | `str \| None` | account default | Chariot TTS model. |
| `speed` | `float \| None` | voice as recorded | Speaking rate, one of the speeds the voice supports. |

Each voice supports a set of speeds under each model, commonly `0.5`, `0.75`,
`0.9`, `1.1`, `1.25` and `1.5`; `GET /v1/voices` lists what a given voice
supports. Voice, model and speed are fixed for the life of a session, so
changing any of them reconnects, which the service handles for you. Leaving
model or speed unset uses your account default, so a newly released model works
without upgrading this package.

There is no language parameter. Any voice speaks any supported language, so the audio follows the text.

## Run the example

The example is a full voice agent: Chariot speaks, and you can interrupt it mid-sentence.

```bash
git clone https://github.com/Chariot-AI/pipecat-chariot.git
cd pipecat-chariot
uv venv && uv pip install -e ".[dev]"
uv pip install "pipecat-ai[openai,silero,webrtc,runner]"
cp .env.example .env    # add CHARIOT_API_KEY and OPENAI_API_KEY
uv run python examples/voice_chariot.py
```

The example runs in the browser over WebRTC, so it needs no third party
transport account. Open the URL it prints, allow the microphone, and talk.
Speech to text and the LLM use OpenAI; swap in any Pipecat services you prefer.

`ChariotTTSService` itself is transport agnostic. It sits between the LLM and
`transport.output()` and never sees the transport, so it works with any of
them: Daily, LiveKit, telephony, WebSocket. The example demos WebRTC because
that needs no account; to try another, add its key to `transport_params` and
install its extra.

## How it behaves

- **Streaming**: audio frames are yielded while the server is still generating. Output is 16-bit mono PCM at 44.1 kHz, and every frame carries the rate the server declares in its `audio.start` event.
- **Interruptions**: built on `InterruptibleTTSService`, the base class for websocket TTS services that support neither context IDs nor a cancel message. On interruption the socket is closed, which stops server side generation immediately.
- **Long text**: the server caps its un-flushed buffer, so an over long utterance is sent as several word boundary segments, each flushed on its own. One generation segment per flush, contiguous audio.
- **Keepalive**: a flush with nothing buffered is a server side no-op that still resets the idle timeout, so a quiet stretch does not drop the connection.
- **Turn boundaries**: `audio.done` marks the end of one flushed segment, not the end of the turn. The turn closes when the LLM turn has completed and every segment sent has come back, so a multi sentence reply is never cut short and `BotStoppedSpeakingFrame` still tracks the real end of audio.
- **Errors**: failures surface as Pipecat `ErrorFrame`s rather than hanging the pipeline. An exhausted credit balance is reported as exactly that.

## Tests

```bash
uv run pytest
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)

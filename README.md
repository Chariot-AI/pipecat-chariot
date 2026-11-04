# pipecat-chariot

Chariot speech-to-text and text-to-speech for [Pipecat](https://github.com/pipecat-ai/pipecat).

[Chariot](https://chariot.in) serves lifelike voices and real-time speech recognition for Indian languages and English, including code-mixed speech. This package gives Pipecat two services:

- `ChariotSTTService` streams the user's audio and returns interim and final transcripts.
- `ChariotTTSService` streams speech back as 16-bit mono PCM while the server is still generating, and cancels a reply the moment the user interrupts, without reconnecting.

This integration is built and maintained by Chariot, the company providing the service.

## Pipecat compatibility

Tested with Pipecat v1.15.0. The package requires `pipecat-ai>=1.13,<2` and Python 3.11 or newer, which is Pipecat's own floor.

## Installation

```bash
uv add pipecat-chariot
```

## Prerequisites

1. Create an account at [platform.chariot.in](https://platform.chariot.in). New accounts include free credits.
2. Copy an API key from the console.
3. Optionally pick a voice from `GET https://api.chariot.in/v1/voices`. Without one the TTS service uses a Chariot stock voice, so everything runs on the API key alone.

```bash
export CHARIOT_API_KEY="your-api-key"
```

## Usage

```python
import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat.transcriptions.language import Language
from pipecat_chariot import ChariotSTTService, ChariotTTSService

stt = ChariotSTTService(
    api_key=os.getenv("CHARIOT_API_KEY"),
    settings=ChariotSTTService.Settings(language=Language.HI),  # omit to detect
)

tts = ChariotTTSService(
    api_key=os.getenv("CHARIOT_API_KEY"),
    settings=ChariotTTSService.Settings(voice=os.getenv("CHARIOT_VOICE_ID")),
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

Both services are transport agnostic, so they work with any Pipecat transport: WebRTC, Daily, LiveKit, telephony or WebSocket.

## Speech-to-text

### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | required | Chariot API key. |
| `sample_rate` | `int \| None` | pipeline rate | Input rate in Hz: 8000, 16000, 22050, 24000, 32000, 44100 or 48000. |
| `commit_strategy` | `"manual" \| "vad"` | `"manual"` | Who decides where an utterance ends. See below. |
| `include_timestamps` | `bool` | `False` | Word timestamps, in each transcription frame's `result["words"]`. |
| `vad_threshold` | `float \| None` | server default | Server VAD speech threshold, 0.05 to 0.95. `vad` only. |
| `silence_duration_ms` | `int \| None` | server default | Server VAD silence that ends an utterance, 100 to 10000. `vad` only. |
| `min_speech_duration_ms` | `int \| None` | server default | Server VAD minimum speech length, 0 to 5000. `vad` only. |
| `prefix_padding_ms` | `int \| None` | server default | Server VAD audio kept before speech, 0 to 2000. `vad` only. |
| `ttfs_p99_latency` | `float \| None` | by mode | P99 seconds from end of speech to final transcript, used by Pipecat to time turn ends. Defaults to 0.6 for `balanced` and 0.8 for `high_accuracy`, plus the silence window with `vad`. |
| `extra_headers` | `dict \| None` | `None` | Additional headers for the WebSocket handshake. |
| `base_url` | `str` | `wss://api.chariot.in` | API endpoint. |

### Settings

Pass through `ChariotSTTService.Settings(...)`. Updatable with `STTUpdateSettingsFrame`, which reconnects.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `language` | `Language \| None` | detected | English or one of Hindi, Bengali, Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, Sanskrit, Tamil or Telugu. |
| `mode` | `"balanced" \| "high_accuracy" \| None` | `balanced` | `high_accuracy` is more accurate (it writes numbers as digits, for example); its interim transcripts update less often, and the final one takes about 0.15 s longer. |
| `style` | `"codemix" \| "native" \| None` | `codemix` | `codemix` writes English words in Latin script inside Indian language speech; `native` writes everything in the language's own script. |
| `model` | `str` | `chariot-1-asr-realtime` | The real-time model. |

### Turn ending

With `commit_strategy="manual"` (the default), Pipecat's VAD decides when the user has stopped speaking. The service then asks Chariot to finalize the utterance, and the transcript that answers is marked final, so the turn ends without waiting out a timeout.

With `commit_strategy="vad"`, Chariot's own voice detection splits utterances. Pipecat's VAD still drives turn taking. The `on_speech_started` and `on_speech_stopped` event handlers report what the server heard.

### Connection

A dropped connection, a session reaching its maximum length, or every speech-to-text session your plan allows being in use is retried with backoff. A refusal that would repeat on every retry, such as an invalid key or an exhausted credit balance, is reported once and stops the service for the session.

### Billing

Speech-to-text is billed per second of audio sent. A voice agent streams audio for the whole call, silence included, so a call is billed for its full length, and the session holds one of your plan's concurrent speech-to-text sessions while it is open.

## Text-to-speech

### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | required | Chariot API key. |
| `voice_id` | `str \| None` | stock voice | Voice UUID from `GET /v1/voices`. `settings.voice` takes precedence. |
| `model` | `str \| None` | `chariot-multilingual-v1` | `chariot-multilingual-v1` or `chariot-v0`. `settings.model` takes precedence. |
| `speed` | `float \| None` | voice as recorded | Speaking rate. `settings.speed` takes precedence. |
| `enable_logging` | `bool \| None` | not sent | Unset or `True`: your workspace's data retention setting decides. `False` keeps the generated audio out of storage. |
| `optimize_streaming_latency` | `int \| None` | server default | 0 to 4. Higher trades quality passes for latency. |
| `sample_rate` | `int \| None` | `44100` | Chariot streams 44.1 kHz; a mismatch logs a warning. |
| `extra_headers` | `dict \| None` | `None` | Additional headers for the WebSocket handshake. |
| `base_url` | `str` | `wss://api.chariot.in` | API endpoint. |

### Settings

Pass through `ChariotTTSService.Settings(...)`. Updatable at runtime with `TTSUpdateSettingsFrame`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `voice` | `str` | stock voice | Voice UUID. Applies from the next turn, on the same connection. |
| `model` | `str` | `chariot-multilingual-v1` | Fixed for a connection, so changing it reconnects. |
| `speed` | `float \| None` | voice as recorded | Applies from the next turn, on the same connection. |

Speaking rates are pre-encoded per voice and per model, so `speed` is one of a fixed set rather than a range, and a rate a voice does not have is refused. `GET /v1/voices` lists what each voice offers under each model as `speeds`.

There is no language parameter. Any voice speaks any supported language, so the audio follows the text.

### How it behaves

- **One connection, one context per turn**: each bot turn is a Chariot context on a single WebSocket. Every sentence is generated as soon as Pipecat hands it over, so audio starts while the LLM is still writing.
- **Interruptions**: barging in cancels the turn's context. Generation stops at once, audio already in flight is dropped, and the next turn starts on the same connection with no new handshake.
- **Keepalive**: a quiet connection, and a turn waiting on a slow LLM, are kept open automatically.
- **Errors**: failures surface as Pipecat `ErrorFrame`s rather than hanging the pipeline. A refusal that would repeat on every retry, such as an invalid key or an exhausted credit balance, is reported once as exactly that and stops the service for the session.

## Run the example

The example is a full voice agent in the browser: Chariot listens and speaks, OpenAI writes the replies, and you can interrupt it mid sentence.

```bash
git clone https://github.com/Chariot-AI/pipecat-chariot.git
cd pipecat-chariot
cp .env.example .env    # add CHARIOT_API_KEY and OPENAI_API_KEY
uv run --extra example examples/voice_chariot.py
```

It runs over WebRTC, so it needs no third party transport account. Open the URL it prints, allow the microphone, and talk.

## Tests

```bash
uv run --extra dev pytest
```

The tests run real Pipecat pipelines against a local stand-in for the Chariot API, so they need no network or API key.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)

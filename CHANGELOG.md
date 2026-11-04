# Changelog

All notable changes to `pipecat-chariot` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0]

### Added

- `ChariotSTTService`, streaming speech-to-text over Chariot's real-time ASR
  WebSocket API: interim and final transcripts, language selection or
  detection, `balanced` and `high_accuracy` modes, `codemix` and `native`
  output styles, and optional word timestamps.
- Turn ending driven by Pipecat's VAD by default: when the user stops, the
  utterance is finalized and its transcript marked final, so the turn ends
  without waiting out a timeout. `commit_strategy="vad"` hands utterance
  splitting to Chariot's server side voice detection instead.
- `enable_logging` on `ChariotTTSService`, sent only when set. `False` keeps
  generated audio out of storage; unset or `True` follows the workspace's data
  retention setting.
- `extra_headers` on both services.

### Changed

- `ChariotTTSService` uses Chariot's multi-context WebSocket. Each bot turn is
  a context on one long lived connection, so an interruption cancels the turn
  without reconnecting and the next turn starts with no new handshake. The base
  class is now `WebsocketTTSService`.
- Voice and speed changes apply from the next turn on the same connection;
  only a model change reconnects.
- The default TTS model is `chariot-multilingual-v1`.
- Connections identify themselves with a `User-Agent` naming this package,
  Pipecat and Python, replacing the `X-Source` and `X-Pipecat-Version` headers.
- Requires `pipecat-ai>=1.13`.

### Fixed

- Updating one setting no longer resets the others: a delta that changed only
  the voice used to clear the speed.
- A pause of more than 3 seconds in generation no longer cuts a turn short.
  Pipecat gives up on a turn after that long without audio by default, which
  dropped the rest of its audio; the turn now ends when Chariot says it has.
- A refusal that would repeat on every retry (an invalid key, an exhausted
  credit balance) is reported once and stops the service for the session,
  instead of being retried.

Tested with Pipecat v1.13.0 and v1.15.0.

## [0.1.2]

First supported release. Earlier 0.1.x uploads were withdrawn before the
package was announced.

### Added

- `ChariotTTSService`, a streaming Pipecat `TTSService` over Chariot's
  text-to-speech WebSocket API.
- Optional `model` and `speed`, both omitted from the request when unset so
  the account default applies and a newly released model needs no upgrade.
  `GET /v1/voices` lists the speeds each voice supports under each model.
- Persistent connection reused across turns, with a keepalive that holds the
  session open through quiet stretches.
- Interruption handling through `InterruptibleTTSService`: closing the socket
  stops server side generation immediately.
- Runtime voice switching through `TTSUpdateSettingsFrame`.
- Segmentation of over long utterances at word boundaries, so a long reply is
  spoken as contiguous audio.
- A Chariot stock voice as the default, so the service runs with just an API key.
- Foundational voice example and unit tests.

Tested with Pipecat v1.10.0.

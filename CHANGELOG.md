# Changelog

All notable changes to `pipecat-chariot` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

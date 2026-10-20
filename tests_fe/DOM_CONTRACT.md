# SoundGuard frontend — DOM & behaviour contract

Reference for writing Playwright tests. Selectors are exact. Served at `/` from `static/index.html`,
logic in `static/soundguard-app.js`, constants in `static/soundguard-data.js`.

## Endpoints the frontend calls
- `POST /api/predict`  body `{sample, corr:{noise,volume,clip}}` → prediction dict
- `POST /api/predict/upload`  multipart `file`, form `noise/volume/clip` → prediction dict
- `GET  /api/dashboard` → `{reliability, riskCoverage, classAcc, metrics, ...}`
- Prediction dict keys: `probs[10]`, `order[[idx,prob]×10] desc`, `conf`, `entropy`, `snrDb`, `levelDb`, `clipPct`, `status∈{reliable,uncertain,reject}`, `cause∈{clean,uncert,noise,quiet,clip}`

## Tabs
- Buttons: `.tab[data-tab="inference"]`, `.tab[data-tab="corruption"]`, `.tab[data-tab="dashboard"]`; selected has `aria-selected="true"`.
- Panels: `#tab-inference`, `#tab-corruption`, `#tab-dashboard`; visible one has class `active` (`.tab-panel.active`).
- Clicking a tab toggles `aria-selected`, toggles panel `.active`, and re-runs `renderResult()`.

## Input bar (inference tab)
- `#btn-upload` opens the hidden `#file-input` (`accept="audio/*"`).
- `#btn-record` toggles recording; its label is `#btn-record .lbl` and names the action the next click performs (text "Record" / "Stop · Xs"); `aria-label` follows it ("Record from the microphone" / "Stop recording"); adds/removes class `recording`. Recording also ends on its own at a fixed cap.
- Sample chips: `.chip[data-sample="dog|siren|jack|horn|kids|gun|ac"]` (7). Selected chip has `aria-pressed="true"`. Default selected: `dog`.
- Now-playing text: `[data-nowplaying]` (there may be several; all get updated).

## Prediction render targets
- `#pred-body`: after a prediction contains `.top1 > .label` (class name, spaces not underscores) and `.top1 > .pct` (e.g. "91%"); `.top1-sub`; `.bars > .bar-row` ×3 (first has class `lead`), each with `.name`, `.bar-track > .bar-fill` (width %), `.bpct`.
- `#conf-body`: `.conf-num` (has status class), `.conf-meta`, `.gauge > .marker` (style.left = conf%), `.gauge-zones`, `.gauge-scale`.
- Status cards: `[data-status-card]` (one in inference right-col, one in corruption readout). Rendered by `renderStatus`: `card.className = "card status-card <status>"`; contains `.status-icon`, `.status-name` (e.g. "Reliable"/"Uncertain"/"Reject"), `.status-tag`, `.status-reason`, `.status-action`. `.status-tag` names the **cause**, not the state: `SG.causeTag[r.cause]` → "noise floor too close" / "level too low" / "samples clipped" / "model hesitating" / "input accepted". The recording axis can raise the verdict at full confidence, so a tag keyed to the state would read "high uncertainty" beside 100%.
- Canvases: `#wave-inf`, `#mel-inf` (inference), `#wave-cor`, `#mel-cor` (corruption). `<canvas>` 760×230/220.

## Corruption tab
- Sliders: `input[type=range][data-slider="noise"][data-group="cor"]`, likewise `volume`, `clip`. min=0 max=100 value=0. `input` event drives a debounced (80 ms) refresh.
- Value labels: `[data-sval="noise-cor"]`, `[data-sval="volume-cor"]`, `[data-sval="clip-cor"]` (text like "40%").
- Reset button: `[data-reset]` → sets all sliders to 0 and refreshes.
- Quality meters: `#qmeters` → `.qmeter.(ok|warn|bad)` ×4 each with `.qk` (SNR/Peak level/Clipped/Entropy) and `.qv`.
- Compare: `#conf-compare` → `.cc-cell.clean .big` and `.cc-cell .big` (+`.cc-delta`).

## Dashboard tab
- `#chart-reliability` → `<svg>` (bars + diagonal). `#chart-risk` → `<svg>` (paths + circles).
- `#metrics-table` → `<table.metrics>` with 4 rows (ECE/NLL/Brier/Accuracy), `.metric-note`.
- `#classacc-rows` → `.ca-row` ×10, each `.ca-name`, `.ca-track > .ca-fill.clean` + `.ca-fill.corrupt`, `.ca-vals`.
- On `/api/dashboard` failure → `showChartsUnavailable()` writes `.chart-empty` (containing a `<code>`) into all four containers `#chart-reliability #chart-risk #metrics-table #classacc-rows`.

## Error handling
- `fetchPredict`/`fetchUpload` throw on non-2xx; the message is the backend JSON `detail` when present.
- `showError(msg)` overwrites every `[data-status-card]` with `class="card status-card reject"`, `.status-name` text "Backend error", `.status-tag` "request failed", and `.status-reason` = the message.  ⚠️ `msg` is inserted via template-literal `innerHTML` — unsanitised.
- Upload/record errors and mic-denied (`"Microphone access denied or unavailable."`) go through `showError`.
- `nowPlaying(name, extra)` sets `[data-nowplaying]` innerHTML to `<b>${name}</b> · ${extra}` — ⚠️ `name` (the uploaded filename) is unsanitised.

## Init sequence (on DOMContentLoaded)
`loadDashboard()` (async) → then `bindTabs/bindInput/bindSliders`, set now-playing to the dog sample, `syncSliders()`, `refresh()`. The first `refresh()` fires two `POST /api/predict` (current + clean baseline).

## Notes for tests
- `page.route()` can fulfil/abort `**/api/predict`, `**/api/predict/upload`, `**/api/dashboard` to inject faults (503, 422 with `{detail}`, malformed JSON, NaN, slow, abort).
- To upload a file use `page.set_input_files("#file-input", ...)` or `FileChooser`; the chip/upload triggers a real POST.
- Debounce is 80 ms — after a slider change, wait for the network/text to settle.
- Real demo clips live in `<project>/samples/*.wav`; a real WAV for upload tests is `samples/dog.wav`.

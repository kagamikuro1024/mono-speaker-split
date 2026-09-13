# Benchmark

Runs the whole sample set through `monosplit.separate`, scores every case against the expectation
written in `cases.json`, prints a markdown table to the screen and writes `benchmark/report.json`.

Nothing is faked here: real models, real files, real clock.

## Running

```bash
MONOSPLIT_MODELS=$HOME/.cache/voice-diar \
    uv run --with-editable '.[asr]' python -m benchmark.run
```

Options:

| Flag | Meaning |
|---|---|
| `--audio DIR` | Take the samples from `DIR` instead of the paths written in `cases.json` (matched by **file name**). |
| `--cases FILE` | Use a different case set. Defaults to `benchmark/cases.json`. |
| `--no-asr` | Skip transcription. Much faster, but layer 2 loses its text evidence and has to guess the role from turn order. |
| `--models DIR` | Directory holding `seg.onnx` / `emb.onnx`. If not passed, `MONOSPLIT_MODELS` is used; without that either, the models are downloaded into `~/.cache/monosplit`. |
| `--out FILE` | Where to write the JSON report. Defaults to `benchmark/report.json`. |

Exit code `0` when every case passes, `1` when a case fails — enough to plug into CI if you have a
fixed sample set.

## Reading the table

| Column | Meaning |
|---|---|
| **Case** | File name. |
| **Mode** | `Result.mode`: `mono`, `stereo_trung_nhau` (the two channels are identical — duplicated mono), `stereo_mot_ben_cam` (one channel silent). |
| **Seconds** | Wall time for that case, decoding and transcription included. |
| **Turns** | Number of turns in the result. |
| **Caller/Agent** | Number of turns assigned to each role. A heavy skew to one side is a sign layer 1 under-collected. |
| **Margin** | Median cosine distance between the two voice embeddings, counting only the turns layer 3 actually scored. Bigger is safer; below `0.10` means the two voices are too close. |
| **Same voice** | `same_voice` — the two sides sound like one voice, so labels have to alternate turn by turn. |
| **Role evidence** | Layer 2's `role_reason`: why this cluster is taken to be the agent. A rejected case shows the error code instead. |
| **Result** | `PASS` / `FAIL` / `CORRECTLY REJECTED`. |

`CORRECTLY REJECTED` is a **success**, not a failure: the control case is a genuine two-channel
recording with the roles already separated, where guessing who is who is redundant and harmful, so
`separate` must raise `hai_kenh_that`.

The last line gives the pass rate and the **seconds of processing per minute of audio** (with
RTF = that number divided by 60). RTF `0.3` means a 10-minute call takes about 3 minutes of machine
time.

## Test cases

`cases.json` is a list of `{file, mo_ta, ky_vong}`:

```json
{
  "file": "/duong/dan/den/cuoc_goi.m4a",
  "mo_ta": "Real call, two speakers",
  "ky_vong": { "hai_vai": true, "so_luot_toi_thieu": 4 }
}
```

* `hai_vai: true` — both roles `caller` and `agent` must be separated out.
* `so_luot_toi_thieu` — a floor on the turn count, to catch the case where the whole call is
  collapsed into one or two huge turns.
* `loi: "hai_kenh_that"` — a control case: `separate` is expected to **reject** with exactly that
  code.

## Note: the audio files are NOT in the repo

Call recordings are real data about real people, not pushed to git. `cases.json` only keeps the
**paths** on the machine that ran the suite. On another machine those cases report
`file not found`.

To use it with your own samples: drop the files into a directory, then

```bash
python -m benchmark.run --audio /duong/dan/mau/cua/ban
```

(matched by file name), or copy `cases.json` elsewhere, edit `file` and `ky_vong` to match your own
set, and run with `--cases`.

The current set consists of ten real call-centre call excerpts (~105 seconds each, fake stereo: two
identical channels, two real speakers) and three genuine two-channel builds as control cases — they
must be rejected, because once there are two channels there is nothing to split.

## Second suite: words recovered from an overlap

`benchmark/run.py` scores role labelling. `benchmark/overlap.py` scores a different question: when
both people talk at once, do the words of both sides come back?

```bash
python -m benchmark.overlap ~/Downloads/benchmark-overlap-30 \
    --separator ~/.cache/voice-models/sepformer16k.onnx \
    --model large-v3 \
    --json /tmp/overlap-report.json
```

Every case needs per-side ground truth, so cases are built from TWO-channel sources: one role per
channel, the agent channel shifted earlier by X ms to create exactly X ms of overlap, then mixed
down to one channel. Shift the whole channel, never splice a single turn - a splice leaves a seam
and the separator locks onto the seam instead of the voice, which flatters the score. The manifest
schema is in the module docstring.

Result on 26 cases carrying ground truth (2026-09-13, SepFormer WHAMR 16 kHz + Whisper large-v3 on
the separated streams):

| Group | n | Improved |
|---|---|---|
| synthetic overlap 200 / 600 / 1200 ms | 18 | 17/18 |
| level mismatch -9 / -5 / +5 dB | 6 | 6/6 |
| control, no overlap | 2 | 1/2 |
| **all** - caller CER 0.65 -> 0.36, agent CER 0.78 -> 0.35 | **26** | **24/26** |

Read the delta, not the absolute CER: the reference is the whole turn while the scoring window only
covers the overlap plus padding, so text outside the window counts as missing on both sides.

The control group is the point of the 400 ms floor: with no overlap to separate, separation can only
make the words worse. The report JSON is gitignored - it is regenerated from the audio set.

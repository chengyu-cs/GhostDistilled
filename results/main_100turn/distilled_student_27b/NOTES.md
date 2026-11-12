# distilled student 27b — 100-turn protocol runs

Runs: replicate1, replicate2, replicate3. Scoring ruler: answer-keyed v2 (`../../evaluation_protocol/answer_key.json`).
Model: locally served 27B student (vGPU, 90-turn variant).

## replicate1

- `lyric_recognition`: `2/11`
- `len_var`: `96.3`
- `sass_rate`: `0.06`
- `memory_recall`: `5/5`
- `labels`: `100/100`
- wrong_song: `1`

## replicate2

- `lyric_recognition`: `1/11`
- `len_var`: `60.8`
- `sass_rate`: `0.01`
- `memory_recall`: `5/5`
- `labels`: `100/100`
- wrong_song: `0`

## replicate3

- `lyric_recognition`: `0/11`
- `len_var`: `40.9`
- `sass_rate`: `0.01`
- `memory_recall`: `5/5`
- `labels`: `100/100`
- wrong_song: `1`

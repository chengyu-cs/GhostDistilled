# sft student 14b — 100-turn protocol runs

Runs: replicate1, replicate2, replicate3, temperature_1.0. Scoring ruler: answer-keyed v2 (`../../evaluation_protocol/answer_key.json`).
Model: hosted 14B plain-SFT student, checkpoint qwen3-14b-35b623e73d9b (T=0.85; temperature_1.0/ = reruns at T=1.0).

## replicate1

- `lyric_recognition`: `2/11`
- `len_var`: `1455.7`
- `sass_rate`: `0.01`
- `memory_recall`: `3/5`
- `labels`: `100/100`
- wrong_song: `0`

## replicate2

- `lyric_recognition`: `2/11`
- `len_var`: `44.0`
- `sass_rate`: `0.01`
- `memory_recall`: `3/5`
- `labels`: `100/100`
- wrong_song: `0`

## replicate3

- `lyric_recognition`: `1/11`
- `len_var`: `263.8`
- `sass_rate`: `0.01`
- `memory_recall`: `5/5`
- `labels`: `100/100`
- wrong_song: `1`

## temperature_1.0

(scores.json 见此目录)

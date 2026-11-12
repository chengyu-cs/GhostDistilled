# distilled student 14b — 100-turn protocol runs

Runs: replicate1, replicate2, replicate3, temperature_1.0. Scoring ruler: answer-keyed v2 (`../../evaluation_protocol/answer_key.json`).
Model: hosted 14B QLoRA student, checkpoint qwen3-14b-1dc235b6bcee (T=0.85; temperature_1.0/ = reruns at T=1.0).

## replicate1

- `lyric_recognition`: `3/11`
- `len_var`: `51.5`
- `sass_rate`: `0.0`
- `memory_recall`: `5/5`
- `labels`: `100/100`
- wrong_song: `4`

## replicate2

- `lyric_recognition`: `4/11`
- `len_var`: `116.3`
- `sass_rate`: `0.01`
- `memory_recall`: `5/5`
- `labels`: `100/100`
- wrong_song: `1`

## replicate3

- `lyric_recognition`: `1/11`
- `len_var`: `49.2`
- `sass_rate`: `0.0`
- `memory_recall`: `5/5`
- `labels`: `100/100`
- wrong_song: `1`

## temperature_1.0

(scores.json 见此目录)

# retrieval fewshot — 100-turn protocol runs

Runs: replicate1, replicate2. Scoring ruler: answer-keyed v2 (`../../evaluation_protocol/answer_key.json`).
Model: same LLM + corpus top-5 exemplars (T=1.0).

## replicate1

- `lyric_recognition`: `6/11`
- `len_var`: `150.7`
- `sass_rate`: `0.01`
- `memory_recall`: `5/5`
- `labels`: `100/100`
- wrong_song: `1`

## replicate2

- `lyric_recognition`: `1/20`
- `len_var`: `11.0`
- `sass_rate`: `0.0`
- `memory_recall`: `4/5`
- `labels`: `5/5`

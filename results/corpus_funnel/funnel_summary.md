# Corpus funnel

Reproducible end to end with `../corpus/make_funnel_counts.py` (October 2026 run).

| Stage | Count | Note |
|---|---:|---|
| Raw attribution pool | 44,641 | 5+1 platforms, 2014--2026 |
| Cleaned pool | 26,062 | dedup, language, speaker (subset of raw pool, verified by id containment) |
| Kept by safety filter | 20,963 | attributed and releasable |
| Filtered out | 5,099 | concentrated in 2024 (2,798) / 2020 (808); 5,002 from the video platform |
| Later collection | +13,378 | ASR 11,522 + three additional platforms 1,554 + official captions 261 + live-interview paraphrases 41 |
| **Released master pool** | **34,341** | = kept + expansion; three official layers: public_safe_all 33,863, 2020_public_safe 437, live_curated 41 |
| Dialogue band (8--80 chars) | 19,837 | spoken 17,175 / written 1,743 / caption 919 |
| Downstream use | 1,186 / 2,931 | SFT sample counts (plain-SFT / distilled); evaluation 100 turns; anchor sampling 997 |

Every containment relation above is verified by set arithmetic on stable item identifiers by the script.
Machine-readable counts: `funnel_counts.json`.

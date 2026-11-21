# GTA VI trophy rarity forecast — make.fun bounty entry

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Tianchen0012/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Tianchen0012/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PlayStation 5 trophy's Exophase earn rate at
2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%).

## Program

`main.py` (Python 3.13, standard library only, no network) reads every
`<id>.in` case in `$MAKEFUN_INPUT_DIR` (else `/input/`) and writes
`<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output/`), deterministic
and well under the 8-second limit.

## Model

Per trophy, in logit space: a weighted least-squares fit of the earn
rate against `ln(1 + hours since release)` over the last 24 hours of
distinct reads, slope shrunk toward its trophy class's prior, anchored
at the smoothed level of the last read. A concave quadratic fit damps
the extrapolation where the data support it, because earn-rate curves
flatten as the player population matures. The fitted forecast is
blended with a research-based prior (each trophy's week-one rate,
interpolated from its last read by horizon). Spreads combine a
random-walk term, the fit residual, and a refresh-jump term that is
large when the history is flat or stale and shrinks as distinct reads
accumulate.

Grounded in public launch reporting (Nov 2026): ~25M copies day one
(75% on PS5); story 60–80h, platinum ~200h, so nobody can hold the
platinum or 100% completion by Nov 26 (medians 0.01% / 0.02%); "Gone
Fishing" is the opening mission every tracked player has (pinned near
100% — it survived a 6x jump in tracked players at 100%); chapter
trophies rise fastest, grinds crawl.

## Local check

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Validated on the kit's practice cases `p00` / `p01`: valid output,
55 rows, byte-identical across runs, ~0.1s per case.

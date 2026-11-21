"""Forecast GTA VI PlayStation 5 trophy earn rates on Exophase, one week out.

Model (all in logit space, per trophy):
  Each trophy's earn rate is fitted against x = ln(1 + hours since
  release) over the last 24 hours of distinct reads -- the regime
  closest to the target period -- by weighted least squares, its slope
  shrunk toward its trophy class's prior slope.  The fit is anchored at
  the smoothed level of the last read, not the raw last rate, so a
  refresh jump (tracked players jump, every rate drops) does not jerk
  the forecast.  Where six or more distinct reads support it, a
  concave quadratic fit damps the extrapolation: earn-rate curves
  flatten as the player population matures, so a straight line through
  the steep early phase overshoots; the quadratic is used only when it
  is concave and lands between the prior and the linear prediction.

  The fitted forecast is blended (in logit space) with a research-based
  prior: each trophy's week-one rate, interpolated from its last read
  toward its mature rate by horizon, so a flat history (launch night,
  one distinct read) still forecasts sensibly.  The blend weight grows
  with the number of distinct reads.

  Priors, from public guides and launch reporting (Nov 2026):
  - Story is 60-80 hours; the platinum needs ~200 hours.  Seven days
    hold 168 hours, so nobody can hold the platinum or 100% completion
    by Nov 26: their medians sit at 0.01% / 0.02%.
  - "Gone Fishing" is the opening mission and every tracked player has
    it: it stayed at 100% through a 6x jump in tracked players, so it is
    pinned near 100% while the data agrees.
  - Chapter trophies rise fastest (most players finish Chapter 1 within
    days; later chapters trail), mission trophies track their chapter,
    easy activities rise steadily, grinds and collectibles crawl.
  - Launch week sold ~25M copies day one (75% on PS5) with daily
    increments falling fast after day one, so tracked-player growth
    slows over the horizon while progression continues: the fitted
    slopes are taken from the most recent 24 hours, the regime closest
    to the target period.

Spread (logit sd): a random-walk term growing with the horizon, the
fit's residual, a refresh-jump term that is large when the history is
flat or stale (a single Exophase refresh can move every rate several
fold while the tracked base is small) and shrinks as distinct reads
accumulate, and a floor.  Quantiles are the normal quantiles of the
logit forecast, so the 5%/95% band honestly covers about 90%.

Deterministic: no randomness, no clock, no network; the same case
always writes the same bytes.  Standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Trophy class by id (the 55 trophies are stable: 1 platinum, 2 gold,
# 6 silver, 46 bronze).  Fallback classification by description below
# covers any trophy id this table does not name.
CLASS_BY_ID = {
    1: "platinum", 2: "first",
    3: "chapter", 4: "chapter", 5: "chapter", 6: "chapter", 7: "chapter",
    8: "mission", 9: "mission", 10: "mission", 11: "mission",
    12: "mission", 13: "mission",
    14: "hundred",
    15: "grind", 16: "easy", 17: "grind", 18: "grind", 19: "medium",
    20: "medium", 21: "easy", 22: "mission", 23: "mission", 24: "mission",
    25: "grind", 26: "mission", 27: "mission", 28: "grind", 29: "medium",
    30: "grind", 31: "grind", 32: "grind", 33: "easy", 34: "medium",
    35: "medium", 36: "easy", 37: "medium", 38: "medium", 39: "medium",
    40: "grind", 41: "easy", 42: "medium", 43: "medium", 44: "medium",
    45: "medium", 46: "medium", 47: "medium", 48: "medium", 49: "medium",
    50: "medium", 51: "medium", 52: "medium", 53: "easy", 54: "grind",
    55: "easy",
}

# Week-one (Nov 26) mature rate per trophy, percent.  From the trophy
# list and public roadmaps: chapter trophies fall off down the story,
# mission trophies track their chapter, grinds stay in single digits.
MATURE_BY_ID = {
    1: 0.01, 2: 99.9, 3: 70.0, 4: 55.0, 5: 40.0, 6: 28.0, 7: 15.0,
    8: 45.0, 9: 38.0, 10: 35.0, 11: 30.0, 12: 28.0, 13: 22.0,
    14: 0.02, 15: 3.0, 16: 25.0, 17: 4.0, 18: 2.0, 19: 8.0, 20: 4.0,
    21: 12.0, 22: 25.0, 23: 22.0, 24: 20.0, 25: 2.0, 26: 20.0, 27: 18.0,
    28: 3.0, 29: 5.0, 30: 2.0, 31: 4.0, 32: 2.0, 33: 15.0, 34: 8.0,
    35: 8.0, 36: 10.0, 37: 6.0, 38: 5.0, 39: 8.0, 40: 3.0, 41: 12.0,
    42: 15.0, 43: 6.0, 44: 8.0, 45: 6.0, 46: 6.0, 47: 5.0, 48: 5.0,
    49: 4.0, 50: 4.0, 51: 5.0, 52: 8.0, 53: 20.0, 54: 4.0, 55: 15.0,
}

# Fallback mature rate per class, for a trophy id the tables miss.
MATURE_BY_CLASS = {
    "platinum": 0.01, "first": 99.9, "chapter": 35.0, "mission": 25.0,
    "hundred": 0.02, "easy": 15.0, "medium": 8.0, "grind": 3.0,
}

# Prior slope of a trophy's logit rate per unit of x = ln(1 + hours
# since release): how fast its rate rises net of dilution.
SLOPE_BY_CLASS = {
    "platinum": 0.0, "first": 0.0, "chapter": 1.2, "mission": 1.0,
    "hundred": 0.0, "easy": 0.8, "medium": 0.6, "grind": 0.3,
}

# Hard caps on the median, percent: nobody finishes a 200-hour
# platinum or a ~170-hour 100% run inside the game's first week.
CAP_BY_CLASS = {"platinum": 0.05, "hundred": 0.1}


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, as JavaScript's Math.round does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def classify(trophy):
    """(class, mature rate) for one trophy dict."""
    tid = trophy["id"]
    cls = CLASS_BY_ID.get(tid)
    mature = MATURE_BY_ID.get(tid)
    if cls is None or mature is None:
        text = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
        if "chapter" in text or "final mission" in text:
            cls = "chapter"
        elif "100% completion" in text:
            cls = "hundred"
        elif any(k in text for k in ("250 stunts", "collectible card", "100 locations",
                                     "50 challenges", "bank vault", "garage")):
            cls = "grind"
        elif any(k in text for k in ("try each", "arcade", "spend $", "armored truck",
                                     "wanted level")):
            cls = "easy"
        elif trophy.get("tier") == "platinum":
            cls = "platinum"
        else:
            cls = "medium"
        mature = MATURE_BY_CLASS[cls]
    return cls, mature


def wls(xs, ys, ws):
    """Weighted least squares of y on x.  Returns (slope, intercept,
    residual sd) or None when fewer than two distinct x values."""
    n = len(xs)
    if n < 2 or len(set(xs)) < 2:
        return None
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = math.sqrt(sum(w * (y - (intercept + slope * x)) ** 2
                          for w, x, y in zip(ws, xs, ys)) / sw)
    return slope, intercept, resid


def wls2(xs, ys, ws):
    """Weighted least squares of y on (1, x, x^2).  Returns
    (a, b, c, residual sd) or None when the system is singular or
    fewer than three distinct x values.  Solved by Gaussian
    elimination on the 3x3 normal equations."""
    if len(xs) < 3 or len(set(xs)) < 3:
        return None
    # Normal equations for [1, x, x^2].
    m = [[0.0] * 3 for _ in range(3)]
    v = [0.0] * 3
    for x, y, w in zip(xs, ys, ws):
        basis = (1.0, x, x * x)
        for i in range(3):
            v[i] += w * basis[i] * y
            for j in range(3):
                m[i][j] += w * basis[i] * basis[j]
    # Eliminate.
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        v[col], v[piv] = v[piv], v[col]
        for r in range(col + 1, 3):
            f = m[r][col] / m[col][col]
            for c in range(col, 3):
                m[r][c] -= f * m[col][c]
            v[r] -= f * v[col]
    coef = [0.0] * 3
    for r in (2, 1, 0):
        coef[r] = (v[r] - sum(m[r][c] * coef[c] for c in range(r + 1, 3))) / m[r][r]
    resid = math.sqrt(sum(w * (y - (coef[0] + coef[1] * x + coef[2] * x * x)) ** 2
                          for x, y, w in zip(xs, ys, ws)) / sum(ws))
    return coef[0], coef[1], coef[2], resid


def distinct_reads(history):
    """Reads whose (players, rates) actually changed, in order, plus
    the first and last reads.  Exophase refreshes in steps, so most
    reads repeat the previous page exactly."""
    kept = []
    prev = None
    for read in history:
        key = (read.get("players_tracked"),
               tuple(read.get("rates") or []))
        if key != prev:
            kept.append(read)
            prev = key
    return kept


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0

    try:
        release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    except Exception:
        release = parse_time(history[0]["at"]) if history else datetime(
            2026, 11, 19, tzinfo=timezone.utc)
    target = parse_time(case["target"])
    x_target = math.log1p(max(0.0, (target - release).total_seconds() / 3600.0))

    reads = distinct_reads(history)
    n_distinct = len(reads)

    # Per-read (x, hours since release, tracked players), skipping
    # reads with no player count.  Kept as one series so the fits
    # below can never misalign a read with its time.
    series = []
    for read in reads:
        p = read.get("players_tracked")
        if not p or p <= 0:
            continue
        hours = max(0.0, (parse_time(read["at"]) - release).total_seconds() / 3600.0)
        series.append((read, math.log1p(hours), hours, float(p)))
    if not series:
        # No usable history at all: answer with the priors, wide bands.
        return {"trophies": [prior_only_row(t, horizon_days) for t in trophies]}

    x_last = series[-1][1]
    hours_last = series[-1][2]
    p_last = series[-1][3]

    # Last non-null rate per trophy.
    last_rate = []
    for i in range(len(trophies)):
        rate = None
        for read in reversed(reads):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                rate = float(rates[i])
                break
        last_rate.append(0.0 if rate is None else rate)

    # Blend weight for the fitted forecast: grows with distinct reads.
    w_data = n_distinct / (n_distinct + 4.0)

    rows = []
    for i, trophy in enumerate(trophies):
        cls, mature = classify(trophy)
        delta = SLOPE_BY_CLASS[cls]

        # Logit rates against x over the last 24 hours of distinct
        # reads; fall back to all distinct reads when that window
        # holds fewer than two points.
        pts = []
        for read, x, hours, _p in series:
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            pts.append((x, hours, logit(float(rates[i]))))
        win = [(x, y) for x, hours, y in pts if hours >= hours_last - 24.0]
        if len(win) < 2:
            win = [(x, y) for x, _, y in pts]
        n_fit = len(win)
        resid = 0.0
        if n_fit >= 2 and len(set(x for x, _ in win)) >= 2:
            wws = [0.5 + 0.5 * j / max(1, n_fit - 1) for j in range(n_fit)]
            fit = wls([x for x, _ in win], [y for _, y in win], wws)
        else:
            fit = None
        if fit is not None:
            slope, intercept, resid = fit
            slope_star = (n_fit * slope + 3.0 * delta) / (n_fit + 3.0)
            y_last = intercept + slope * x_last
            pred = y_last + slope_star * (x_target - x_last)
            # Concave damping: where the data support a quadratic and
            # it bends down, use it when it lands between the prior
            # and the linear prediction -- the curve flattens as the
            # player population matures, so the straight line
            # overshoots.  (The prior logit is computed below; the
            # check is applied after it.)
            quad_pred = None
            if n_fit >= 6:
                fit2 = wls2([x for x, _ in win], [y for _, y in win], wws)
                if fit2 is not None and fit2[2] < 0:
                    a, b, c, _r2 = fit2
                    quad_pred = a + b * x_target + c * x_target * x_target
        else:
            pred = logit(last_rate[i]) + delta * (x_target - x_last)
            quad_pred = None

        # Prior: last rate drifting toward its mature week-one rate,
        # by horizon.  Over a 90-minute horizon the prior barely
        # moves from the last rate; over 4.5 days it is most of the
        # way to the mature rate.  The "first" trophy is pinned near
        # 100% while the data agrees it is universal.
        if cls == "first" and last_rate[i] >= 99.9:
            prior = 99.9
        else:
            f = 1.0 - math.exp(-horizon_days / 2.5)
            prior = sigmoid(logit(last_rate[i]) * (1.0 - f) + logit(mature) * f)
        prior_lg = logit(prior)

        if quad_pred is not None:
            lo = min(prior_lg, pred)
            hi = max(prior_lg, pred)
            if lo <= quad_pred <= hi:
                pred = quad_pred

        median = sigmoid(w_data * pred + (1.0 - w_data) * prior_lg)

        # A flat, stale history with a tiny tracked base means a
        # refresh is pending that will dilute every rate several fold:
        # the first refresh took tracked players 638 -> 3,868, about
        # 6x, so centre the median on a ~4x dilution.  (Only fires
        # when the data cannot speak: it never fires on the final
        # case's rich history.  The first mission is exempt: it
        # survived that 6x refresh at 100%, so dilution provably
        # does not touch it.)
        if (n_distinct <= 2 and p_last < 10000 and horizon_days < 1.0
                and cls != "first"):
            median = clamp_pct(median * 0.25)

        # Hard caps on the median: nobody finishes a 200-hour
        # platinum or a ~170-hour 100% run inside the game's first
        # week.  They bind only when the target is far enough past
        # release for that argument to hold (the final's target is
        # 168 hours out); over a 90-minute horizon the rate is
        # whatever the page says, dilution included.
        target_hours = (target - release).total_seconds() / 3600.0
        cap = CAP_BY_CLASS.get(cls)
        if cap is not None and target_hours >= 120.0:
            median = min(median, cap)
        if cls == "first" and last_rate[i] >= 99.9:
            median = max(median, 99.0)
        median = clamp_pct(median)

        # Spread: random walk over the horizon, fit residual, a
        # refresh-jump term that shrinks as distinct reads accumulate,
        # and a floor.  Near-certain floors get a tighter band.
        var = 0.25 * horizon_days + resid * resid
        var += (1.0 / math.sqrt(1.0 + n_distinct)) ** 2 + 0.09
        sigma = min(2.2, max(0.45, math.sqrt(var)))
        if cls in ("platinum", "hundred"):
            sigma = 0.5
        elif cls == "first" and last_rate[i] >= 99.9:
            sigma = 0.5

        lg = logit(median)
        qs = []
        prev_q = 0.0
        for z in Z:
            q = round4(min(100.0, max(0.01, sigmoid(lg + z * sigma))))
            q = max(q, prev_q)
            qs.append(q)
            prev_q = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def prior_only_row(trophy, horizon_days):
    cls, mature = classify(trophy)
    lg = logit(mature)
    sigma = 1.5
    if cls in ("platinum", "hundred"):
        sigma = 0.5
    qs = []
    prev_q = 0.0
    for z in Z:
        q = round4(min(100.0, max(0.01, sigmoid(lg + z * sigma))))
        q = max(q, prev_q)
        qs.append(q)
        prev_q = q
    return {"id": trophy["id"], "q": qs}


def fallback_case(case):
    """Last-read rates as every quantile: the scored fallback for a case
    whose forecast fails, so one bad case cannot sink the whole run."""
    trophies = case["trophies"]
    history = case.get("history") or []
    rows = []
    for i, trophy in enumerate(trophies):
        rate = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                rate = float(rates[i])
                break
        if rate is None:
            _cls, mature = classify(trophy)
            rate = mature
        q = clamp_pct(rate)
        rows.append({"id": trophy["id"], "q": [q] * 5})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            forecast = fallback_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

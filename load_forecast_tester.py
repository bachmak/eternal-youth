# forecast_test_pipeline.py
# Zweck: Load-Forecast zuverlässig testen (Interface + Robustheit + einfache Plausibilität)
# Output: Konsole (PASS/FAIL + Stats + Timing + optional Beispiele)
#
# Nutzung:
#   python forecast_test_pipeline.py
#
# Erwartet:
# - data_parser.batch_collect verfügbar
# - load_forecast_profile.predict_load_horizon verfügbar
# - Daten wie in eurer pipeline (Spalten: time, pv, consumption, soc, ...)
#
# Hinweis: Dieser Tester ändert eure echte pipeline NICHT.

import os.path
import time
import numpy as np
import pandas as pd

from data_parser import batch_collect
from load_forecast_profile import predict_load_horizon


class Config:
    HISTORY_SAMPLES = 7 * 288
    HORIZON_SAMPLES = 288

    # Wie viele idx-Schritte testen (ab warmup)?
    TEST_STEPS = 5000

    # Wie oft Fortschritt drucken
    PRINT_EVERY = 200

    # Wieviele Beispiel-Forecasts am Ende anzeigen (nur Werte, kein Plot)
    SHOW_EXAMPLES = 3

    # Optional: streng prüfen, ob forecast[0] vor dem Overwrite plausibel ist
    CHECK_FIRST_ELEMENT = True
    MAX_FIRST_ELEMENT_REL_ERR = 1.0  # 100% relative Abweichung tolerated (nur grob)

    # Numeric coercion (empfohlen)
    COERCE_NUMERIC = True


def get_table(df_cache: str, data_folder: str) -> pd.DataFrame:
    if os.path.exists(df_cache):
        return pd.read_csv(df_cache)

    df = batch_collect(data_folder)
    df["soc"] = df["soc"] / 100.0

    df["import"] = (
        df["consumption"] -
        df["pv_consumption"] + df["charge"] -
        df["discharge"]
    ).clip(lower=0)

    df["export"] = (df["pv"] - df["pv_consumption"]).clip(lower=0)

    df.to_csv(df_cache, index=False)
    return df


def predict_values(df: pd.DataFrame, curr_idx: int, history_size: int, horizon_size: int) -> np.ndarray:
    return predict_load_horizon(
        df=df,
        curr_idx=int(curr_idx),
        horizon_size=int(horizon_size),
        history_size=int(history_size),
        load_col="consumption",
        pv_col="pv",
        mode="auto",
        # time_col="time",  # nur setzen, wenn du es erzwingen willst
    )


def _as_float_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr


def main():
    cfg = Config()
    df = get_table("out/df.csv", "data/days-range")

    print("\n=== LOAD FORECAST TEST PIPELINE ===")
    print("DF shape:", df.shape)
    print("Columns:", list(df.columns))
    if "time" in df.columns:
        print("time dtype:", df["time"].dtype)
        print("time head:", df["time"].head(2).tolist())
    else:
        print("[WARN] No 'time' column found (OK if forecast uses synthetic index).")

    required_cols = ["consumption", "pv"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for forecast: {missing}")

    if cfg.COERCE_NUMERIC:
        for c in ["consumption", "pv"]:
            before_nan = int(pd.isna(df[c]).sum())
            df[c] = pd.to_numeric(df[c], errors="coerce")
            after_nan = int(pd.isna(df[c]).sum())
            if after_nan > before_nan:
                print(f"[WARN] Coercion introduced NaNs in '{c}': {before_nan} -> {after_nan}")

    # Warmup: ab hier soll Forecast stabil laufen
    warmup = cfg.HISTORY_SAMPLES
    if len(df) < warmup + cfg.HORIZON_SAMPLES + 2:
        raise ValueError("Dataset too short for configured warmup+horizon.")

    n_steps = min(cfg.TEST_STEPS, len(df) - warmup - cfg.HORIZON_SAMPLES - 1)
    print(f"Warmup starts at idx={warmup}")
    print(f"Testing n_steps={n_steps} (idx range {warmup}..{warmup + n_steps - 1})")
    print("HISTORY_SAMPLES:", cfg.HISTORY_SAMPLES, "| HORIZON_SAMPLES:", cfg.HORIZON_SAMPLES)
    print("-----------------------------------")

    # Stats collector
    failures = 0
    first_elem_warnings = 0

    nan_count_total = 0
    neg_count_total = 0
    wrong_len_total = 0
    exception_total = 0

    # Timing
    t_call = []
    # Keep a few examples
    examples = []

    for i in range(n_steps):
        idx = warmup + i

        t0 = time.perf_counter()
        try:
            forecast = predict_values(df, idx, cfg.HISTORY_SAMPLES, cfg.HORIZON_SAMPLES)
            forecast = _as_float_array(forecast)
            ok_exception = True
        except Exception as e:
            ok_exception = False
            exception_total += 1
            failures += 1
            if (i % cfg.PRINT_EVERY) == 0 or failures <= 5:
                print(f"[FAIL] idx={idx} threw exception: {type(e).__name__}: {e}")
            continue
        finally:
            t1 = time.perf_counter()
            if ok_exception:
                t_call.append((t1 - t0) * 1000.0)  # ms

        # Check length
        if len(forecast) != cfg.HORIZON_SAMPLES:
            wrong_len_total += 1
            failures += 1
            if wrong_len_total <= 5:
                print(f"[FAIL] idx={idx} wrong length: {len(forecast)} != {cfg.HORIZON_SAMPLES}")
            continue

        # Check finite
        finite_mask = np.isfinite(forecast)
        nan_count = int(np.size(forecast) - np.sum(finite_mask))
        nan_count_total += nan_count
        if nan_count > 0:
            failures += 1
            if nan_count_total <= 50:
                bad_pos = np.where(~finite_mask)[0][:10].tolist()
                print(f"[FAIL] idx={idx} contains non-finite values. count={nan_count}, first_pos={bad_pos}")
            continue

        # Check nonnegative (if your forecast clips nonnegative, this should always pass)
        neg_count = int(np.sum(forecast < -1e-12))
        neg_count_total += neg_count
        if neg_count > 0:
            failures += 1
            if neg_count_total <= 50:
                minv = float(np.min(forecast))
                print(f"[FAIL] idx={idx} contains negative values. neg_count={neg_count}, min={minv:.6f}")
            continue

        # Optional: check forecast[0] plausibility *before* overwrite
        if cfg.CHECK_FIRST_ELEMENT:
            y0 = float(df["consumption"].iloc[idx])
            if np.isfinite(y0) and y0 > 1e-6:
                rel_err = abs(float(forecast[0]) - y0) / y0
                if rel_err > cfg.MAX_FIRST_ELEMENT_REL_ERR:
                    first_elem_warnings += 1
                    # Warning only; not a failure, because pipeline overwrites anyway
                    if first_elem_warnings <= 5:
                        print(f"[WARN] idx={idx} forecast[0] far from actual. "
                              f"forecast0={forecast[0]:.3f}, actual={y0:.3f}, rel_err={rel_err:.2f}")

        # Save some examples
        if len(examples) < cfg.SHOW_EXAMPLES:
            examples.append((idx, forecast.copy()))

        # Progress print
        if (i % cfg.PRINT_EVERY) == 0:
            meanv = float(np.mean(forecast))
            maxv = float(np.max(forecast))
            minv = float(np.min(forecast))
            med_ms = float(np.median(t_call)) if t_call else float("nan")
            print(f"[OK] step={i}/{n_steps} idx={idx} "
                  f"pred[min/mean/max]={minv:.3f}/{meanv:.3f}/{maxv:.3f} "
                  f"median_call_ms={med_ms:.2f}")

    # Final report
    print("\n=== REPORT ===")
    print(f"Total tested steps: {n_steps}")
    print(f"Failures (hard): {failures}")
    print(f"  Exceptions: {exception_total}")
    print(f"  Wrong length: {wrong_len_total}")
    print(f"  Non-finite values total: {nan_count_total}")
    print(f"  Negative values total: {neg_count_total}")
    if cfg.CHECK_FIRST_ELEMENT:
        print(f"First-element warnings (not failures): {first_elem_warnings}")

    if t_call:
        print("\n=== TIMING (ms per forecast call) ===")
        print(f"calls: {len(t_call)}")
        print(f"mean:   {np.mean(t_call):.2f} ms")
        print(f"median: {np.median(t_call):.2f} ms")
        print(f"p95:    {np.percentile(t_call, 95):.2f} ms")
        print(f"max:    {np.max(t_call):.2f} ms")

    # Examples
    if examples:
        print("\n=== EXAMPLES (first 10 values of forecast) ===")
        for idx, fc in examples:
            y0 = float(df["consumption"].iloc[idx])
            print(f"- idx={idx} actual(consumption)={y0:.3f} | forecast[:10]={np.round(fc[:10], 3).tolist()}")

    # Strong "confidence" banner
    if failures == 0:
        print("\n✅ PASS: Load forecast produced correct length + finite + nonnegative outputs for all tested steps.")
    else:
        print("\n❌ FAIL: See above failures. Load forecast is not yet fully robust under this test run.")


if __name__ == "__main__":
    main()

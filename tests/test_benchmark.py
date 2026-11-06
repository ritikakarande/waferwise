import numpy as np

from waferwise.benchmark import summarise_case, summarise_run


def test_summary_skips_warmup_and_short_final_batch():
    timings = np.array([
        [4, 1.0, 100.0, 1000.0],  # warm-up
        [4, 1.0, 2.0, 4.0],
        [4, 1.0, 2.0, 4.0],
        [1, 1.0, 50.0, 51.0],  # short final batch
    ])

    summary = summarise_run(timings, warmup_seconds=1.0)

    assert summary["latency_p50_ms"] == 2.0
    assert summary["latency_p99_ms"] == 2.0
    assert summary["inference_wafers_per_s"] == 2000.0
    assert summary["end_to_end_wafers_per_s"] == 1000.0


def test_case_summary_keeps_median_and_range():
    runs = [{"speed": 1.0}, {"speed": 3.0}, {"speed": 2.0}]

    assert summarise_case(runs) == {"speed": {"median": 2.0, "min": 1.0, "max": 3.0}}

"""
GeoCapt-UTRP Demo — Headless Automatizer
========================================

Runs the same Capt-Temp pipeline that `app.py` runs, but headless and in a
loop. Each iteration uses a different parameter combination, all with:

    * search-related parameters maxed out (max_iter=10, initial_count=10000,
      pricing_count=1000, ls_rounds=20, time_limit=1200 s)
    * min_routes = 5, max_routes = 12

Results (HTML dashboard + JSON metrics) are written to ``results/`` and a
running summary is appended to ``results/runs.csv`` so you can compare
adéquation and ``dun`` (non-desservi) across runs.

Press Ctrl+C at any time to stop cleanly.

Usage:
    python automate.py                  # infinite loop, random combos
    python automate.py --runs 20        # stop after 20 runs
    python automate.py --grid           # walk full grid then shuffle and repeat
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Stub Streamlit so we can import app.py without a Streamlit runtime.
# Must happen BEFORE any "import app" statement or anything that imports
# streamlit transitively.
# ---------------------------------------------------------------------------
import sys
import types


class _StreamlitStub:
    """Returns itself for any attribute access / call so app.py imports
    cleanly. Comparison falls back to identity-not-equal, so the chain of
    `if step == "...": elif step == "...":` branches all skip."""

    def __getattr__(self, name):
        return _StreamlitStub()

    def __call__(self, *args, **kwargs):
        return _StreamlitStub()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __bool__(self):
        return False


_stub = _StreamlitStub()
sys.modules.setdefault("streamlit", _stub)
_components = types.ModuleType("streamlit.components")
_components.v1 = _stub
sys.modules.setdefault("streamlit.components", _components)
sys.modules.setdefault("streamlit.components.v1", _stub)


# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------
import argparse
import csv
import hashlib
import itertools
import json
import random
import signal
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
DATA_DIR = DEMO_DIR / "data"
GTFS_DIR = DATA_DIR / "GTFS"
POM_CSV = DATA_DIR / "pom_poissy.csv"
RESULTS_DIR = DEMO_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CAPT_TEMP_DIR = str(DEMO_DIR.parent / "Capt-Temp")
if CAPT_TEMP_DIR not in sys.path:
    sys.path.insert(0, CAPT_TEMP_DIR)


# ---------------------------------------------------------------------------
# Pull the dashboard HTML helper out of app.py (Streamlit is stubbed above)
# ---------------------------------------------------------------------------
import app as _app  # noqa: E402

generate_dashboard_html = _app.generate_dashboard_html


# ---------------------------------------------------------------------------
# Pipeline imports (must come AFTER sys.path is patched)
# ---------------------------------------------------------------------------
import config as cfg  # noqa: E402

from geocapt_loader import load_all  # noqa: E402
from geocapt_colgen import colgen_loop_joint  # noqa: E402
from geocapt_gurobi import evaluate_full, adequation_report  # noqa: E402
from geocapt_localsearch import local_search_joint  # noqa: E402
from geocapt_route_quality import build_node_coords  # noqa: E402
from geocapt_routes import _axis_sort_route  # noqa: E402


# ---------------------------------------------------------------------------
# Constants — these match the maxes exposed in the Streamlit sidebar.
# ---------------------------------------------------------------------------
ALL_BANDS = ["pot_hp", "pot_hc", "pot_soir", "pot_nuit"]
PERIOD_WEIGHTS = {
    "pot_hp": 0.40,
    "pot_hc": 0.30,
    "pot_soir": 0.20,
    "pot_nuit": 0.10,
}

MAX_PARAMS = {
    "min_routes": 5,       # solver may pick anywhere from 5 to 12 routes
    "max_routes": 12,
    "time_limit": 1200,    # per Gurobi solve
    "max_iter": 7,         # column-generation iterations
    "initial_count": 10000,  # number_input max
    "pricing_count": 1000,
    "ls_rounds": 20,       # slider max
}

# Model-option flags we always keep ON — the goal is to find good
# alpha_pass / alpha_adeq / alpha_oper, not to compare model variants.
FIXED_FLAGS = {
    "use_new_od":   True,
    "route_choice": True,
    "freq_opt":     True,
    "route_rules":  True,
}

# Search grid for the OBJECTIVE WEIGHTS we want to sweep.
GRID = {
    "alpha_pass":  [0.5, 1.0, 2.0, 5.0],
    "alpha_adeq":  [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    "alpha_oper":  [0.005, 0.01, 0.05],
}


# ---------------------------------------------------------------------------
# Graceful Ctrl+C
# ---------------------------------------------------------------------------
_STOP = False


def _on_sigint(signum, frame):
    global _STOP
    _STOP = True
    print("\n[automate] Stop requested — finishing current run then exiting…")


signal.signal(signal.SIGINT, _on_sigint)


# ---------------------------------------------------------------------------
# Combination generation
# ---------------------------------------------------------------------------

def _all_combos():
    keys = list(GRID.keys())
    for vals in itertools.product(*[GRID[k] for k in keys]):
        yield dict(zip(keys, vals))


def combo_iter(mode: str, seed: int):
    """Yields parameter combinations indefinitely (until _STOP).

    Each yielded dict is the swept GRID values merged with FIXED_FLAGS so
    every run keeps use_new_od / route_choice / freq_opt / route_rules ON.
    """
    rng = random.Random(seed)
    if mode == "grid":
        combos = list(_all_combos())
        while True:
            rng.shuffle(combos)
            for c in combos:
                yield {**FIXED_FLAGS, **c}
    else:  # random
        while True:
            c = {k: rng.choice(v) for k, v in GRID.items()}
            yield {**FIXED_FLAGS, **c}


def combo_hash(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def run_one(params: dict, all_data: dict, G, nodes, prox, points,
            run_id: str, log_path: Path) -> dict:
    """Run the joint optimisation once with the given parameters.

    Mirrors the STEP-4 logic of app.py but writes outputs to disk."""

    sample = all_data[ALL_BANDS[0]]
    is_gtfs = sample.get("mode") == "gtfs_osm"
    route_gen_G = sample.get("route_gen_G")
    min_rl = cfg.GTFS_MIN_ROUTE_NODES if is_gtfs else cfg.MIN_ROUTE_NODES
    max_rl = cfg.GTFS_MAX_ROUTE_NODES if is_gtfs else cfg.MAX_ROUTE_NODES

    eff_top_k = 0 if not params["route_choice"] else cfg.TOP_K_DIRECT
    eff_alpha_quality = 0.0 if not params["route_rules"] else cfg.ALPHA_QUALITY
    eff_f_max = cfg.F_MIN if not params["freq_opt"] else cfg.F_MAX

    t_start = time.time()

    result = colgen_loop_joint(
        all_data, G, nodes, prox, PERIOD_WEIGHTS,
        initial_count=MAX_PARAMS["initial_count"],
        pricing_count=MAX_PARAMS["pricing_count"],
        min_routes=MAX_PARAMS["min_routes"],
        max_routes=MAX_PARAMS["max_routes"],
        alpha_pass=params["alpha_pass"],
        alpha_adeq=params["alpha_adeq"],
        alpha_oper=params["alpha_oper"],
        top_k=eff_top_k,
        alpha_quality=eff_alpha_quality,
        f_max=eff_f_max,
        time_limit=MAX_PARAMS["time_limit"],
        mip_gap=cfg.GUROBI_MIP_GAP,
        max_iter=MAX_PARAMS["max_iter"],
        min_improvement=0.001,
        seed=42,
        verbose=True,
        route_gen_G=route_gen_G,
        min_route_len=min_rl,
        max_route_len=max_rl,
        points=sample.get("points"),
    )

    selected = result.get("selected_routes")
    freqs_pp = result.get("freqs_per_period")
    if selected is None:
        raise RuntimeError("Aucune solution réalisable trouvée.")

    # Local-search refinement
    if MAX_PARAMS["ls_rounds"] > 0:
        periods_ls = {
            p: {
                "node_pot": all_data[p]["node_pot"],
                "node_q_min": all_data[p]["node_q_min"],
                "node_q_max": all_data[p]["node_q_max"],
            } for p in ALL_BANDS
        }
        # Build node_coords now so the LS quality guard can fire
        ls_node_coords = build_node_coords(points) if points else None

        selected, freqs_pp = local_search_joint(
            selected, freqs_pp,
            G, nodes, periods_ls, prox, PERIOD_WEIGHTS,
            alpha_adeq=params["alpha_adeq"],
            alpha_oper=params["alpha_oper"],
            max_rounds=MAX_PARAMS["ls_rounds"],
            max_route_len=max_rl,
            node_coords=ls_node_coords,
            route_gen_G=route_gen_G,
            alpha_quality=eff_alpha_quality,
        )

    # Axis-sort for nicer rendering
    nc_sort = build_node_coords(points) if points else {}
    if nc_sort:
        selected = [_axis_sort_route(r, nc_sort) for r in selected]

    # Evaluation per period
    peak_band = ALL_BANDS[0]
    peak_data = all_data[peak_band]
    peak_freqs = freqs_pp[peak_band]
    node_coords = build_node_coords(peak_data.get("points")) if peak_data.get("points") else {}

    peak_metrics = evaluate_full(
        selected, peak_data["G"], peak_data["nodes"], peak_data["demand_triples"])
    peak_adeq = adequation_report(
        selected, peak_freqs, peak_data["nodes"], peak_data["prox"],
        peak_data["node_pot"], peak_data["node_q_min"], peak_data["node_q_max"])

    # Per-node adequation status (needed for the dashboard)
    quom = defaultdict(float)
    for route, fk in zip(selected, peak_freqs):
        if fk <= 0:
            continue
        for stop_node in set(route):
            for j, w in peak_data["prox"].get(stop_node, []):
                quom[j] += w * fk

    adeq_node_status = {}
    for j in peak_data["nodes"]:
        pot = peak_data["node_pot"].get(j, 0)
        if pot <= 0:
            adeq_node_status[j] = "none"
        elif quom.get(j, 0) < peak_data["node_q_min"].get(j, 0):
            adeq_node_status[j] = "under"
        elif quom.get(j, 0) > peak_data["node_q_max"].get(j, float("inf")):
            adeq_node_status[j] = "over"
        else:
            adeq_node_status[j] = "ok"

    stop_names = {}
    if peak_data.get("points"):
        for p in peak_data["points"]:
            nid = p.get("id", 0)
            stop_names[nid] = p.get("stop_name", f"Stop {nid}")

    elapsed = time.time() - t_start

    # Per-period results (lightweight)
    period_summary = {}
    for band in ALL_BANDS:
        d = all_data[band]
        f_b = freqs_pp[band]
        m_b = evaluate_full(selected, d["G"], d["nodes"], d["demand_triples"])
        a_b = adequation_report(
            selected, f_b, d["nodes"], d["prox"],
            d["node_pot"], d["node_q_min"], d["node_q_max"])
        period_summary[band] = {
            "d0": m_b.get("d0"), "d1": m_b.get("d1"),
            "d2": m_b.get("d2"), "d3+": m_b.get("d3+"),
            "dun": m_b.get("dun"), "ATT": m_b.get("ATT"),
            "adeq_pct_ok": a_b.get("pct_ok"),
            "avg_freq": (sum(f_b) / len(f_b)) if f_b else 0,
        }

    # ----- Outputs -----
    title = (
        f"GeoCapt Demo — {len(selected)} lignes — "
        f"alpha_adeq={params['alpha_adeq']} (pot_hp)"
    )
    html_str = generate_dashboard_html(
        selected, peak_freqs, node_coords, G,
        peak_metrics, peak_adeq, adeq_node_status, stop_names,
        title=title,
    )
    html_path = RESULTS_DIR / f"{run_id}.html"
    html_path.write_text(html_str, encoding="utf-8")

    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": round(elapsed, 1),
        "params": params,
        "fixed_params": MAX_PARAMS,
        "n_routes": len(selected),
        "peak_band": peak_band,
        "peak_metrics": {
            "d0": peak_metrics.get("d0"),
            "d1": peak_metrics.get("d1"),
            "d2": peak_metrics.get("d2"),
            "d3+": peak_metrics.get("d3+"),
            "dun": peak_metrics.get("dun"),
            "ATT": peak_metrics.get("ATT"),
        },
        "peak_adequation_pct_ok": peak_adeq.get("pct_ok"),
        "period_results": period_summary,
        "selected_routes": selected,
        "freqs_per_period": {b: list(freqs_pp[b]) for b in ALL_BANDS},
    }
    json_path = RESULTS_DIR / f"{run_id}.json"
    json_path.write_text(json.dumps(summary, indent=2, default=float))

    # Append to runs.csv
    csv_row = {
        "run_id": run_id,
        "timestamp": summary["timestamp"],
        "elapsed_s": summary["elapsed_s"],
        "n_routes": summary["n_routes"],
        "adequation_pct_ok": summary["peak_adequation_pct_ok"],
        "dun": summary["peak_metrics"]["dun"],
        "d0": summary["peak_metrics"]["d0"],
        "d1": summary["peak_metrics"]["d1"],
        "d2": summary["peak_metrics"]["d2"],
        "d3p": summary["peak_metrics"]["d3+"],
        "ATT": summary["peak_metrics"]["ATT"],
        **{f"p_{k}": v for k, v in params.items()},
        "html": html_path.name,
        "json": json_path.name,
    }
    write_header = not log_path.exists()
    with log_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(csv_row)

    return csv_row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=0,
                    help="Stop after N successful runs (0 = infinite, until Ctrl+C).")
    ap.add_argument("--grid", action="store_true",
                    help="Walk the full parameter grid (shuffled) instead of pure random sampling.")
    ap.add_argument("--seed", type=int, default=int(time.time()),
                    help="RNG seed for parameter sampling.")
    ap.add_argument("--gtfs", type=str, default=str(GTFS_DIR),
                    help="GTFS directory (default: bundled Poissy).")
    args = ap.parse_args()

    print(f"[automate] results dir: {RESULTS_DIR}")
    print(f"[automate] mode: {'grid' if args.grid else 'random'}, seed={args.seed}")
    print(f"[automate] fixed params: {MAX_PARAMS}")

    # Override GTFS path on the cfg module so the loader sees it
    cfg.GTFS_STOPS_PATH = str(Path(args.gtfs) / "stops.txt")

    # ----- Load data once (use_new_od is always ON, see FIXED_FLAGS) -----
    print("[automate] loading data for all periods…")
    all_data = {}
    G = None
    nodes = None
    prox = None
    for band in ALL_BANDS:
        all_data[band] = load_all(
            csv_path=str(POM_CSV), pot_col=band,
            use_new_od=FIXED_FLAGS["use_new_od"],
        )
        if G is None:
            G = all_data[band]["G"]
            nodes = all_data[band]["nodes"]
            prox = all_data[band]["prox"]
        print(f"  {band}: {all_data[band]['n']} nœuds, "
              f"{len(all_data[band]['demand_triples'])} paires OD")

    points = all_data[ALL_BANDS[0]].get("points")

    log_path = RESULTS_DIR / "runs.csv"
    print(f"[automate] log: {log_path}")
    print(f"[automate] starting loop — Ctrl+C to stop\n")

    n_done = 0
    n_failed = 0
    combos = combo_iter("grid" if args.grid else "random", args.seed)

    for params in combos:
        if _STOP:
            break

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{ts}_{combo_hash(params)}"
        print(f"\n{'=' * 70}")
        print(f"[automate] RUN #{n_done + 1}  id={run_id}")
        print(f"  params: {params}")
        print('=' * 70)

        try:
            row = run_one(params, all_data, G, nodes, prox, points,
                          run_id, log_path)
            n_done += 1
            print(f"[automate] ✔ done in {row['elapsed_s']}s — "
                  f"adéquation={row['adequation_pct_ok']:.1f}%, "
                  f"dun={row['dun']:.1f}%, "
                  f"d0={row['d0']:.1f}%, lignes={row['n_routes']}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            n_failed += 1
            print(f"[automate] ✘ run failed: {e}")
            traceback.print_exc()
            err_path = RESULTS_DIR / f"{run_id}.error.txt"
            err_path.write_text(
                f"params: {json.dumps(params, indent=2, default=str)}\n\n"
                f"{traceback.format_exc()}"
            )

        if args.runs and n_done >= args.runs:
            print(f"[automate] reached --runs={args.runs}, stopping.")
            break

    print(f"\n[automate] finished — {n_done} runs ok, {n_failed} failed.")
    print(f"[automate] see {log_path}")


if __name__ == "__main__":
    main()

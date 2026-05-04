"""
GeoCapt-UTRP Demo — Interactive Transit Network Optimisation
============================================================
Streamlit demo for the Capt-Temp pipeline.  Upload GTFS data,
configure priorities (adequation vs direct transfers), and run
the Gurobi joint multi-period optimiser.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Add Capt-Temp to the Python path so we can import the pipeline modules
# ---------------------------------------------------------------------------
CAPT_TEMP_DIR = str(Path(__file__).resolve().parent.parent / "Capt-Temp")
if CAPT_TEMP_DIR not in sys.path:
    sys.path.insert(0, CAPT_TEMP_DIR)

DEMO_DIR = Path(__file__).resolve().parent
DATA_DIR = DEMO_DIR / "data"
GTFS_DIR = DATA_DIR / "GTFS"
POM_CSV = DATA_DIR / "pom_poissy.csv"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GeoCapt-UTRP Demo",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: #1a1a2e;
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        text-align: center;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: bold;
        color: #4fc3f7;
    }
    .metric-card .label {
        font-size: 12px;
        color: #aaa;
        text-transform: uppercase;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    div[data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    .route-badge {
        display: inline-block;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        color: white;
        font-size: 12px;
        font-weight: bold;
        text-align: center;
        line-height: 24px;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Helper: save uploaded GTFS files to disk
# ============================================================================

GTFS_REQUIRED = ["stops.txt", "routes.txt", "trips.txt", "stop_times.txt",
                  "shapes.txt", "calendar.txt", "agency.txt"]


def save_uploaded_gtfs(uploaded_files: dict[str, object], dest_dir: Path) -> list[str]:
    """Write uploaded files to dest_dir.  Returns list of saved filenames."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, fobj in uploaded_files.items():
        (dest_dir / name).write_bytes(fobj.getvalue())
        saved.append(name)
    return saved


# ============================================================================
# Helper: generate the interactive HTML dashboard (like plot_routes_map)
# ============================================================================

_ROUTE_COLOURS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#000075", "#a9a9a9", "#e6beff", "#ffe119", "#ffd8b1",
]


# ============================================================================
# Marketing translation layer
# ============================================================================
# Maps the optimizer's OR-jargon metrics to customer-and-brand French labels
# used in the "Vue Marketing" tab and the side-by-side scenario ledger.

MKT_LABELS = {
    "d0":       "Voyages directs (sans correspondance)",
    "d1":       "Voyages avec 1 correspondance",
    "d2plus":   "Voyages avec 2 correspondances ou plus",
    "dun":      "Demande non desservie",
    "ATT":      "Durée moyenne d'un trajet (min)",
    "pct_ok":   "Arrêts à bonne offre",
    "n_over":   "Capacité gaspillée (bus à moitié vides)",
    "n_under":  "Arrêts saturés (file d'attente)",
    "n_routes": "Lignes au plan",
}

# +1 higher-is-better, -1 lower-is-better, 0 neutral (no arrow colour)
MKT_DIRECTION = {
    "d0": +1, "d1": 0, "d2plus": -1, "dun": -1, "ATT": -1,
    "pct_ok": +1, "n_over": -1, "n_under": -1, "n_routes": 0,
}

# Pretty value formatter per metric — used in headline cards and the ledger.
def _mkt_format(key: str, value: float | int) -> str:
    if value is None:
        return "—"
    if key in ("d0", "d1", "d2plus", "dun", "pct_ok"):
        return f"{value:.1f} %"
    if key == "ATT":
        return f"{value:.2f} min"
    if key in ("n_over", "n_under", "n_routes"):
        return f"{int(value)}"
    return f"{value:.2f}"


def _normalize_mix(mix: dict) -> dict:
    """Normalize the Mix temporel sliders so the 4 weights sum to 1.0."""
    s = sum(mix.values())
    if s <= 0:
        return {"pot_hp": 0.40, "pot_hc": 0.30, "pot_soir": 0.20, "pot_nuit": 0.10}
    return {k: v / s for k, v in mix.items()}


def _marketing_to_params(confort: int, budget: int, mix: dict) -> dict:
    """Map marketing-friendly knobs to the technical params dict.

    confort 0..100: 0=coverage-priority, 50=balanced, 100=direct-priority.
                    Anchors mirror the three Step 3 PRESETS.
    budget  0..100: 0=lean fleet (3 lignes, alpha_oper=0.05),
                    100=generous fleet (20 lignes, alpha_oper=0.005).
    mix:            raw weights for the 4 time bands; normalized internally.
    """
    if confort <= 50:
        t = confort / 50.0
        alpha_pass = 0.5 + t * (1.0 - 0.5)
        alpha_adeq = 0.5 + t * (0.1 - 0.5)
    else:
        t = (confort - 50) / 50.0
        alpha_pass = 1.0
        alpha_adeq = 0.1 + t * (0.01 - 0.1)

    t_b = budget / 100.0
    max_routes = int(round(3 + t_b * (20 - 3)))
    alpha_oper = 0.05 + t_b * (0.005 - 0.05)

    return {
        "alpha_pass": alpha_pass,
        "alpha_adeq": alpha_adeq,
        "alpha_oper": alpha_oper,
        "use_new_od": True,
        "route_choice": True,
        "freq_opt": True,
        "route_rules": True,
        "min_routes": 3,
        "max_routes": max_routes,
        "time_limit": 450,
        "max_iter": 3,
        "initial_count": 5000,
        "pricing_count": 500,
        "ls_rounds": 7,
        "_period_weights": _normalize_mix(mix),
    }


def _extract_mkt_values(results: dict) -> dict:
    """Pull the headline marketing values from a `results` dict (peak band)."""
    if not results:
        return {}
    m = results.get("peak_metrics", {}) or {}
    a = results.get("peak_adeq", {}) or {}
    return {
        "d0":       m.get("d0", 0),
        "d1":       m.get("d1", 0),
        "d2plus":   m.get("d2", 0) + m.get("d3+", 0),
        "dun":      m.get("dun", 0),
        "ATT":      m.get("ATT", 0),
        "pct_ok":   a.get("pct_ok", 0),
        "n_over":   a.get("n_over", 0),
        "n_under":  a.get("n_under", 0),
        "n_routes": results.get("n_routes", 0),
    }


def _js_esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")


def _bearing(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Initial compass bearing (0-360°) from point a to point b.
    Both inputs are (lat, lon)."""
    import math

    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = (math.cos(lat1) * math.sin(lat2)
         - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def _fetch_segment_geometry(
    a: tuple[float, float],
    b: tuple[float, float],
    access_token: str,
    profile: str = "driving",
    entry_bearing: float | None = None,
) -> list[tuple[float, float]]:
    """Fetch the driving path between two waypoints.

    If entry_bearing is provided, constrains the direction of travel at 'a'
    so the route leaves 'a' heading in approximately that direction (±45°).
    This is how segment-by-segment fetching preserves continuity at
    junctions and avoids U-turns.
    Returns list of (lat, lon); falls back to straight line on failure.
    """
    import urllib.request

    coords_str = f"{a[1]},{a[0]};{b[1]},{b[0]}"  # lon,lat;lon,lat
    params = ["geometries=geojson", "overview=full"]
    if entry_bearing is not None:
        # Bearings format: "brg,range;brg,range" — one entry per waypoint.
        # Constrain entry at A, leave B unconstrained (trailing ';').
        params.append(f"bearings={int(round(entry_bearing))},45;")
    url = (
        f"https://api.mapbox.com/directions/v5/mapbox/{profile}/{coords_str}"
        f"?{'&'.join(params)}&access_token={access_token}"
    )
    try:
        resp = urllib.request.urlopen(urllib.request.Request(url))
        data = json.loads(resp.read())
        if data.get("code") == "Ok" and data.get("routes"):
            geojson_coords = data["routes"][0]["geometry"]["coordinates"]
            return [(lat, lon) for lon, lat in geojson_coords]
    except Exception:
        pass
    # If the constrained call fails, retry once without the bearing so a
    # geometrically impossible constraint doesn't drop us to a straight line.
    if entry_bearing is not None:
        return _fetch_segment_geometry(a, b, access_token, profile, None)
    return [a, b]


def _fetch_road_geometry(
    waypoints: list[tuple[float, float]],
    access_token: str,
    profile: str = "driving",
) -> list[tuple[float, float]]:
    """Segment-by-segment road geometry with bearing continuity.

    Each consecutive pair (A->B, B->C, ...) is fetched independently, but
    every call after the first receives an entry bearing derived from the
    tail of the previous segment.  This prevents the shortest-path solver
    from U-turning at intermediate junctions while still letting each
    segment be solved in isolation.
    """
    if len(waypoints) < 2:
        return list(waypoints)

    full: list[tuple[float, float]] = []
    entry_bearing: float | None = None
    for i in range(len(waypoints) - 1):
        seg = _fetch_segment_geometry(
            waypoints[i], waypoints[i + 1], access_token, profile,
            entry_bearing=entry_bearing,
        )
        if i == 0:
            full.extend(seg)
        else:
            # Avoid duplicating the junction point
            full.extend(seg[1:] if seg else [])
        # Derive the next segment's entry bearing from this segment's tail
        if len(seg) >= 2:
            entry_bearing = _bearing(seg[-2], seg[-1])
        else:
            entry_bearing = None
        time.sleep(0.05)
    return full


def generate_dashboard_html(
    routes, freqs, node_coords, G, metrics, adeq,
    adeq_node_status, stop_names, title="Optimised Routes",
) -> str:
    """Build the self-contained HTML dashboard string."""

    # Mapbox road-snapped geometry cache
    cache_dir = DEMO_DIR / ".mapbox_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "route_geometries.json"
    geom_cache = {}
    if cache_file.exists():
        try:
            geom_cache = json.loads(cache_file.read_text())
        except Exception:
            pass

    # Mapbox token from .env or env var
    access_token = os.environ.get("MAPBOX_ACCESS_TOKEN", "")
    if not access_token:
        env_path = DEMO_DIR / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("MAPBOX_ACCESS_TOKEN="):
                    access_token = line.split("=", 1)[1].strip()
    use_mapbox = bool(access_token)
    # use_mapbox = False
    cache_dirty = False

    # Build route infos
    route_infos = []
    total_veh_min = 0.0
    for i, (route, freq) in enumerate(zip(routes, freqs)):
        color = _ROUTE_COLOURS[i % len(_ROUTE_COLOURS)]
        waypoints = []
        for nid in route:
            c = node_coords.get(nid)
            if c:
                waypoints.append((c[1], c[0]))  # (lat, lon)

        # Try cache first, then Mapbox API, then straight lines
        # v4 suffix invalidates v3 (single multi-waypoint) geometries; the
        # current fetcher is segment-by-segment with bearing continuity at
        # each junction to prevent U-turns.
        cache_key = "v4|" + ";".join(f"{lat:.6f},{lon:.6f}" for lat, lon in waypoints)
        if cache_key in geom_cache:
            coords = [f"[{c[0]}, {c[1]}]" for c in geom_cache[cache_key]]
        elif use_mapbox and waypoints:
            road_coords = _fetch_road_geometry(
                waypoints, access_token, "driving")
            geom_cache[cache_key] = road_coords
            cache_dirty = True
            coords = [f"[{lat}, {lon}]" for lat, lon in road_coords]
        else:
            coords = [f"[{lat}, {lon}]" for lat, lon in waypoints]

        # Route travel time
        rt = 0.0
        for k in range(len(route) - 1):
            u, v = route[k], route[k + 1]
            for nb, tt in G.get(u, []):
                if nb == v:
                    rt += tt
                    break
        total_veh_min += freq * rt
        snames = [stop_names.get(n, f"Node {n}") for n in route]

        route_infos.append({
            "idx": i + 1, "color": color, "coords": coords,
            "freq": freq, "n_stops": len(route),
            "stop_names": snames, "time": rt,
        })

    # Save geometry cache if we fetched new geometries
    if cache_dirty:
        try:
            cache_file.write_text(json.dumps(geom_cache))
        except Exception:
            pass

    # Metrics
    m_d0 = metrics.get("d0", 0) if metrics else 0
    m_d1 = metrics.get("d1", 0) if metrics else 0
    m_d2p = (metrics.get("d2", 0) + metrics.get("d3+", 0)) if metrics else 0
    m_dun = metrics.get("dun", 0) if metrics else 0
    m_att = metrics.get("ATT", 0) if metrics else 0
    m_pct_ok = adeq.get("pct_ok", 0) if adeq else 0

    # Adequation node colours
    adeq_colors = {}
    if adeq_node_status:
        for nid, status in adeq_node_status.items():
            if status == "ok":
                adeq_colors[nid] = "#DAA520"
            elif status == "under":
                adeq_colors[nid] = "blue"
            elif status == "over":
                adeq_colors[nid] = "red"
            else:
                adeq_colors[nid] = "gray"

    # Map centre
    all_lats, all_lons = [], []
    for route in routes:
        for nid in route:
            c = node_coords.get(nid)
            if c:
                all_lons.append(c[0])
                all_lats.append(c[1])
    avg_lat = sum(all_lats) / len(all_lats) if all_lats else 48.93
    avg_lon = sum(all_lons) / len(all_lons) if all_lons else 2.04

    # Build routes table HTML
    routes_table = ""
    for ri in route_infos:
        itin = " &rarr; ".join(f"<span>{_js_esc(n)}</span>" for n in ri["stop_names"])
        routes_table += f"""
        <tr class="route-row" data-route="{ri['idx']}" onclick="selectRoute({ri['idx']})">
          <td><span class="route-badge" style="background:{ri['color']}">{ri['idx']}</span></td>
          <td>{ri['freq']:.1f}</td>
          <td>{ri['n_stops']}</td>
          <td>{ri['time']:.1f}</td>
          <td class="itinerary">{itin}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>{title}</title>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: 'Segoe UI', Arial, sans-serif; display: flex; flex-direction: column; height: 100vh; }}
  #top-bar {{ background: #1a1a2e; color: white; padding: 10px 20px; display: flex; align-items: center; gap: 30px; flex-shrink: 0; }}
  #top-bar h2 {{ margin: 0; font-size: 16px; }}
  .metric {{ text-align: center; }}
  .metric-val {{ font-size: 18px; font-weight: bold; color: #4fc3f7; }}
  .metric-lbl {{ font-size: 10px; color: #aaa; text-transform: uppercase; }}
  #main {{ display: flex; flex: 1; overflow: hidden; }}
  #map {{ flex: 1; min-width: 0; }}
  #sidebar {{ width: 420px; background: #f8f9fa; border-left: 1px solid #ddd; overflow-y: auto; flex-shrink: 0; }}
  #sidebar h3 {{ margin: 12px 15px 8px; font-size: 14px; color: #333; }}
  .legend-bar {{ display: flex; gap: 15px; padding: 0 15px 10px; font-size: 12px; color: #555; }}
  .legend-bar span {{ display: flex; align-items: center; gap: 4px; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  table.routes {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  table.routes th {{ background: #e9ecef; padding: 6px 8px; text-align: left; position: sticky; top: 0; }}
  table.routes td {{ padding: 5px 8px; border-bottom: 1px solid #eee; vertical-align: top; }}
  table.routes tr {{ cursor: pointer; }}
  table.routes tr:hover {{ background: #e3f2fd; }}
  table.routes tr.selected {{ background: #bbdefb; }}
  .route-badge {{ display: inline-block; width: 22px; height: 22px; border-radius: 50%; color: white;
    font-size: 11px; font-weight: bold; text-align: center; line-height: 22px; }}
  .itinerary {{ font-size: 11px; color: #555; max-width: 220px; line-height: 1.4; }}
  .itinerary span {{ white-space: nowrap; }}
</style>
</head>
<body>
<div id="top-bar">
  <h2>{title}</h2>
  <div class="metric"><div class="metric-val">{m_d0:.1f}%</div><div class="metric-lbl">d0 direct</div></div>
  <div class="metric"><div class="metric-val">{m_d1:.1f}%</div><div class="metric-lbl">d1 correspondance</div></div>
  <div class="metric"><div class="metric-val">{m_d2p:.1f}%</div><div class="metric-lbl">d2+</div></div>
  <div class="metric"><div class="metric-val">{m_dun:.1f}%</div><div class="metric-lbl">dun non desservi</div></div>
  <div class="metric"><div class="metric-val">{m_att:.2f}</div><div class="metric-lbl">ATT (min)</div></div>
  <div class="metric"><div class="metric-val">{m_pct_ok:.1f}%</div><div class="metric-lbl">Adéquation</div></div>
</div>
<div id="main">
<div id="map"></div>
<div id="sidebar">
  <h3>Légende adéquation</h3>
  <div class="legend-bar">
    <span><div class="legend-dot" style="background:#DAA520;"></div> OK</span>
    <span><div class="legend-dot" style="background:blue;"></div> Sous-offre</span>
    <span><div class="legend-dot" style="background:red;"></div> Sur-offre</span>
    <span><div class="legend-dot" style="background:gray;"></div> Aucune donnée</span>
  </div>
  <h3>Lignes ({len(routes)})</h3>
  <table class="routes">
    <thead><tr><th>#</th><th>f/h</th><th>Arrêts</th><th>Min</th><th>Itinéraire</th></tr></thead>
    <tbody>{routes_table}</tbody>
  </table>
</div>
</div>
<script>
var map = L.map('map').setView([{avg_lat}, {avg_lon}], 14);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '&copy; OpenStreetMap &copy; CARTO',
  subdomains: 'abcd',
  maxZoom: 20
}}).addTo(map);
var routeLayers = {{}};
"""
    # Stop markers
    served_nodes = set()
    for route in routes:
        served_nodes.update(route)

    for nid in node_coords:
        c = node_coords[nid]
        lat, lon = c[1], c[0]
        is_served = nid in served_nodes
        color = adeq_colors.get(nid, "gray" if not is_served else "#DAA520")
        name = _js_esc(stop_names.get(nid, f"Node {nid}"))
        radius = 6 if is_served else 3
        opacity = 0.8 if is_served else 0.4
        html += f"""
L.circleMarker([{lat}, {lon}], {{
  radius: {radius}, fillColor: '{color}', color: '#333', weight: 1, fillOpacity: {opacity}
}}).addTo(map).bindPopup('{name}');
"""

    # Route polylines
    for ri in route_infos:
        coords_str = ",".join(ri["coords"])
        weight = max(4, min(8, ri["freq"] / 2))
        popup_stops = " &rarr; ".join(_js_esc(n) for n in ri["stop_names"])
        popup_text = _js_esc(
            f"<b>Ligne {ri['idx']}</b><br>"
            f"f={ri['freq']:.1f} véh/h, {ri['n_stops']} arrêts, {ri['time']:.1f} min<br>"
            f"<small>{popup_stops}</small>"
        )
        html += f"""
routeLayers[{ri['idx']}] = L.polyline([{coords_str}], {{
  color: '{ri['color']}', weight: {weight:.0f}, opacity: 1.0
}}).addTo(map).bindPopup('{popup_text}');
"""

    html += """
var selectedRoute = null;
var defaultWeights = {};
for (var id in routeLayers) { defaultWeights[id] = routeLayers[id].options.weight; }
function clearSelection() {
  for (var id in routeLayers) { routeLayers[id].setStyle({weight: defaultWeights[id], opacity: 1.0}); }
  selectedRoute = null;
  document.querySelectorAll('.route-row').forEach(function(r) { r.classList.remove('selected'); });
}
function selectRoute(id) {
  if (selectedRoute === id) { clearSelection(); return; }
  clearSelection(); selectedRoute = id;
  for (var rid in routeLayers) { routeLayers[rid].setStyle({weight: defaultWeights[rid], opacity: 0.2}); }
  routeLayers[id].setStyle({weight: 9, opacity: 1.0}); routeLayers[id].bringToFront();
  var row = document.querySelector('.route-row[data-route="' + id + '"]');
  if (row) row.classList.add('selected');
}
document.addEventListener('keydown', function(e) { if (e.key === 'Escape') clearSelection(); });
</script>
</body>
</html>"""
    return html


# ============================================================================
# Pipeline runner — shared by the engineering view (Step 4) and the
# marketing view (Step 5).  Renders progress widgets at the call site.
# ============================================================================

def _apply_zone_multipliers(data: dict, factors: dict) -> None:
    """In-place: scale node_pot/q_min/q_max and demand_triples by per-node factor."""
    if not factors:
        return
    for node, f in factors.items():
        if node in data.get("node_pot", {}):
            data["node_pot"][node] *= f
        if node in data.get("node_q_min", {}):
            data["node_q_min"][node] *= f
        if node in data.get("node_q_max", {}):
            data["node_q_max"][node] *= f
    if "demand_triples" in data:
        data["demand_triples"] = [
            (i, j, d * (factors.get(i, 1.0) + factors.get(j, 1.0)) / 2.0)
            for (i, j, d) in data["demand_triples"]
        ]


def run_optimization(params: dict, gtfs_path: str) -> dict | None:
    """Run the joint multi-period pipeline and return a results dict.

    `params` accepts the same shape produced by Step 3 / Step 5.  Optional
    keys (used by the marketing vues):
      * `_period_weights` — override for the time-band mix.
      * `_f_min` / `_f_max` (veh/h) — override frequency floor / ceiling.
      * `_transfer_penalty` (min) — override `cfg.TRANSFER_PENALTY_MIN`.
      * `_zone_factors: dict[node_id, float]` — geographic demand boost.

    Returns None on failure (after surfacing st.error).
    Renders st.progress + st.status widgets at the call site.
    """
    progress = st.progress(0, text="Importation des modules du pipeline...")

    try:
        import config as cfg
        cfg.GTFS_STOPS_PATH = str(Path(gtfs_path) / "stops.txt")

        from geocapt_loader import load_all
        from geocapt_colgen import colgen_loop_joint
        from geocapt_gurobi import evaluate_full, adequation_report
        from geocapt_localsearch import local_search_joint
        from geocapt_route_quality import build_node_coords
    except ImportError as e:
        st.error(f"Échec de l'importation des modules du pipeline : {e}")
        st.info(f"Assurez-vous que Capt-Temp se trouve à : `{CAPT_TEMP_DIR}`")
        return None

    # Snapshot cfg attrs we may temporarily mutate so each run is independent.
    saved_f_min = cfg.F_MIN
    saved_transfer = cfg.TRANSFER_PENALTY_MIN
    if "_f_min" in params:
        cfg.F_MIN = float(params["_f_min"])
    if "_transfer_penalty" in params:
        cfg.TRANSFER_PENALTY_MIN = float(params["_transfer_penalty"])

    try:
        return _run_optimization_inner(
            params, cfg, progress,
            load_all, colgen_loop_joint, evaluate_full, adequation_report,
            local_search_joint, build_node_coords,
        )
    finally:
        cfg.F_MIN = saved_f_min
        cfg.TRANSFER_PENALTY_MIN = saved_transfer


def _run_optimization_inner(
    params, cfg, progress,
    load_all, colgen_loop_joint, evaluate_full, adequation_report,
    local_search_joint, build_node_coords,
) -> dict | None:
    status = st.status("Exécution de l'optimisation conjointe multi-période...", expanded=True)
    progress.progress(5, text="Phase 1 : Chargement des données pour toutes les périodes...")

    ALL_BANDS = ["pot_hp", "pot_hc", "pot_soir", "pot_nuit"]
    period_weights = params.get("_period_weights") or {
        "pot_hp": 0.40, "pot_hc": 0.30, "pot_soir": 0.20, "pot_nuit": 0.10,
    }
    zone_factors = params.get("_zone_factors") or {}

    all_data = {}
    G = None
    nodes = None
    prox = None
    t_start = time.time()

    for band in ALL_BANDS:
        status.write(f"Chargement de {band}...")
        data = load_all(
            csv_path=str(POM_CSV),
            pot_col=band,
            use_new_od=params["use_new_od"],
        )
        if zone_factors:
            _apply_zone_multipliers(data, zone_factors)
        all_data[band] = data
        if G is None:
            G = data["G"]
            nodes = data["nodes"]
            prox = data["prox"]
        status.write(f"  {data['n']} nœuds, {len(data['demand_triples'])} paires OD")

    progress.progress(20, text="Phase 1 terminée. Démarrage de l'optimisation...")

    sample_data = all_data[ALL_BANDS[0]]
    is_gtfs = sample_data.get("mode") == "gtfs_osm"
    route_gen_G = sample_data.get("route_gen_G")
    min_rl = cfg.GTFS_MIN_ROUTE_NODES if is_gtfs else cfg.MIN_ROUTE_NODES
    max_rl = cfg.GTFS_MAX_ROUTE_NODES if is_gtfs else cfg.MAX_ROUTE_NODES

    progress.progress(25, text="Phase 2-3 : Génération conjointe de colonnes...")
    status.write("Exécution de la génération de colonnes (peut prendre plusieurs minutes)...")

    eff_top_k = 0 if not params["route_choice"] else cfg.TOP_K_DIRECT
    eff_alpha_quality = 0.0 if not params["route_rules"] else cfg.ALPHA_QUALITY
    f_max_override = float(params["_f_max"]) if "_f_max" in params else cfg.F_MAX
    eff_f_max = cfg.F_MIN if not params["freq_opt"] else f_max_override

    result = colgen_loop_joint(
        all_data, G, nodes, prox, period_weights,
        initial_count=params.get("initial_count", 1000),
        pricing_count=params.get("pricing_count", 500),
        min_routes=params["min_routes"],
        max_routes=params["max_routes"],
        alpha_pass=params["alpha_pass"],
        alpha_adeq=params["alpha_adeq"],
        alpha_oper=params["alpha_oper"],
        top_k=eff_top_k,
        alpha_quality=eff_alpha_quality,
        f_max=eff_f_max,
        time_limit=params["time_limit"],
        mip_gap=cfg.GUROBI_MIP_GAP,
        max_iter=params["max_iter"],
        min_improvement=0.001,
        seed=42,
        verbose=True,
        route_gen_G=route_gen_G,
        min_route_len=min_rl,
        max_route_len=max_rl,
        points=sample_data.get("points"),
    )

    selected = result.get("selected_routes")
    freqs_pp = result.get("freqs_per_period")

    if selected is None:
        st.error("Aucune solution réalisable trouvée. Essayez d'ajuster les paramètres.")
        return None

    progress.progress(70, text="Génération de colonnes terminée. Exécution de la recherche locale...")

    ls_rounds = params.get("ls_rounds", 3)
    if ls_rounds > 0:
        status.write(f"Exécution du raffinement par recherche locale ({ls_rounds} tours)...")
        periods_ls = {
            p: {"node_pot": all_data[p]["node_pot"],
                "node_q_min": all_data[p]["node_q_min"],
                "node_q_max": all_data[p]["node_q_max"]}
            for p in ALL_BANDS
        }
        ls_points = sample_data.get("points")
        ls_node_coords = build_node_coords(ls_points) if ls_points else None
        selected, freqs_pp = local_search_joint(
            selected, freqs_pp,
            G, nodes, periods_ls, prox, period_weights,
            alpha_adeq=params["alpha_adeq"],
            alpha_oper=params["alpha_oper"],
            max_rounds=ls_rounds,
            max_route_len=max_rl,
            node_coords=ls_node_coords,
            route_gen_G=route_gen_G,
            alpha_quality=eff_alpha_quality,
        )
    else:
        status.write("Recherche locale ignorée.")

    status.write("Tri par axe des lignes finales pour une meilleure lisibilité...")
    from geocapt_routes import _axis_sort_route
    points_for_sort = all_data[ALL_BANDS[0]].get("points")
    nc_sort = build_node_coords(points_for_sort) if points_for_sort else {}
    if nc_sort:
        selected = [_axis_sort_route(r, nc_sort) for r in selected]

    progress.progress(85, text="Évaluation des résultats...")

    peak_band = ALL_BANDS[0]
    peak_data = all_data[peak_band]
    peak_freqs = freqs_pp[peak_band]
    points = peak_data.get("points")
    node_coords = build_node_coords(points) if points else {}

    peak_metrics = evaluate_full(
        selected, peak_data["G"], peak_data["nodes"],
        peak_data["demand_triples"])
    peak_adeq = adequation_report(
        selected, peak_freqs, peak_data["nodes"], peak_data["prox"],
        peak_data["node_pot"], peak_data["node_q_min"],
        peak_data["node_q_max"])

    quom = defaultdict(float)
    for k, (route, fk) in enumerate(zip(selected, peak_freqs)):
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
    if points:
        for p in points:
            nid = p.get("id", 0)
            stop_names[nid] = p.get("stop_name", f"Stop {nid}")

    elapsed = time.time() - t_start
    progress.progress(95, text="Génération de la carte...")

    period_results = {}
    for band in ALL_BANDS:
        d = all_data[band]
        f_b = freqs_pp[band]
        m_b = evaluate_full(selected, d["G"], d["nodes"], d["demand_triples"])
        a_b = adequation_report(
            selected, f_b, d["nodes"], d["prox"],
            d["node_pot"], d["node_q_min"], d["node_q_max"])
        period_results[band] = {"metrics": m_b, "adeq": a_b, "freqs": f_b}

    progress.progress(100, text="Terminé !")
    status.update(label=f"Optimisation terminée en {elapsed:.0f} s", state="complete")

    return {
        "selected": selected,
        "freqs_pp": freqs_pp,
        "peak_metrics": peak_metrics,
        "peak_adeq": peak_adeq,
        "adeq_node_status": adeq_node_status,
        "node_coords": node_coords,
        "stop_names": stop_names,
        "G": G,
        "elapsed": elapsed,
        "n_routes": len(selected),
        "period_results": period_results,
        "history": result.get("history", []),
    }


# ============================================================================
# Ledger renderer — side-by-side scenario comparison for the marketing view.
# ============================================================================

def render_ledger(scenario_A: dict | None, scenario_B: dict | None) -> pd.DataFrame:
    """Build a comparison DataFrame from two locked marketing scenarios.

    Each `scenario_*` is `{"results": <results dict>, "knobs": <dict>}`.
    The DataFrame is rendered via `st.dataframe` with delta arrows.
    """
    a_vals = _extract_mkt_values(scenario_A.get("results") if scenario_A else None)
    b_vals = _extract_mkt_values(scenario_B.get("results") if scenario_B else None)

    rows = []
    for key, label in MKT_LABELS.items():
        av = a_vals.get(key)
        bv = b_vals.get(key)
        if av is None or bv is None:
            continue
        delta = bv - av
        direction = MKT_DIRECTION.get(key, 0)
        if abs(delta) < 1e-3 or direction == 0:
            arrow = "—"
        elif (delta > 0 and direction > 0) or (delta < 0 and direction < 0):
            arrow = f"↑ {delta:+.2f}"  # better
        else:
            arrow = f"↓ {delta:+.2f}"  # worse
        rows.append({
            "Indicateur": label,
            "Scénario A": _mkt_format(key, av),
            "Scénario B": _mkt_format(key, bv),
            "Δ (B vs A)": arrow,
        })
    return pd.DataFrame(rows)


def _style_ledger_delta(val: str) -> str:
    if isinstance(val, str) and val.startswith("↑"):
        return "color: #1b8e3a; font-weight: 700;"
    if isinstance(val, str) and val.startswith("↓"):
        return "color: #c0392b; font-weight: 700;"
    return ""


def _styled_dataframe(df, delta_col: str = "Δ (B vs A)") -> "pd.io.formats.style.Styler":
    """Apply the green/red delta colouring; cope with pandas 2.0 vs 2.1+ Styler API."""
    styler = df.style
    apply_fn = getattr(styler, "map", None) or styler.applymap
    return apply_fn(_style_ledger_delta, subset=[delta_col])


def _compute_veh_hours(routes: list, freqs: list, G: dict) -> float:
    """Total vehicle-hours per service hour for the given routes & frequencies."""
    total_min = 0.0
    for r, f in zip(routes, freqs):
        if not r or f <= 0:
            continue
        rt = 0.0
        for k in range(len(r) - 1):
            u, v = r[k], r[k + 1]
            for nb, tt in G.get(u, []):
                if nb == v:
                    rt += tt
                    break
        total_min += f * rt
    return total_min / 60.0


def _render_headline_panel(vals: dict) -> None:
    """Render the 9-card customer-experience headline panel used by V1/V2/V4."""
    row1 = st.columns(3)
    row1[0].metric(MKT_LABELS["d0"], _mkt_format("d0", vals["d0"]),
                   help="Part des voyageurs qui restent dans le même bus du début à la fin.")
    row1[1].metric(MKT_LABELS["d1"], _mkt_format("d1", vals["d1"]),
                   help="Part des voyageurs qui doivent changer une fois de bus.")
    d2_val = vals["d2plus"]
    row1[2].metric(
        MKT_LABELS["d2plus"], _mkt_format("d2plus", d2_val),
        delta=("⚠ trop élevé" if d2_val > 5 else "OK"),
        delta_color="inverse",
        help="Voyageurs qui doivent changer 2 fois ou plus — signal d'alerte.",
    )
    row2 = st.columns(3)
    dun_val = vals["dun"]
    row2[0].metric(
        MKT_LABELS["dun"], _mkt_format("dun", dun_val),
        delta=("⚠ promesse non tenue" if dun_val > 1 else "OK"),
        delta_color="inverse",
        help="Demande laissée sans solution — voyageurs au bord du trottoir.",
    )
    row2[1].metric(MKT_LABELS["ATT"], _mkt_format("ATT", vals["ATT"]),
                   help="Temps moyen passé à bord pour un trajet.")
    row2[2].metric(MKT_LABELS["pct_ok"], _mkt_format("pct_ok", vals["pct_ok"]),
                   help="Part des arrêts où l'offre de bus correspond à la demande.")
    row3 = st.columns(3)
    row3[0].metric(MKT_LABELS["n_over"], _mkt_format("n_over", vals["n_over"]),
                   help="Arrêts en sur-offre — bus à moitié vides.")
    row3[1].metric(MKT_LABELS["n_under"], _mkt_format("n_under", vals["n_under"]),
                   help="Arrêts en sous-offre — files d'attente.")
    row3[2].metric(MKT_LABELS["n_routes"], _mkt_format("n_routes", vals["n_routes"]),
                   help="Nombre de lignes au plan du réseau optimisé.")


def _render_map_for_results(res: dict, title: str, height: int = 620) -> None:
    """Render the interactive Leaflet map for a `results` dict in any vue."""
    html_str = generate_dashboard_html(
        res["selected"], res["freqs_pp"]["pot_hp"],
        res["node_coords"], res["G"],
        res["peak_metrics"], res["peak_adeq"],
        res["adeq_node_status"], res["stop_names"],
        title=title,
    )
    st.components.v1.html(html_str, height=height, scrolling=True)


def _render_lock_swap_reset(vue_id: str, current: dict | None) -> tuple:
    """Render the four A/B controls; return (scenario_A, scenario_B) from session state."""
    key_A = f"mkt_v{vue_id}_scenario_A"
    key_B = f"mkt_v{vue_id}_scenario_B"
    cols = st.columns(4)
    if cols[0].button("Verrouiller comme scénario A", use_container_width=True,
                      key=f"lock_a_v{vue_id}", disabled=current is None):
        st.session_state[key_A] = current
        st.success("Scénario A verrouillé.")
    if cols[1].button("Verrouiller comme scénario B", use_container_width=True,
                      key=f"lock_b_v{vue_id}", disabled=current is None):
        st.session_state[key_B] = current
        st.success("Scénario B verrouillé.")
    if cols[2].button("Échanger A ↔ B", use_container_width=True, key=f"swap_v{vue_id}"):
        a = st.session_state.get(key_A)
        b = st.session_state.get(key_B)
        st.session_state[key_A] = b
        st.session_state[key_B] = a
    if cols[3].button("Réinitialiser", use_container_width=True, key=f"reset_v{vue_id}"):
        st.session_state.pop(key_A, None)
        st.session_state.pop(key_B, None)
    return st.session_state.get(key_A), st.session_state.get(key_B)


# ============================================================================
# Personas (V4 — Public cible)
# ============================================================================

PERSONAS = {
    "Pendulaires": {
        "label": "Pendulaires (heure de pointe, trajets directs)",
        "confort": 80, "budget": 70,
        "mix": {"pot_hp": 70, "pot_hc": 20, "pot_soir": 5, "pot_nuit": 5},
        "summary": ("Les pendulaires recherchent des trajets directs et fréquents "
                    "pendant les heures de pointe matin/soir."),
    },
    "Vie nocturne": {
        "label": "Vie nocturne (sorties soir et nuit)",
        "confort": 40, "budget": 50,
        "mix": {"pot_hp": 5, "pot_hc": 15, "pot_soir": 50, "pot_nuit": 30},
        "summary": ("Public sortant : la promesse est la couverture en soirée et "
                    "la nuit, pas le confort ultra-rapide."),
    },
    "Familles & loisirs": {
        "label": "Familles & loisirs (heure creuse, week-ends)",
        "confort": 30, "budget": 60,
        "mix": {"pot_hp": 15, "pot_hc": 55, "pot_soir": 25, "pot_nuit": 5},
        "summary": ("Déplacements de loisirs en journée — la fluidité hors pointe "
                    "et la couverture des pôles familiaux comptent plus que la vitesse."),
    },
    "Inclusion territoriale": {
        "label": "Inclusion territoriale (couverture équitable)",
        "confort": 10, "budget": 50,
        "mix": {"pot_hp": 25, "pot_hc": 25, "pot_soir": 25, "pot_nuit": 25},
        "summary": ("Promesse de service public : aucun arrêt oublié, équité entre "
                    "les périodes, même au prix de correspondances."),
    },
    "Étudiants": {
        "label": "Étudiants (campus + soirée)",
        "confort": 50, "budget": 55,
        "mix": {"pot_hp": 35, "pot_hc": 25, "pot_soir": 30, "pot_nuit": 10},
        "summary": ("Alternance entre rythme universitaire (matin et fin d'après-midi) "
                    "et déplacements sociaux en soirée."),
    },
}


def _persona_blend(persona_id: str, intensity: int) -> tuple[int, int, dict]:
    """Linearly blend the 'Équilibré' anchor toward the persona at `intensity` %."""
    base_confort, base_budget = 50, 60
    base_mix = {"pot_hp": 40, "pot_hc": 30, "pot_soir": 20, "pot_nuit": 10}
    p = PERSONAS[persona_id]
    t = intensity / 100.0
    confort = int(round(base_confort + t * (p["confort"] - base_confort)))
    budget = int(round(base_budget + t * (p["budget"] - base_budget)))
    mix = {
        k: float(base_mix[k] + t * (p["mix"][k] - base_mix[k]))
        for k in base_mix
    }
    return confort, budget, mix


# ============================================================================
# Vues Marketing — five thematic entry points sharing run_optimization.
# Each _render_vueN owns its own session-state keys (mkt_v{N}_current, _A, _B).
# ============================================================================

def _knob_summary_v1(s: dict) -> str:
    k = s.get("knobs", {})
    mix = k.get("mix", {})
    return (
        f"Confort/Couverture: **{k.get('confort', '?')}**  ·  "
        f"Budget flotte: **{k.get('budget', '?')}**  ·  "
        f"Mix: P{mix.get('pot_hp', 0)*100:.0f} / J{mix.get('pot_hc', 0)*100:.0f} / "
        f"S{mix.get('pot_soir', 0)*100:.0f} / N{mix.get('pot_nuit', 0)*100:.0f}"
    )


def _render_vue1(gtfs_path: str) -> None:
    """V1 — Pilotage stratégique : Confort/Couverture, Budget flotte, Mix temporel."""
    st.markdown("""
    Ré-exprime l'optimisation en **expérience voyageur** et **promesse réseau**.
    Réglez les deux curseurs et le mix temporel, lancez le scénario, puis
    verrouillez-le pour comparer plusieurs stratégies dans le **Comparateur**.
    """)
    st.subheader("Réglages du scénario")
    col_a, col_b = st.columns(2)
    with col_a:
        confort = st.slider(
            "Confort voyageur ↔ Couverture territoriale",
            0, 100, 50, 5, key="v1_confort",
            help=("0 = couverture du territoire (plus de correspondances). "
                  "100 = trajets directs sur les axes forts. 50 = équilibré."),
        )
        st.caption("← Couverture territoriale    |    Confort voyageur →")
    with col_b:
        budget = st.slider(
            "Budget flotte", 0, 100, 60, 5, key="v1_budget",
            help="0 = 3 lignes seulement. 100 = jusqu'à 20 lignes.",
        )
        st.caption("← Flotte légère    |    Flotte généreuse →")

    st.subheader("Mix temporel")
    st.caption("Quels moments de la journée prioriser ? Normalisé automatiquement.")
    mix_cols = st.columns(4)
    mix_hp = mix_cols[0].number_input("Heure de pointe", 0, 100, 40, 5, key="v1_mix_hp")
    mix_hc = mix_cols[1].number_input("Journée (creuse)", 0, 100, 30, 5, key="v1_mix_hc")
    mix_soir = mix_cols[2].number_input("Soirée", 0, 100, 20, 5, key="v1_mix_soir")
    mix_nuit = mix_cols[3].number_input("Nuit", 0, 100, 10, 5, key="v1_mix_nuit")

    raw_mix = {"pot_hp": float(mix_hp), "pot_hc": float(mix_hc),
               "pot_soir": float(mix_soir), "pot_nuit": float(mix_nuit)}
    norm_mix = _normalize_mix(raw_mix)
    mix_summary = "  ·  ".join(
        f"{lbl}: {norm_mix[k]*100:.0f} %"
        for k, lbl in [("pot_hp", "Pointe"), ("pot_hc", "Journée"),
                       ("pot_soir", "Soirée"), ("pot_nuit", "Nuit")]
    )
    st.caption(f"**Pondération effective :** {mix_summary}")

    knobs = {"confort": confort, "budget": budget, "mix": norm_mix}
    mkt_params = _marketing_to_params(confort, budget, raw_mix)

    with st.expander("Détails techniques (pour info)", expanded=False):
        st.json({
            "Confort vs Couverture (0-100)": confort,
            "Budget flotte (0-100)": budget,
            "Mix temporel": {k: round(v, 3) for k, v in norm_mix.items()},
            "→ alpha_pass": round(mkt_params["alpha_pass"], 3),
            "→ alpha_adeq": round(mkt_params["alpha_adeq"], 3),
            "→ alpha_oper": round(mkt_params["alpha_oper"], 4),
            "→ max_routes": mkt_params["max_routes"],
        })

    if st.button("Lancer ce scénario", type="primary", use_container_width=True,
                 key="v1_run"):
        out = run_optimization(mkt_params, gtfs_path)
        if out is not None:
            st.session_state["mkt_v1_current"] = {"results": out, "knobs": knobs}

    current = st.session_state.get("mkt_v1_current")
    if current is None:
        st.info("Cliquez sur **Lancer ce scénario** pour voir les résultats.")
        return

    res = current["results"]
    st.markdown("---")
    st.subheader("Indicateurs voyageur")
    _render_headline_panel(_extract_mkt_values(res))

    st.markdown("---")
    st.subheader("Carte du réseau optimisé")
    _render_map_for_results(
        res, f"V1 — {res['n_routes']} lignes (heure de pointe)")

    st.markdown("---")
    st.subheader("Comparateur de scénarios")
    sc_A, sc_B = _render_lock_swap_reset("1", current)

    if sc_A is None or sc_B is None:
        st.info("Verrouillez deux scénarios pour afficher le comparateur.")
    else:
        st.markdown(f"**Scénario A** — {_knob_summary_v1(sc_A)}")
        st.markdown(f"**Scénario B** — {_knob_summary_v1(sc_B)}")
        ledger_df = render_ledger(sc_A, sc_B)
        st.dataframe(_styled_dataframe(ledger_df), use_container_width=True,
                     hide_index=True)
        st.caption("↑ vert = scénario B améliore l'indicateur ·  ↓ rouge = il le dégrade ·  — = neutre.")


# ----------------------------------------------------------------------------
# V2 — Cadrage opérationnel
# ----------------------------------------------------------------------------

def _render_vue2(gtfs_path: str) -> None:
    st.markdown("""
    Vue **opérationnelle** : vous fixez directement la **taille du parc** et
    la **cadence des bus**.  Idéal pour parler chiffres avec l'exploitation.
    """)
    st.subheader("Cadrage de la flotte")
    col1, col2 = st.columns(2)
    min_lignes = col1.slider("Lignes minimum", 3, 15, 5, 1, key="v2_minl")
    max_lignes = col2.slider("Lignes maximum", 3, 25, 15, 1, key="v2_maxl")
    if max_lignes < min_lignes:
        st.warning("Le maximum doit être ≥ minimum. Ajustement automatique.")
        max_lignes = min_lignes

    st.subheader("Cadence des bus")
    col3, col4 = st.columns(2)
    intervalle_pire = col3.slider(
        "Au pire, un bus toutes les … minutes", 10, 60, 60, 5, key="v2_pire",
        help="Plancher d'intervalle. Plus c'est petit, plus on impose une fréquence minimale.")
    intervalle_mieux = col4.slider(
        "Au mieux, un bus toutes les … minutes", 3, 30, 3, 1, key="v2_mieux",
        help="Plafond d'intervalle (intervalle minimal admissible).")
    if intervalle_mieux >= intervalle_pire:
        st.warning("'Au mieux' doit être < 'Au pire'.")

    f_min_vh = 60.0 / intervalle_pire
    f_max_vh = 60.0 / intervalle_mieux

    st.subheader("Tolérance correspondance")
    transfer_pen = st.slider(
        "Pénalité d'une correspondance (minutes équivalentes ressenties)",
        0, 15, 5, 1, key="v2_pen",
        help=("Combien de minutes coûte 'ressenti' une correspondance. "
              "Plus c'est petit, plus on accepte facilement de faire changer le voyageur."),
    )

    knobs = {
        "min_lignes": min_lignes, "max_lignes": max_lignes,
        "intervalle_pire": intervalle_pire, "intervalle_mieux": intervalle_mieux,
        "transfer_pen": transfer_pen,
    }

    # Build params: balanced alphas + the operational overrides
    params = _marketing_to_params(50, 60, {"pot_hp": 40, "pot_hc": 30,
                                            "pot_soir": 20, "pot_nuit": 10})
    params["min_routes"] = min_lignes
    params["max_routes"] = max_lignes
    params["_f_min"] = f_min_vh
    params["_f_max"] = f_max_vh
    params["_transfer_penalty"] = float(transfer_pen)

    with st.expander("Détails techniques (pour info)", expanded=False):
        st.json({
            "min_routes": min_lignes, "max_routes": max_lignes,
            "f_min (veh/h)": round(f_min_vh, 2),
            "f_max (veh/h)": round(f_max_vh, 2),
            "transfer_penalty (min)": transfer_pen,
        })

    if st.button("Lancer ce scénario", type="primary", use_container_width=True,
                 key="v2_run"):
        out = run_optimization(params, gtfs_path)
        if out is not None:
            st.session_state["mkt_v2_current"] = {"results": out, "knobs": knobs}

    current = st.session_state.get("mkt_v2_current")
    if current is None:
        st.info("Cliquez sur **Lancer ce scénario** pour voir les résultats.")
        return

    res = current["results"]
    peak_freqs = res["freqs_pp"]["pot_hp"]
    positive = [f for f in peak_freqs if f > 0]
    avg_freq = sum(positive) / len(positive) if positive else 0
    interv_max_obs = 60.0 / min(positive) if positive else float("nan")
    interv_min_obs = 60.0 / max(positive) if positive else float("nan")
    veh_h = _compute_veh_hours(res["selected"], peak_freqs, res["G"])

    st.markdown("---")
    st.subheader("Tableau de bord opérationnel")
    op = st.columns(3)
    op[0].metric("Lignes au plan", f"{res['n_routes']}")
    op[1].metric("Fréquence moyenne en pointe", f"{avg_freq:.1f} bus/h")
    op[2].metric("Heures-bus / heure de service", f"{veh_h:.1f} h")

    op2 = st.columns(3)
    op2[0].metric("Intervalle observé le plus serré", f"{interv_min_obs:.1f} min")
    op2[1].metric("Intervalle observé le plus large", f"{interv_max_obs:.1f} min")
    op2[2].metric("Voyages directs", _mkt_format("d0", res["peak_metrics"].get("d0", 0)))

    op3 = st.columns(3)
    op3[0].metric("Demande non desservie",
                  _mkt_format("dun", res["peak_metrics"].get("dun", 0)))
    op3[1].metric("Voyages avec 1 correspondance",
                  _mkt_format("d1", res["peak_metrics"].get("d1", 0)))
    op3[2].metric("Voyages avec 2+ correspondances",
                  _mkt_format("d2plus",
                              res["peak_metrics"].get("d2", 0) + res["peak_metrics"].get("d3+", 0)))

    st.markdown("---")
    st.subheader("Carte du réseau optimisé")
    _render_map_for_results(
        res, f"V2 — {res['n_routes']} lignes (cadrage opérationnel)")

    st.markdown("---")
    st.subheader("Comparateur de scénarios opérationnels")
    sc_A, sc_B = _render_lock_swap_reset("2", current)

    if sc_A is None or sc_B is None:
        st.info("Verrouillez deux scénarios pour afficher le comparateur.")
    else:
        def _v2_summary(s):
            k = s.get("knobs", {})
            return (f"{k.get('min_lignes', '?')}–{k.get('max_lignes', '?')} lignes  ·  "
                    f"intervalle {k.get('intervalle_mieux', '?')}–{k.get('intervalle_pire', '?')} min  ·  "
                    f"pénalité corresp. {k.get('transfer_pen', '?')} min")
        st.markdown(f"**Scénario A** — {_v2_summary(sc_A)}")
        st.markdown(f"**Scénario B** — {_v2_summary(sc_B)}")

        # Build an op-flavoured ledger
        def _v2_vals(s):
            r = s["results"]
            pf = r["freqs_pp"]["pot_hp"]
            pos = [f for f in pf if f > 0]
            return {
                "Lignes au plan": f"{r['n_routes']}",
                "Fréquence moyenne en pointe (bus/h)":
                    f"{(sum(pos)/len(pos) if pos else 0):.1f}",
                "Intervalle le plus serré (min)":
                    f"{(60.0/max(pos) if pos else 0):.1f}",
                "Intervalle le plus large (min)":
                    f"{(60.0/min(pos) if pos else 0):.1f}",
                "Heures-bus / heure de service":
                    f"{_compute_veh_hours(r['selected'], pf, r['G']):.1f}",
                "Voyages directs (%)":
                    f"{r['peak_metrics'].get('d0', 0):.1f}",
                "Demande non desservie (%)":
                    f"{r['peak_metrics'].get('dun', 0):.1f}",
            }
        a_vals = _v2_vals(sc_A)
        b_vals = _v2_vals(sc_B)
        op_rows = [{"Indicateur": k, "Scénario A": a_vals[k], "Scénario B": b_vals[k]}
                   for k in a_vals]
        st.dataframe(pd.DataFrame(op_rows), use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# V3 — Promesse de service
# ----------------------------------------------------------------------------

def _render_vue3(gtfs_path: str) -> None:
    st.markdown("""
    Posez vos **promesses de service** comme une charte client.  L'optimiseur
    fournit un réseau, et nous vérifions si chaque promesse est **tenue** ou
    **manquée**.  Mode *Boucle fermée* : le solveur tente automatiquement
    de tenir toutes les promesses (jusqu'à 3 itérations).
    """)
    st.subheader("Vos promesses")
    col1, col2 = st.columns(2)
    with col1:
        promesse_d0 = st.slider(
            "Voyages directs promis (%)", 50, 100, 80, 5, key="v3_p_d0",
            help="« Au moins X % de nos voyageurs ne changent jamais de bus. »")
        promesse_pct_ok = st.slider(
            "Arrêts bien servis promis (%)", 50, 100, 75, 5, key="v3_p_ok",
            help="« Au moins X % des arrêts ont un service dimensionné à la demande. »")
    with col2:
        promesse_att = st.slider(
            "Temps moyen maximal promis (min)", 3.0, 15.0, 6.0, 0.5, key="v3_p_att",
            help="« Un trajet moyen dure au plus X minutes. »")
        promesse_intv = st.slider(
            "Intervalle maximal promis sur les axes forts (min)", 5, 30, 12, 1,
            key="v3_p_intv",
            help="« Sur la ligne la plus fréquente, un bus passe au moins toutes les X minutes. »")

    st.subheader("Mode")
    mode = st.radio(
        "Comment vérifier les promesses ?",
        ["Post-hoc (rapide)", "Boucle fermée (auto-réglage)"],
        index=0, key="v3_mode",
        help="Post-hoc lance une optimisation neutre puis compare. Boucle fermée tente "
             "de pousser le solveur jusqu'à tenir toutes les promesses.",
    )

    targets = {
        "d0": promesse_d0, "pct_ok": promesse_pct_ok,
        "ATT": promesse_att, "intv": promesse_intv,
    }
    knobs = {**targets, "mode": mode}

    if st.button("Lancer ce scénario", type="primary", use_container_width=True,
                 key="v3_run"):
        # Base params: balanced
        base_params = _marketing_to_params(
            50, 60, {"pot_hp": 40, "pot_hc": 30, "pot_soir": 20, "pot_nuit": 10})
        if mode.startswith("Post-hoc"):
            out = run_optimization(base_params, gtfs_path)
            iterations = []
            if out is not None:
                iterations.append({
                    "iter": 1, "alpha_pass": base_params["alpha_pass"],
                    "alpha_adeq": base_params["alpha_adeq"],
                    "f_min": base_params.get("_f_min"),
                    "result_summary": _v3_evaluate(out, targets),
                })
            if out is not None:
                st.session_state["mkt_v3_current"] = {
                    "results": out, "knobs": knobs, "iterations": iterations}
        else:
            iterations = []
            params = dict(base_params)
            out = None
            for it in range(1, 4):
                st.write(f"**Itération {it}/3** — alpha_pass={params['alpha_pass']:.2f}, "
                         f"alpha_adeq={params['alpha_adeq']:.2f}, "
                         f"f_min={params.get('_f_min', 'défaut')}")
                out = run_optimization(params, gtfs_path)
                if out is None:
                    break
                eval_ = _v3_evaluate(out, targets)
                iterations.append({
                    "iter": it, "alpha_pass": params["alpha_pass"],
                    "alpha_adeq": params["alpha_adeq"],
                    "f_min": params.get("_f_min"),
                    "result_summary": eval_,
                })
                if all(v["status"] == "Tenue" for v in eval_.values()):
                    st.success(f"Toutes les promesses tenues à l'itération {it}.")
                    break
                # Bump alphas based on misses
                if eval_["d0"]["status"] == "Manquée":
                    params["alpha_pass"] *= 1.5
                if eval_["pct_ok"]["status"] == "Manquée":
                    params["alpha_adeq"] *= 2.0
                if eval_["ATT"]["status"] == "Manquée":
                    params["alpha_pass"] *= 1.3
                if eval_["intv"]["status"] == "Manquée":
                    params["_f_min"] = float(params.get("_f_min", 1.0)) + 1.0
            if out is not None:
                st.session_state["mkt_v3_current"] = {
                    "results": out, "knobs": knobs, "iterations": iterations}

    current = st.session_state.get("mkt_v3_current")
    if current is None:
        st.info("Cliquez sur **Lancer ce scénario** pour évaluer vos promesses.")
        return

    res = current["results"]
    eval_ = _v3_evaluate(res, current["knobs"])

    # Score header
    n_kept = sum(1 for v in eval_.values() if v["status"] == "Tenue")
    total = len(eval_)
    if n_kept >= 3:
        bar_color = "#1b8e3a"; flag = "🎯"
    elif n_kept == 2:
        bar_color = "#d68910"; flag = "⚖"
    else:
        bar_color = "#c0392b"; flag = "⚠"

    st.markdown("---")
    st.subheader("Score de la charte")
    st.markdown(
        f"<div style='font-size:28px;font-weight:700;color:{bar_color}'>"
        f"{flag} Promesses tenues : {n_kept} / {total}</div>",
        unsafe_allow_html=True,
    )
    st.progress(n_kept / total)

    # Scorecard table
    rows = []
    label_map = {
        "d0": ("Voyages directs", "%"),
        "pct_ok": ("Arrêts bien servis", "%"),
        "ATT": ("Temps moyen", "min"),
        "intv": ("Intervalle max sur axes forts", "min"),
    }
    op_map = {"d0": "≥", "pct_ok": "≥", "ATT": "≤", "intv": "≤"}
    for key, ev in eval_.items():
        lbl, unit = label_map[key]
        rows.append({
            "Promesse": lbl,
            "Cible": f"{op_map[key]} {ev['target']:.1f} {unit}",
            "Réalisé": f"{ev['actual']:.1f} {unit}",
            "Statut": ev["status"],
            "Δ": ev["delta_str"],
        })

    def _style_status(val):
        if val == "Tenue": return "color: #1b8e3a; font-weight: 700;"
        if val == "Manquée": return "color: #c0392b; font-weight: 700;"
        return ""

    df = pd.DataFrame(rows)
    styler = df.style
    apply_fn = getattr(styler, "map", None) or styler.applymap
    st.dataframe(apply_fn(_style_status, subset=["Statut"]),
                 use_container_width=True, hide_index=True)

    # Iterations log
    iters = current.get("iterations", [])
    if len(iters) > 1:
        with st.expander(f"Journal de la boucle fermée ({len(iters)} itérations)",
                         expanded=False):
            for it in iters:
                kept = sum(1 for v in it["result_summary"].values()
                           if v["status"] == "Tenue")
                st.markdown(
                    f"**Itération {it['iter']}** — α_pass={it['alpha_pass']:.2f}, "
                    f"α_adeq={it['alpha_adeq']:.2f}, "
                    f"f_min={it['f_min']}  →  promesses tenues : {kept}/4")

    st.markdown("---")
    st.subheader("Carte du réseau optimisé")
    _render_map_for_results(res, f"V3 — promesses tenues {n_kept}/{total}")

    st.markdown("---")
    st.subheader("Comparateur de chartes")
    sc_A, sc_B = _render_lock_swap_reset("3", current)

    if sc_A is None or sc_B is None:
        st.info("Verrouillez deux chartes pour afficher le comparateur.")
    else:
        def _v3_card(s):
            k = s["knobs"]
            return (f"Directs ≥ {k['d0']}%  ·  Bien servis ≥ {k['pct_ok']}%  ·  "
                    f"ATT ≤ {k['ATT']} min  ·  Intervalle ≤ {k['intv']} min")
        st.markdown(f"**Charte A** — {_v3_card(sc_A)}")
        st.markdown(f"**Charte B** — {_v3_card(sc_B)}")

        a_eval = _v3_evaluate(sc_A["results"], sc_A["knobs"])
        b_eval = _v3_evaluate(sc_B["results"], sc_B["knobs"])
        a_kept = sum(1 for v in a_eval.values() if v["status"] == "Tenue")
        b_kept = sum(1 for v in b_eval.values() if v["status"] == "Tenue")
        st.markdown(f"**Charte A** : {a_kept}/4 tenues  ·  **Charte B** : {b_kept}/4 tenues")

        rows = []
        for key, lbl in [("d0", "Voyages directs"), ("pct_ok", "Arrêts bien servis"),
                         ("ATT", "Temps moyen"), ("intv", "Intervalle max")]:
            rows.append({
                "Promesse": lbl,
                "Charte A — réalisé": f"{a_eval[key]['actual']:.1f}",
                "Charte A — statut": a_eval[key]["status"],
                "Charte B — réalisé": f"{b_eval[key]['actual']:.1f}",
                "Charte B — statut": b_eval[key]["status"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _v3_evaluate(results: dict, targets: dict) -> dict:
    """Compare an optimization result against the V3 promise targets."""
    pf = results["freqs_pp"].get("pot_hp", [])
    positive = [f for f in pf if f > 0]
    intv_obs_max = (60.0 / max(positive)) if positive else float("inf")
    actuals = {
        "d0": results["peak_metrics"].get("d0", 0),
        "pct_ok": results["peak_adeq"].get("pct_ok", 0),
        "ATT": results["peak_metrics"].get("ATT", 0),
        "intv": intv_obs_max,
    }
    out = {}
    for key, target in targets.items():
        if key not in actuals:  # ignore "mode"
            continue
        a = actuals[key]
        if key in ("d0", "pct_ok"):
            ok = a >= target
            delta = a - target
            unit = "pp"
        else:  # ATT, intv: lower-better
            ok = a <= target
            delta = a - target
            unit = "min" if key == "ATT" or key == "intv" else ""
        out[key] = {
            "target": float(target),
            "actual": float(a),
            "status": "Tenue" if ok else "Manquée",
            "delta": float(delta),
            "delta_str": f"{delta:+.1f} {unit}",
        }
    return out


# ----------------------------------------------------------------------------
# V4 — Public cible (persona)
# ----------------------------------------------------------------------------

def _render_vue4(gtfs_path: str) -> None:
    st.markdown("""
    Choisissez un **public cible** et l'**intensité** de la promesse pour ce
    public.  Le réseau est optimisé pour servir prioritairement ce profil.
    Le tableau de bord ci-dessous met en avant les **KPI propres au persona**.
    """)
    st.subheader("Public cible")
    persona_id = st.selectbox(
        "Persona", list(PERSONAS), index=0, key="v4_persona",
        format_func=lambda p: PERSONAS[p]["label"],
    )
    st.caption(PERSONAS[persona_id]["summary"])

    intensity = st.slider(
        "Intensité de la promesse", 0, 100, 70, 5, key="v4_intensity",
        help=("0 = profil neutre (Équilibré).  100 = on optimise à fond pour ce "
              "public, au risque de moins bien servir les autres."),
    )

    confort, budget, raw_mix = _persona_blend(persona_id, intensity)
    norm_mix = _normalize_mix(raw_mix)
    knobs = {"persona": persona_id, "intensity": intensity,
             "confort": confort, "budget": budget, "mix": norm_mix}
    params = _marketing_to_params(confort, budget, raw_mix)

    with st.expander("Détails techniques (pour info)", expanded=False):
        st.json({
            "persona": persona_id, "intensity": intensity,
            "confort_eq": confort, "budget_eq": budget,
            "mix": {k: round(v, 3) for k, v in norm_mix.items()},
            "→ alpha_pass": round(params["alpha_pass"], 3),
            "→ alpha_adeq": round(params["alpha_adeq"], 3),
            "→ alpha_oper": round(params["alpha_oper"], 4),
            "→ max_routes": params["max_routes"],
        })

    if st.button("Lancer ce scénario", type="primary", use_container_width=True,
                 key="v4_run"):
        out = run_optimization(params, gtfs_path)
        if out is not None:
            st.session_state["mkt_v4_current"] = {"results": out, "knobs": knobs}

    current = st.session_state.get("mkt_v4_current")
    if current is None:
        st.info("Cliquez sur **Lancer ce scénario** pour voir les résultats.")
        return

    res = current["results"]
    st.markdown("---")
    st.subheader(f"Tableau de bord — {persona_id}")
    df = _persona_ledger(res, current["knobs"]["persona"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Vue voyageur générale")
    _render_headline_panel(_extract_mkt_values(res))

    st.markdown("---")
    st.subheader("Carte du réseau optimisé")
    _render_map_for_results(res, f"V4 — Persona « {persona_id} »")

    st.markdown("---")
    st.subheader("Comparateur de personas / intensités")
    sc_A, sc_B = _render_lock_swap_reset("4", current)

    if sc_A is None or sc_B is None:
        st.info("Verrouillez deux scénarios pour afficher le comparateur.")
    else:
        def _v4_summary(s):
            k = s["knobs"]
            return f"« {k['persona']} » à intensité **{k['intensity']}%**"
        st.markdown(f"**Scénario A** — {_v4_summary(sc_A)}")
        st.markdown(f"**Scénario B** — {_v4_summary(sc_B)}")
        ledger_df = render_ledger(sc_A, sc_B)
        st.dataframe(_styled_dataframe(ledger_df), use_container_width=True,
                     hide_index=True)


def _persona_ledger(res: dict, persona_id: str) -> pd.DataFrame:
    """Build a persona-specific KPI scorecard."""
    pr = res.get("period_results", {})
    pf = res.get("freqs_pp", {})
    rows = []

    def _m(band, key, default=0):
        return pr.get(band, {}).get("metrics", {}).get(key, default)

    def _a(band, key, default=0):
        return pr.get(band, {}).get("adeq", {}).get(key, default)

    def _n_high_freq(band, threshold=10.0):
        return sum(1 for f in pf.get(band, []) if f >= threshold)

    def _n_served(band):
        return sum(1 for f in pf.get(band, []) if f > 0)

    if persona_id == "Pendulaires":
        rows = [
            {"KPI Pendulaires": "Voyages directs en pointe",
             "Valeur": f"{_m('pot_hp', 'd0'):.1f} %"},
            {"KPI Pendulaires": "Temps moyen en pointe",
             "Valeur": f"{_m('pot_hp', 'ATT'):.2f} min"},
            {"KPI Pendulaires": "Lignes en pointe avec ≥10 bus/h",
             "Valeur": f"{_n_high_freq('pot_hp', 10):d}"},
            {"KPI Pendulaires": "Demande non desservie en pointe",
             "Valeur": f"{_m('pot_hp', 'dun'):.2f} %"},
        ]
    elif persona_id == "Vie nocturne":
        rows = [
            {"KPI Vie nocturne": "Voyages directs en soirée",
             "Valeur": f"{_m('pot_soir', 'd0'):.1f} %"},
            {"KPI Vie nocturne": "Voyages directs la nuit",
             "Valeur": f"{_m('pot_nuit', 'd0'):.1f} %"},
            {"KPI Vie nocturne": "Arrêts bien servis la nuit",
             "Valeur": f"{_a('pot_nuit', 'pct_ok'):.1f} %"},
            {"KPI Vie nocturne": "Lignes actives la nuit",
             "Valeur": f"{_n_served('pot_nuit'):d}"},
        ]
    elif persona_id == "Familles & loisirs":
        rows = [
            {"KPI Familles & loisirs": "Arrêts bien servis en heure creuse",
             "Valeur": f"{_a('pot_hc', 'pct_ok'):.1f} %"},
            {"KPI Familles & loisirs": "Temps moyen en heure creuse",
             "Valeur": f"{_m('pot_hc', 'ATT'):.2f} min"},
            {"KPI Familles & loisirs": "Voyages directs en heure creuse",
             "Valeur": f"{_m('pot_hc', 'd0'):.1f} %"},
            {"KPI Familles & loisirs": "Lignes actives en heure creuse",
             "Valeur": f"{_n_served('pot_hc'):d}"},
        ]
    elif persona_id == "Inclusion territoriale":
        pcts = [_a(b, "pct_ok") for b in ("pot_hp", "pot_hc", "pot_soir", "pot_nuit")]
        avg = sum(pcts) / len(pcts) if pcts else 0
        std = (sum((x - avg) ** 2 for x in pcts) / len(pcts)) ** 0.5 if pcts else 0
        n_total = len(res.get("node_coords", {})) or 1
        n_served_any = sum(1 for f in pf.get("pot_hp", []) if f > 0)
        rows = [
            {"KPI Inclusion": "Arrêts bien servis (moyenne 4 périodes)",
             "Valeur": f"{avg:.1f} %"},
            {"KPI Inclusion": "Écart-type entre périodes (équité)",
             "Valeur": f"{std:.1f} pp (plus c'est bas, plus c'est équitable)"},
            {"KPI Inclusion": "Couverture (lignes actives)",
             "Valeur": f"{n_served_any:d} lignes"},
            {"KPI Inclusion": "Demande non desservie (pondérée)",
             "Valeur": f"{(sum(_m(b, 'dun') for b in ('pot_hp', 'pot_hc', 'pot_soir', 'pot_nuit'))/4):.2f} %"},
        ]
    elif persona_id == "Étudiants":
        rows = [
            {"KPI Étudiants": "Voyages directs en pointe",
             "Valeur": f"{_m('pot_hp', 'd0'):.1f} %"},
            {"KPI Étudiants": "Voyages directs en soirée",
             "Valeur": f"{_m('pot_soir', 'd0'):.1f} %"},
            {"KPI Étudiants": "Voyages avec 1 correspondance en pointe",
             "Valeur": f"{_m('pot_hp', 'd1'):.1f} %"},
            {"KPI Étudiants": "Arrêts bien servis en soirée",
             "Valeur": f"{_a('pot_soir', 'pct_ok'):.1f} %"},
        ]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# V5 — Comparateur territorial (zones)
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _v5_load_stops(gtfs_path: str) -> pd.DataFrame:
    """Read GTFS stops.txt for the zone construction."""
    path = Path(gtfs_path) / "stops.txt"
    df = pd.read_csv(path)
    return df[["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()


def _v5_build_zones(stops_df: pd.DataFrame) -> dict:
    """Bucket each stop into a quadrant relative to the median centroid.
    Returns dict {stop_id: zone_label}.  Caller maps stop_id → node_id."""
    lat_c = stops_df["stop_lat"].median()
    lon_c = stops_df["stop_lon"].median()
    out = {}
    for _, row in stops_df.iterrows():
        n = row["stop_lat"] >= lat_c
        e = row["stop_lon"] >= lon_c
        out[row["stop_id"]] = (
            "Nord-Est" if n and e else
            "Nord-Ouest" if n and not e else
            "Sud-Est" if not n and e else
            "Sud-Ouest"
        )
    return out


def _render_vue5(gtfs_path: str) -> None:
    st.markdown("""
    Découpe le territoire en **4 zones** (Nord-Est / Nord-Ouest / Sud-Est /
    Sud-Ouest) à partir du centroïde des arrêts.  Réglez l'**Importance**
    accordée à chaque zone : 100 = neutre, 200 = double la demande captée
    dans cette zone, 50 = la divise par deux.
    """)

    try:
        stops_df = _v5_load_stops(gtfs_path)
    except Exception as e:
        st.error(f"Impossible de lire stops.txt : {e}")
        return

    zones_by_stop_id = _v5_build_zones(stops_df)
    zones_count = {z: 0 for z in ("Nord-Est", "Nord-Ouest", "Sud-Est", "Sud-Ouest")}
    for z in zones_by_stop_id.values():
        zones_count[z] = zones_count.get(z, 0) + 1

    st.subheader("Importance par zone")
    z_cols = st.columns(4)
    weights = {}
    for col, zname in zip(z_cols, ("Nord-Est", "Nord-Ouest", "Sud-Est", "Sud-Ouest")):
        w = col.slider(
            f"Importance {zname}", 50, 200, 100, 5,
            key=f"v5_w_{zname}",
            help=f"{zones_count[zname]} arrêts dans cette zone.",
        )
        weights[zname] = w
        col.caption(f"{zones_count[zname]} arrêts  ·  ×{w/100:.2f}")

    st.subheader("Cadrage flotte (commun aux quatre zones)")
    col_b1, col_b2 = st.columns(2)
    confort_v5 = col_b1.slider("Confort voyageur ↔ Couverture", 0, 100, 50, 5,
                                key="v5_confort")
    budget_v5 = col_b2.slider("Budget flotte", 0, 100, 60, 5, key="v5_budget")

    knobs = {"weights": weights, "confort": confort_v5, "budget": budget_v5}

    if st.button("Lancer ce scénario territorial", type="primary",
                 use_container_width=True, key="v5_run"):
        params = _marketing_to_params(
            confort_v5, budget_v5,
            {"pot_hp": 40, "pot_hc": 30, "pot_soir": 20, "pot_nuit": 10})

        # Map stop_id → node_id by loading once cheaply through Capt-Temp
        try:
            sys.path.insert(0, CAPT_TEMP_DIR)
            from geocapt_loader import load_all
            data_for_zones = load_all(csv_path=str(POM_CSV), pot_col="pot_hp",
                                       use_new_od=True)
            points = data_for_zones.get("points", [])
        except Exception as e:
            st.error(f"Échec de la cartographie zones → noeuds : {e}")
            return

        stop_id_to_node = {p.get("stop_id"): p.get("id") for p in points
                           if p.get("stop_id") is not None}
        zone_factors = {}
        for sid, zname in zones_by_stop_id.items():
            nid = stop_id_to_node.get(sid)
            if nid is not None:
                zone_factors[nid] = weights[zname] / 100.0

        params["_zone_factors"] = zone_factors
        knobs["zone_factors_count"] = len(zone_factors)
        knobs["zones_by_node"] = {}
        for sid, zname in zones_by_stop_id.items():
            nid = stop_id_to_node.get(sid)
            if nid is not None:
                knobs["zones_by_node"][nid] = zname

        out = run_optimization(params, gtfs_path)
        if out is not None:
            st.session_state["mkt_v5_current"] = {
                "results": out, "knobs": knobs,
                "stops_df": stops_df,
                "zones_by_stop_id": zones_by_stop_id,
            }

    current = st.session_state.get("mkt_v5_current")
    if current is None:
        st.info("Cliquez sur **Lancer ce scénario territorial** pour voir le découpage.")
        return

    res = current["results"]
    knobs = current["knobs"]
    zones_by_node = knobs.get("zones_by_node", {})

    st.markdown("---")
    st.subheader("Tableau de bord par zone")

    # Aggregate per zone
    selected = res["selected"]
    adeq_status = res.get("adeq_node_status", {})

    zone_rows = []
    for zname in ("Nord-Est", "Nord-Ouest", "Sud-Est", "Sud-Ouest"):
        nodes_in_zone = {n for n, z in zones_by_node.items() if z == zname}
        n_stops = len(nodes_in_zone)
        n_lines_traversing = sum(
            1 for r in selected if any(n in nodes_in_zone for n in r))
        statuses = [adeq_status.get(n, "none") for n in nodes_in_zone]
        n_ok = sum(1 for s in statuses if s == "ok")
        n_under = sum(1 for s in statuses if s == "under")
        n_over = sum(1 for s in statuses if s == "over")
        pct_ok = (100.0 * n_ok / max(n_stops, 1)) if n_stops else 0.0
        zone_rows.append({
            "Zone": zname,
            "Pondération": f"×{weights[zname]/100:.2f}",
            "Arrêts": n_stops,
            "Lignes traversantes": n_lines_traversing,
            "Arrêts bien servis": f"{pct_ok:.0f} %",
            "Arrêts saturés": n_under,
            "Bus à vide": n_over,
        })

    st.dataframe(pd.DataFrame(zone_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Indicateurs voyageur — global")
    _render_headline_panel(_extract_mkt_values(res))

    st.markdown("---")
    st.subheader("Carte du réseau optimisé")
    _render_map_for_results(res, "V5 — Comparateur territorial")


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.image(str(DEMO_DIR / "Transdev_logo_2018.png"), width=180)
st.sidebar.title("GeoCapt-UTRP Demo")
st.sidebar.markdown("---")

# Step indicator
step = st.sidebar.radio(
    "Navigation",
    ["1 - Données GTFS", "2 - Cadre POT", "3 - Priorités", "4 - Exécution & Résultats",
     "5 - Vues Marketing"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("Transdev")


# ============================================================================
# STEP 1 — GTFS DATA
# ============================================================================

if step == "1 - Données GTFS":
    st.header("Étape 1 : Données GTFS")
    st.markdown("""
    Téléversez vos fichiers GTFS ou utilisez le jeu de données **GTFS Poissy préchargé**.
    Le GTFS fournit les localisations des arrêts de bus, les lignes et les horaires qui
    forment la topologie du réseau pour l'optimisation.
    """)

    use_default = st.toggle("Utiliser le GTFS Poissy préchargé", value=True)

    if use_default:
        gtfs_path = GTFS_DIR
        st.success(f"Utilisation du GTFS préchargé depuis `{gtfs_path}`")

        # Show GTFS summary
        stops_df = pd.read_csv(gtfs_path / "stops.txt")
        routes_df = pd.read_csv(gtfs_path / "routes.txt")
        trips_df = pd.read_csv(gtfs_path / "trips.txt")

        col1, col2, col3 = st.columns(3)
        col1.metric("Arrêts", len(stops_df))
        col2.metric("Lignes", len(routes_df))
        col3.metric("Voyages", len(trips_df))

        st.subheader("Arrêts de bus")
        st.dataframe(
            stops_df[["stop_id", "stop_name", "stop_lat", "stop_lon"]],
            use_container_width=True, height=300,
        )

        # Quick map of stops
        st.subheader("Localisation des arrêts")
        st.map(stops_df.rename(columns={"stop_lat": "latitude", "stop_lon": "longitude"}))

        st.session_state["gtfs_path"] = str(gtfs_path)

    else:
        st.info("Téléversez ci-dessous les 7 fichiers GTFS requis :")
        uploaded = {}
        for fname in GTFS_REQUIRED:
            f = st.file_uploader(fname, type=["txt", "csv"], key=f"gtfs_{fname}")
            if f is not None:
                uploaded[fname] = f

        if len(uploaded) == len(GTFS_REQUIRED):
            custom_dir = DEMO_DIR / "data" / "GTFS_custom"
            save_uploaded_gtfs(uploaded, custom_dir)
            st.success(f"Les {len(GTFS_REQUIRED)} fichiers GTFS ont été téléversés !")

            stops_df = pd.read_csv(custom_dir / "stops.txt")
            st.metric("Arrêts chargés", len(stops_df))
            st.map(stops_df.rename(columns={"stop_lat": "latitude", "stop_lon": "longitude"}))

            st.session_state["gtfs_path"] = str(custom_dir)
        elif uploaded:
            missing = [f for f in GTFS_REQUIRED if f not in uploaded]
            st.warning(f"Fichiers manquants : {', '.join(missing)}")


# ============================================================================
# STEP 2 — POT FRAMEWORK
# ============================================================================

elif step == "2 - Cadre POT":
    st.header("Étape 2 : Cadre de données POT")
    st.markdown("""
    Le cadre **POM (Potentiel Offre Mobilité)** fournit le potentiel de demande
    pour chaque cellule de la grille sur 4 périodes horaires. Il est pré-calculé
    à partir des données INSEE de population/emploi et calibré pour Poissy.
    """)

    if POM_CSV.exists():
        pom_df = pd.read_csv(POM_CSV)
        st.success(f"Données POM chargées : **{len(pom_df)} cellules de grille**")

        st.subheader("Potentiels par période")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("pot_hp (pointe)", f"{pom_df['pot_hp'].sum():.0f}", help="Potentiel heure de pointe")
        col2.metric("pot_hc (creuse)", f"{pom_df['pot_hc'].sum():.0f}", help="Potentiel heure creuse")
        col3.metric("pot_soir (soir)", f"{pom_df['pot_soir'].sum():.0f}", help="Potentiel soirée")
        col4.metric("pot_nuit (nuit)", f"{pom_df['pot_nuit'].sum():.0f}", help="Potentiel nuit")

        st.subheader("Pondérations par période (optimisation conjointe)")
        st.markdown("""
        | Période | Pondération | Description |
        |---------|-------------|-------------|
        | pot_hp | **0.40** | Heure de pointe |
        | pot_hc | **0.30** | Heure creuse |
        | pot_soir | **0.20** | Soirée |
        | pot_nuit | **0.10** | Nuit |
        """)

        st.subheader("Échantillon des données POM")
        st.dataframe(pom_df.head(20), use_container_width=True)

        st.subheader("Distribution de la demande")
        tab1, tab2 = st.tabs(["Heure de pointe (pot_hp)", "Toutes les périodes"])
        with tab1:
            st.bar_chart(pom_df["pot_hp"].sort_values(ascending=False).reset_index(drop=True))
        with tab2:
            st.line_chart(pom_df[["pot_hp", "pot_hc", "pot_soir", "pot_nuit"]].sort_values(
                "pot_hp", ascending=False).reset_index(drop=True))

        st.subheader("Distribution spatiale")
        map_df = pom_df[["y", "x", "pot_hp"]].rename(columns={"y": "latitude", "x": "longitude"})
        st.map(map_df, size="pot_hp")

    else:
        st.error("CSV POM introuvable. Veuillez placer `pom_poissy.csv` dans le répertoire data/.")


# ============================================================================
# STEP 3 — PRIORITIES
# ============================================================================

elif step == "3 - Priorités":
    st.header("Étape 3 : Priorités d'optimisation")
    st.markdown("""
    Choisissez comment équilibrer les deux objectifs principaux :
    - **Adéquation** : Faire correspondre l'offre (lignes + fréquence) à la demande à chaque arrêt
    - **Correspondances directes** : Maximiser les liaisons directes (d0) et minimiser les correspondances
    """)

    st.subheader("Mode de priorité")
    priority = st.radio(
        "Sélectionnez votre priorité :",
        [
            "Équilibré (recommandé)",
            "Priorité à l'adéquation",
            "Priorité aux correspondances directes",
            "Personnalisé",
        ],
        index=0,
        help="Ajuste les pondérations d'objectif dans le modèle MIP Gurobi",
    )

    # Preset configurations (tuned for fast demo runs)
    PRESETS = {
        "Équilibré (recommandé)": {
            "alpha_pass": 1.0, "alpha_adeq": 0.1, "alpha_oper": 0.01,
            "use_new_od": True, "route_choice": True,
            "freq_opt": True, "route_rules": True,
            "min_routes": 3, "max_routes": 15,
            "time_limit": 450, "max_iter": 3,
            "initial_count": 5000, "pricing_count": 500,
            "ls_rounds": 7,
        },
        "Priorité à l'adéquation": {
            "alpha_pass": 0.5, "alpha_adeq": 0.5, "alpha_oper": 0.01,
            "use_new_od": True, "route_choice": False,
            "freq_opt": False, "route_rules": False,
            "min_routes": 3, "max_routes": 15,
            "time_limit": 450, "max_iter": 3,
            "initial_count": 5000, "pricing_count": 500,
            "ls_rounds": 7,
        },
        "Priorité aux correspondances directes": {
            "alpha_pass": 1.0, "alpha_adeq": 0.01, "alpha_oper": 0.01,
            "use_new_od": False, "route_choice": True,
            "freq_opt": True, "route_rules": True,
            "min_routes": 3, "max_routes": 15,
            "time_limit": 450, "max_iter": 3,
            "initial_count": 5000, "pricing_count": 500,
            "ls_rounds": 7,
        },
    }

    if priority == "Personnalisé":
        st.subheader("Pondérations personnalisées")
        col1, col2, col3 = st.columns(3)
        alpha_pass = col1.slider("Alpha Passager", 0.0, 10.0, 1.0, 0.05,
                                 help="Pondération sur la minimisation du temps de trajet")
        alpha_adeq = col2.slider("Alpha Adéquation", 0.0, 1.0, 0.1, 0.01,
                                 help="Pondération sur l'équilibre offre-demande")
        alpha_oper = col3.slider("Alpha Opérateur", 0.0, 0.5, 0.01, 0.005,
                                 help="Pondération sur le coût opérateur (véhicules-heures)")

        st.subheader("Options du modèle")
        col1, col2 = st.columns(2)
        use_new_od = col1.checkbox("Nouveau modèle OD P/A gravitaire", value=True,
                                   help="Utiliser le modèle gravitaire production-attraction")
        route_choice = col1.checkbox("Choix de ligne (top-k)", value=True,
                                     help="Autoriser le choix de ligne parmi les top-k lignes directes")
        freq_opt = col2.checkbox("Optimisation des fréquences", value=True,
                                 help="Optimiser les fréquences (vs f_min fixé)")
        route_rules = col2.checkbox("Règles de qualité des lignes", value=True,
                                    help="Pénaliser les boucles, tiroirs, culs-de-sac")

        st.subheader("Paramètres du solveur")
        col1, col2, col3 = st.columns(3)
        min_routes = col1.number_input("Lignes min", 1, 25, 3)
        max_routes = col2.number_input("Lignes max", 3, 30, 15)
        time_limit = col3.number_input("Limite de temps (s)", 30, 1200, 450, step=30)
        col1, col2, col3 = st.columns(3)
        max_iter = col1.slider("Itérations CG", 1, 10, 3)
        initial_count = col2.number_input("Candidats initiaux", 500, 10000, 5000, step=500)
        ls_rounds = col3.slider("Tours de recherche locale", 0, 20, 7)

        params = {
            "alpha_pass": alpha_pass, "alpha_adeq": alpha_adeq,
            "alpha_oper": alpha_oper, "use_new_od": use_new_od,
            "route_choice": route_choice, "freq_opt": freq_opt,
            "route_rules": route_rules, "min_routes": min_routes,
            "max_routes": max_routes, "time_limit": time_limit,
            "max_iter": max_iter, "initial_count": initial_count,
            "pricing_count": 500, "ls_rounds": ls_rounds,
        }
    else:
        params = PRESETS[priority]

        # Display the preset
        st.subheader("Configuration")
        col1, col2, col3 = st.columns(3)
        col1.metric("Alpha Passager", f"{params['alpha_pass']:.2f}")
        col2.metric("Alpha Adéquation", f"{params['alpha_adeq']:.2f}")
        col3.metric("Alpha Opérateur", f"{params['alpha_oper']:.3f}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nouveau modèle OD", "Oui" if params["use_new_od"] else "Non")
        col2.metric("Choix de ligne", "Oui" if params["route_choice"] else "Non")
        col3.metric("Optim. fréq.", "Oui" if params["freq_opt"] else "Non")
        col4.metric("Règles de ligne", "Oui" if params["route_rules"] else "Non")

    st.session_state["params"] = params

    # Show what to expect
    st.markdown("---")
    st.subheader("Comportement attendu")
    if priority == "Priorité à l'adéquation":
        st.info("""
        **La priorité à l'adéquation** maximise l'équilibre offre-demande.
        Attendez-vous à des scores d'adéquation plus élevés (~77 %) mais potentiellement à plus de correspondances.
        Le modèle répartira les lignes uniformément sur les pôles de demande.
        """)
    elif priority == "Priorité aux correspondances directes":
        st.info("""
        **La priorité aux correspondances directes** maximise d0 (liaisons directes).
        Attendez-vous à un d0 très élevé (~99 %) et un ATT faible (~2 min), mais l'adéquation peut
        baisser (~66 %). Le modèle concentre les lignes sur les paires OD à forte demande.
        """)
    else:
        st.info("""
        **Le mode équilibré** trouve un bon compromis entre adéquation et correspondances.
        Attendez-vous à d0 ~27 %, ATT ~17 min, adéquation ~74 %.
        Les règles de qualité des lignes garantissent des lignes pratiques, sans boucles.
        """)


# ============================================================================
# STEP 4 — RUN & RESULTS
# ============================================================================

elif step == "4 - Exécution & Résultats":
    st.header("Étape 4 : Exécution de l'optimisation & résultats")

    params = st.session_state.get("params")
    gtfs_path = st.session_state.get("gtfs_path", str(GTFS_DIR))

    if params is None:
        st.warning("Veuillez d'abord configurer les priorités à l'étape 3.")
        st.stop()

    # Show current config
    with st.expander("Configuration actuelle", expanded=False):
        st.json(params)

    # Run button
    if st.button("Lancer l'optimisation", type="primary", use_container_width=True):
        out = run_optimization(params, gtfs_path)
        if out is not None:
            st.session_state["results"] = out

    # ----------------------------------------------------------------
    # Display results (persistent across reruns)
    # ----------------------------------------------------------------
    results = st.session_state.get("results")
    if results is None:
        st.info("Cliquez sur **Lancer l'optimisation** pour commencer.")
        st.stop()

    st.markdown("---")

    # Key metrics top bar
    m = results["peak_metrics"]
    a = results["peak_adeq"]
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("d0 (direct)", f"{m['d0']:.1f}%")
    c2.metric("d1 (1 correspondance)", f"{m['d1']:.1f}%")
    c3.metric("d2+", f"{m.get('d2', 0) + m.get('d3+', 0):.1f}%")
    c4.metric("dun (non desservi)", f"{m['dun']:.1f}%")
    c5.metric("ATT (min)", f"{m['ATT']:.2f}")
    c6.metric("Adéquation", f"{a['pct_ok']:.1f}%")
    c7.metric("Lignes", f"{results['n_routes']}")

    # Tabs for different views
    tab_map, tab_periods, tab_routes, tab_history = st.tabs([
        "Carte interactive", "Résultats par période", "Détails des lignes", "Historique CG",
    ])

    with tab_map:
        st.subheader("Carte interactive des lignes")
        html_str = generate_dashboard_html(
            results["selected"], results["freqs_pp"]["pot_hp"],
            results["node_coords"], results["G"],
            results["peak_metrics"], results["peak_adeq"],
            results["adeq_node_status"], results["stop_names"],
            title=f"GeoCapt Demo — {results['n_routes']} lignes (pot_hp)",
        )
        st.components.v1.html(html_str, height=700, scrolling=True)

        # Download button for the HTML
        st.download_button(
            "Télécharger la carte HTML",
            html_str,
            file_name="geocapt_demo_result.html",
            mime="text/html",
        )

    with tab_periods:
        st.subheader("Évaluation par période")

        # Build comparison table
        rows = []
        for band in ["pot_hp", "pot_hc", "pot_soir", "pot_nuit"]:
            pr = results["period_results"].get(band)
            if pr is None:
                continue
            pm = pr["metrics"]
            pa = pr["adeq"]
            avg_f = sum(pr["freqs"]) / len(pr["freqs"]) if pr["freqs"] else 0
            rows.append({
                "Période": band,
                "d0 %": round(pm["d0"], 1),
                "d1 %": round(pm["d1"], 1),
                "d2+ %": round(pm.get("d2", 0) + pm.get("d3+", 0), 1),
                "dun %": round(pm["dun"], 2),
                "ATT (min)": round(pm["ATT"], 2),
                "Adéq %": round(pa["pct_ok"], 1),
                "Sur-offre": pa["n_over"],
                "Sous-offre": pa["n_under"],
                "Fréq. moy.": round(avg_f, 2),
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Frequency table
        st.subheader("Tableau des fréquences par période")
        freq_rows = []
        for idx, route in enumerate(results["selected"]):
            row = {"Ligne": f"R{idx+1}", "Arrêts": len(route)}
            for band in ["pot_hp", "pot_hc", "pot_soir", "pot_nuit"]:
                row[band] = round(results["freqs_pp"][band][idx], 2)
            freq_rows.append(row)
        st.dataframe(pd.DataFrame(freq_rows), use_container_width=True, hide_index=True)

    with tab_routes:
        st.subheader("Lignes sélectionnées")
        for idx, route in enumerate(results["selected"]):
            freq = results["freqs_pp"]["pot_hp"][idx]
            color = _ROUTE_COLOURS[idx % len(_ROUTE_COLOURS)]
            names = [results["stop_names"].get(n, f"#{n}") for n in route]
            with st.expander(
                f"Ligne {idx+1} — {len(route)} arrêts, f={freq:.1f}/h",
                expanded=False,
            ):
                st.markdown(f"**Itinéraire :** {' -> '.join(names)}")
                st.markdown(f"**IDs des nœuds :** {' - '.join(map(str, route))}")

        # Coverage stats
        covered = set()
        for r in results["selected"]:
            covered.update(r)
        total_nodes = len(results["node_coords"])
        st.metric("Couverture des nœuds", f"{len(covered)}/{total_nodes} ({100*len(covered)/total_nodes:.0f}%)")

    with tab_history:
        st.subheader("Historique de la génération de colonnes")
        hist = results.get("history", [])
        if hist:
            hist_rows = []
            for h in hist:
                impr = f"{h['improvement']:.2%}" if h["iteration"] > 1 else "---"
                hist_rows.append({
                    "Itération": h["iteration"],
                    "Taille du pool": h["pool_size"],
                    "Objectif": round(h["obj_value"], 2),
                    "Amélioration": impr,
                    "Temps (s)": round(h["time"], 0),
                })
            st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Aucun historique disponible.")

        st.metric("Temps total", f"{results['elapsed']:.0f} s")


# ============================================================================
# STEP 5 — MARKETING VIEW
# ============================================================================

elif step == "5 - Vues Marketing":
    st.header("Vues Marketing — pilotage du réseau")
    st.markdown(
        "Cinq angles de pilotage. Chaque vue a ses propres réglages, son "
        "propre tableau de bord, et son propre comparateur A/B."
    )
    gtfs_path = st.session_state.get("gtfs_path", str(GTFS_DIR))

    tab_v1, tab_v2, tab_v3, tab_v4, tab_v5 = st.tabs([
        "V1 — Pilotage stratégique",
        "V2 — Cadrage opérationnel",
        "V3 — Promesse de service",
        "V4 — Public cible",
        "V5 — Comparateur territorial",
    ])
    with tab_v1:
        _render_vue1(gtfs_path)
    with tab_v2:
        _render_vue2(gtfs_path)
    with tab_v3:
        _render_vue3(gtfs_path)
    with tab_v4:
        _render_vue4(gtfs_path)
    with tab_v5:
        _render_vue5(gtfs_path)

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


def _js_esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")


def _fetch_segment_geometry(
    a: tuple[float, float],
    b: tuple[float, float],
    access_token: str,
    profile: str = "driving",
) -> list[tuple[float, float]]:
    """Fetch the shortest driving path between two single waypoints.
    Returns list of (lat, lon).  Falls back to straight line."""
    import urllib.request

    coords_str = f"{a[1]},{a[0]};{b[1]},{b[0]}"  # lon,lat;lon,lat
    url = (
        f"https://api.mapbox.com/directions/v5/mapbox/{profile}/{coords_str}"
        f"?geometries=geojson&overview=full&access_token={access_token}"
    )
    try:
        resp = urllib.request.urlopen(urllib.request.Request(url))
        data = json.loads(resp.read())
        if data.get("code") == "Ok" and data.get("routes"):
            geojson_coords = data["routes"][0]["geometry"]["coordinates"]
            return [(lat, lon) for lon, lat in geojson_coords]
    except Exception:
        pass
    return [a, b]


def _fetch_road_geometry(
    waypoints: list[tuple[float, float]],
    access_token: str,
    profile: str = "driving",
) -> list[tuple[float, float]]:
    """Get road-snapped geometry for a waypoint sequence.

    Renders segment-by-segment: each consecutive pair (A->B, B->C, ...)
    is fetched independently and concatenated.  This avoids Mapbox's
    multi-waypoint trajectory optimiser, which can introduce visual
    U-turns/loops when the "best" path through all waypoints differs
    from the point-to-point shortest path.
    """
    if len(waypoints) < 2:
        return list(waypoints)

    full: list[tuple[float, float]] = []
    for i in range(len(waypoints) - 1):
        seg = _fetch_segment_geometry(waypoints[i], waypoints[i + 1],
                                       access_token, profile)
        if i == 0:
            full.extend(seg)
        else:
            # Avoid duplicating the junction point
            full.extend(seg[1:] if seg else [])
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
        # v2 suffix invalidates cache entries from the old API call
        # (without continue_straight=true, which caused visual U-turns)
        cache_key = "v2|" + ";".join(f"{lat:.6f},{lon:.6f}" for lat, lon in waypoints)
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
  <div class="metric"><div class="metric-val">{m_d1:.1f}%</div><div class="metric-lbl">d1 transfer</div></div>
  <div class="metric"><div class="metric-val">{m_d2p:.1f}%</div><div class="metric-lbl">d2+</div></div>
  <div class="metric"><div class="metric-val">{m_dun:.1f}%</div><div class="metric-lbl">dun unserved</div></div>
  <div class="metric"><div class="metric-val">{m_att:.2f}</div><div class="metric-lbl">ATT (min)</div></div>
  <div class="metric"><div class="metric-val">{m_pct_ok:.1f}%</div><div class="metric-lbl">Adequation</div></div>
  <div class="metric"><div class="metric-val">{total_veh_min:.0f}</div><div class="metric-lbl">Veh-min/h</div></div>
</div>
<div id="main">
<div id="map"></div>
<div id="sidebar">
  <h3>Adequation Legend</h3>
  <div class="legend-bar">
    <span><div class="legend-dot" style="background:#DAA520;"></div> OK</span>
    <span><div class="legend-dot" style="background:blue;"></div> Under-supply</span>
    <span><div class="legend-dot" style="background:red;"></div> Over-supply</span>
    <span><div class="legend-dot" style="background:gray;"></div> No data</span>
  </div>
  <h3>Routes ({len(routes)})</h3>
  <table class="routes">
    <thead><tr><th>#</th><th>f/h</th><th>Stops</th><th>Min</th><th>Itinerary</th></tr></thead>
    <tbody>{routes_table}</tbody>
  </table>
</div>
</div>
<script>
var map = L.map('map').setView([{avg_lat}, {avg_lon}], 14);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution: '&copy; OpenStreetMap'
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
        weight = max(2, min(6, ri["freq"] / 3))
        popup_stops = " &rarr; ".join(_js_esc(n) for n in ri["stop_names"])
        popup_text = _js_esc(
            f"<b>Route {ri['idx']}</b><br>"
            f"f={ri['freq']:.1f} veh/h, {ri['n_stops']} stops, {ri['time']:.1f} min<br>"
            f"<small>{popup_stops}</small>"
        )
        html += f"""
routeLayers[{ri['idx']}] = L.polyline([{coords_str}], {{
  color: '{ri['color']}', weight: {weight:.0f}, opacity: 0.75
}}).addTo(map).bindPopup('{popup_text}');
"""

    html += """
var selectedRoute = null;
var defaultWeights = {};
for (var id in routeLayers) { defaultWeights[id] = routeLayers[id].options.weight; }
function clearSelection() {
  for (var id in routeLayers) { routeLayers[id].setStyle({weight: defaultWeights[id], opacity: 0.75}); }
  selectedRoute = null;
  document.querySelectorAll('.route-row').forEach(function(r) { r.classList.remove('selected'); });
}
function selectRoute(id) {
  if (selectedRoute === id) { clearSelection(); return; }
  clearSelection(); selectedRoute = id;
  for (var rid in routeLayers) { routeLayers[rid].setStyle({weight: defaultWeights[rid], opacity: 0.15}); }
  routeLayers[id].setStyle({weight: 8, opacity: 1.0}); routeLayers[id].bringToFront();
  var row = document.querySelector('.route-row[data-route="' + id + '"]');
  if (row) row.classList.add('selected');
}
document.addEventListener('keydown', function(e) { if (e.key === 'Escape') clearSelection(); });
</script>
</body>
</html>"""
    return html


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.image(str(DEMO_DIR / "Transdev_logo_2018.png"), width=180)
st.sidebar.title("GeoCapt-UTRP Demo")
st.sidebar.markdown("---")

# Step indicator
step = st.sidebar.radio(
    "Navigation",
    ["1 - GTFS Data", "2 - POT Framework", "3 - Priorities", "4 - Run & Results"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("Transdev")


# ============================================================================
# STEP 1 — GTFS DATA
# ============================================================================

if step == "1 - GTFS Data":
    st.header("Step 1: GTFS Data")
    st.markdown("""
    Upload your GTFS files or use the **pre-loaded Poissy GTFS** dataset.
    The GTFS provides bus stop locations, routes, and schedules that form the
    network topology for optimisation.
    """)

    use_default = st.toggle("Use pre-loaded Poissy GTFS", value=True)

    if use_default:
        gtfs_path = GTFS_DIR
        st.success(f"Using pre-loaded GTFS from `{gtfs_path}`")

        # Show GTFS summary
        stops_df = pd.read_csv(gtfs_path / "stops.txt")
        routes_df = pd.read_csv(gtfs_path / "routes.txt")
        trips_df = pd.read_csv(gtfs_path / "trips.txt")

        col1, col2, col3 = st.columns(3)
        col1.metric("Stops", len(stops_df))
        col2.metric("Routes", len(routes_df))
        col3.metric("Trips", len(trips_df))

        st.subheader("Bus Stops")
        st.dataframe(
            stops_df[["stop_id", "stop_name", "stop_lat", "stop_lon"]],
            use_container_width=True, height=300,
        )

        # Quick map of stops
        st.subheader("Stop Locations")
        st.map(stops_df.rename(columns={"stop_lat": "latitude", "stop_lon": "longitude"}))

        st.session_state["gtfs_path"] = str(gtfs_path)

    else:
        st.info("Upload the 7 required GTFS files below:")
        uploaded = {}
        for fname in GTFS_REQUIRED:
            f = st.file_uploader(fname, type=["txt", "csv"], key=f"gtfs_{fname}")
            if f is not None:
                uploaded[fname] = f

        if len(uploaded) == len(GTFS_REQUIRED):
            custom_dir = DEMO_DIR / "data" / "GTFS_custom"
            save_uploaded_gtfs(uploaded, custom_dir)
            st.success(f"All {len(GTFS_REQUIRED)} GTFS files uploaded!")

            stops_df = pd.read_csv(custom_dir / "stops.txt")
            st.metric("Stops loaded", len(stops_df))
            st.map(stops_df.rename(columns={"stop_lat": "latitude", "stop_lon": "longitude"}))

            st.session_state["gtfs_path"] = str(custom_dir)
        elif uploaded:
            missing = [f for f in GTFS_REQUIRED if f not in uploaded]
            st.warning(f"Missing files: {', '.join(missing)}")


# ============================================================================
# STEP 2 — POT FRAMEWORK
# ============================================================================

elif step == "2 - POT Framework":
    st.header("Step 2: POT Data Framework")
    st.markdown("""
    The **POM (Potentiel Offre Mobilite)** framework provides demand potential
    for each grid cell across 4 time periods.  This is pre-computed from
    INSEE population/employment data and calibrated for Poissy.
    """)

    if POM_CSV.exists():
        pom_df = pd.read_csv(POM_CSV)
        st.success(f"POM data loaded: **{len(pom_df)} grid cells**")

        st.subheader("Period Potentials")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("pot_hp (peak)", f"{pom_df['pot_hp'].sum():.0f}", help="Rush hour potential")
        col2.metric("pot_hc (off-peak)", f"{pom_df['pot_hc'].sum():.0f}", help="Heure creuse potential")
        col3.metric("pot_soir (evening)", f"{pom_df['pot_soir'].sum():.0f}", help="Evening potential")
        col4.metric("pot_nuit (night)", f"{pom_df['pot_nuit'].sum():.0f}", help="Night potential")

        st.subheader("Period Weights (Joint Optimisation)")
        st.markdown("""
        | Period | Weight | Description |
        |--------|--------|-------------|
        | pot_hp | **0.40** | Peak hour (rush) |
        | pot_hc | **0.30** | Off-peak |
        | pot_soir | **0.20** | Evening |
        | pot_nuit | **0.10** | Night |
        """)

        st.subheader("POM Data Sample")
        st.dataframe(pom_df.head(20), use_container_width=True)

        st.subheader("Demand Distribution")
        tab1, tab2 = st.tabs(["Peak Hour (pot_hp)", "All Periods"])
        with tab1:
            st.bar_chart(pom_df["pot_hp"].sort_values(ascending=False).reset_index(drop=True))
        with tab2:
            st.line_chart(pom_df[["pot_hp", "pot_hc", "pot_soir", "pot_nuit"]].sort_values(
                "pot_hp", ascending=False).reset_index(drop=True))

        st.subheader("Spatial Distribution")
        map_df = pom_df[["y", "x", "pot_hp"]].rename(columns={"y": "latitude", "x": "longitude"})
        st.map(map_df, size="pot_hp")

    else:
        st.error("POM CSV not found. Please place `pom_poissy.csv` in the data/ directory.")


# ============================================================================
# STEP 3 — PRIORITIES
# ============================================================================

elif step == "3 - Priorities":
    st.header("Step 3: Optimisation Priorities")
    st.markdown("""
    Choose how to balance the two main objectives:
    - **Adequation**: Match supply (routes + frequency) to demand at each stop
    - **Direct Transfers**: Maximise direct connections (d0) and minimise transfers
    """)

    st.subheader("Priority Mode")
    priority = st.radio(
        "Select your focus:",
        [
            "Balanced (recommended)",
            "Focus on Adequation",
            "Focus on Direct Transfers",
            "Custom",
        ],
        index=0,
        help="This adjusts the objective weights in the Gurobi MIP model",
    )

    # Preset configurations (tuned for fast demo runs)
    PRESETS = {
        "Balanced (recommended)": {
            "alpha_pass": 1.0, "alpha_adeq": 0.1, "alpha_oper": 0.01,
            "use_new_od": True, "route_choice": True,
            "freq_opt": True, "route_rules": True,
            "min_routes": 3, "max_routes": 15,
            "time_limit": 60, "max_iter": 1,
            "initial_count": 1000, "pricing_count": 500,
            "ls_rounds": 3,
        },
        "Focus on Adequation": {
            "alpha_pass": 0.5, "alpha_adeq": 0.5, "alpha_oper": 0.01,
            "use_new_od": True, "route_choice": False,
            "freq_opt": False, "route_rules": False,
            "min_routes": 3, "max_routes": 15,
            "time_limit": 60, "max_iter": 1,
            "initial_count": 1000, "pricing_count": 500,
            "ls_rounds": 3,
        },
        "Focus on Direct Transfers": {
            "alpha_pass": 1.0, "alpha_adeq": 0.01, "alpha_oper": 0.01,
            "use_new_od": False, "route_choice": True,
            "freq_opt": True, "route_rules": True,
            "min_routes": 3, "max_routes": 15,
            "time_limit": 60, "max_iter": 1,
            "initial_count": 1000, "pricing_count": 500,
            "ls_rounds": 3,
        },
    }

    if priority == "Custom":
        st.subheader("Custom Weights")
        col1, col2, col3 = st.columns(3)
        alpha_pass = col1.slider("Alpha Passenger", 0.0, 2.0, 1.0, 0.05,
                                 help="Weight on travel time minimisation")
        alpha_adeq = col2.slider("Alpha Adequation", 0.0, 1.0, 0.1, 0.01,
                                 help="Weight on supply-demand balance")
        alpha_oper = col3.slider("Alpha Operator", 0.0, 0.5, 0.01, 0.005,
                                 help="Weight on operator cost (vehicle-hours)")

        st.subheader("Model Options")
        col1, col2 = st.columns(2)
        use_new_od = col1.checkbox("New P/A Gravity OD", value=True,
                                   help="Use production-attraction gravity model")
        route_choice = col1.checkbox("Route Choice (top-k)", value=True,
                                     help="Allow route choice among top-k direct routes")
        freq_opt = col2.checkbox("Frequency Optimisation", value=True,
                                 help="Optimise frequencies (vs fixed f_min)")
        route_rules = col2.checkbox("Route Quality Rules", value=True,
                                    help="Penalise loops, tiroirs, cul-de-sacs")

        st.subheader("Solver Settings")
        col1, col2, col3 = st.columns(3)
        min_routes = col1.number_input("Min Routes", 1, 25, 3)
        max_routes = col2.number_input("Max Routes", 3, 30, 15)
        time_limit = col3.number_input("Time Limit (s)", 30, 1200, 60, step=30)
        col1, col2, col3 = st.columns(3)
        max_iter = col1.slider("CG Iterations", 1, 10, 1)
        initial_count = col2.number_input("Initial Candidates", 500, 10000, 1000, step=500)
        ls_rounds = col3.slider("Local Search Rounds", 0, 20, 3)

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
        col1.metric("Alpha Passenger", f"{params['alpha_pass']:.2f}")
        col2.metric("Alpha Adequation", f"{params['alpha_adeq']:.2f}")
        col3.metric("Alpha Operator", f"{params['alpha_oper']:.3f}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("New OD Model", "Yes" if params["use_new_od"] else "No")
        col2.metric("Route Choice", "Yes" if params["route_choice"] else "No")
        col3.metric("Freq. Optim.", "Yes" if params["freq_opt"] else "No")
        col4.metric("Route Rules", "Yes" if params["route_rules"] else "No")

    st.session_state["params"] = params

    # Show what to expect
    st.markdown("---")
    st.subheader("Expected Behaviour")
    if priority == "Focus on Adequation":
        st.info("""
        **Adequation focus** maximises the supply-demand balance.
        Expect higher adequation scores (~77%) but potentially more transfers.
        The model will spread routes evenly across demand centres.
        """)
    elif priority == "Focus on Direct Transfers":
        st.info("""
        **Direct transfer focus** maximises d0 (direct connections).
        Expect very high d0 (~99%) and low ATT (~2 min) but adequation may
        drop (~66%).  The model concentrates routes on high-demand OD pairs.
        """)
    else:
        st.info("""
        **Balanced mode** finds a good trade-off between adequation and transfers.
        Expect d0 ~27%, ATT ~17 min, adequation ~74%.
        Route quality rules ensure practical, non-looping routes.
        """)


# ============================================================================
# STEP 4 — RUN & RESULTS
# ============================================================================

elif step == "4 - Run & Results":
    st.header("Step 4: Run Optimisation & Results")

    params = st.session_state.get("params")
    gtfs_path = st.session_state.get("gtfs_path", str(GTFS_DIR))

    if params is None:
        st.warning("Please configure priorities in Step 3 first.")
        st.stop()

    # Show current config
    with st.expander("Current Configuration", expanded=False):
        st.json(params)

    # Run button
    if st.button("Run Optimisation", type="primary", use_container_width=True):

        # ----------------------------------------------------------------
        # Import Capt-Temp pipeline modules
        # ----------------------------------------------------------------
        progress = st.progress(0, text="Importing pipeline modules...")

        try:
            import config as cfg

            # Override config GTFS path if using custom GTFS
            cfg.GTFS_STOPS_PATH = str(Path(gtfs_path) / "stops.txt")

            from geocapt_loader import load_all
            from geocapt_colgen import colgen_loop_joint
            from geocapt_gurobi import evaluate_full, adequation_report
            from geocapt_localsearch import local_search_joint
            from geocapt_route_quality import build_node_coords
        except ImportError as e:
            st.error(f"Failed to import pipeline modules: {e}")
            st.info(f"Make sure Capt-Temp is at: `{CAPT_TEMP_DIR}`")
            st.stop()

        # ----------------------------------------------------------------
        # Phase 1: Load data for all periods
        # ----------------------------------------------------------------
        progress.progress(5, text="Phase 1: Loading data for all periods...")
        status = st.status("Running joint multi-period optimisation...", expanded=True)

        ALL_BANDS = ["pot_hp", "pot_hc", "pot_soir", "pot_nuit"]
        period_weights = {
            "pot_hp": 0.40, "pot_hc": 0.30, "pot_soir": 0.20, "pot_nuit": 0.10,
        }

        all_data = {}
        G = None
        nodes = None
        prox = None

        t_start = time.time()

        for i, band in enumerate(ALL_BANDS):
            status.write(f"Loading {band}...")
            data = load_all(
                csv_path=str(POM_CSV),
                pot_col=band,
                use_new_od=params["use_new_od"],
            )
            all_data[band] = data
            if G is None:
                G = data["G"]
                nodes = data["nodes"]
                prox = data["prox"]
            status.write(f"  {data['n']} nodes, {len(data['demand_triples'])} OD pairs")

        progress.progress(20, text="Phase 1 complete. Starting optimisation...")

        sample_data = all_data[ALL_BANDS[0]]
        is_gtfs = sample_data.get("mode") == "gtfs_osm"
        route_gen_G = sample_data.get("route_gen_G")
        min_rl = cfg.GTFS_MIN_ROUTE_NODES if is_gtfs else cfg.MIN_ROUTE_NODES
        max_rl = cfg.GTFS_MAX_ROUTE_NODES if is_gtfs else cfg.MAX_ROUTE_NODES

        # ----------------------------------------------------------------
        # Phase 2-3: Joint column generation
        # ----------------------------------------------------------------
        progress.progress(25, text="Phase 2-3: Joint column generation...")
        status.write("Running column generation (this may take several minutes)...")

        eff_top_k = 0 if not params["route_choice"] else cfg.TOP_K_DIRECT
        eff_alpha_quality = 0.0 if not params["route_rules"] else cfg.ALPHA_QUALITY
        eff_f_max = cfg.F_MIN if not params["freq_opt"] else cfg.F_MAX

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
            st.error("No feasible solution found. Try adjusting parameters.")
            st.stop()

        progress.progress(70, text="Column generation done. Running local search...")

        # ----------------------------------------------------------------
        # Phase 4: Local search refinement
        # ----------------------------------------------------------------
        ls_rounds = params.get("ls_rounds", 3)
        if ls_rounds > 0:
            status.write(f"Running local search refinement ({ls_rounds} rounds)...")
            periods_ls = {
                p: {"node_pot": all_data[p]["node_pot"],
                    "node_q_min": all_data[p]["node_q_min"],
                    "node_q_max": all_data[p]["node_q_max"]}
                for p in ALL_BANDS
            }

            selected, freqs_pp = local_search_joint(
                selected, freqs_pp,
                G, nodes, periods_ls, prox, period_weights,
                alpha_adeq=params["alpha_adeq"],
                alpha_oper=params["alpha_oper"],
                max_rounds=ls_rounds,
            )
        else:
            status.write("Skipping local search.")

        # ----------------------------------------------------------------
        # Phase 4b: Final axis-sort on selected routes
        # ----------------------------------------------------------------
        # The solver + local search optimise for passenger cost and adequation,
        # not visual coherence.  Reorder each final route's stops monotonically
        # along its principal axis so the rendered polyline follows a clean
        # corridor direction without zigzag.  This preserves the SET of stops
        # on each route (so direct OD coverage is unchanged), only the order.
        status.write("Axis-sorting final routes for clean visuals...")
        from geocapt_routes import _axis_sort_route
        points_for_sort = all_data[ALL_BANDS[0]].get("points")
        nc_sort = build_node_coords(points_for_sort) if points_for_sort else {}
        if nc_sort:
            selected = [_axis_sort_route(r, nc_sort) for r in selected]

        progress.progress(85, text="Evaluating results...")

        # ----------------------------------------------------------------
        # Phase 5: Evaluation
        # ----------------------------------------------------------------
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

        # Per-node adequation status
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

        # Stop names
        stop_names = {}
        if points:
            for p in points:
                nid = p.get("id", 0)
                stop_names[nid] = p.get("stop_name", f"Stop {nid}")

        elapsed = time.time() - t_start
        progress.progress(95, text="Generating map...")

        # Per-period evaluation
        period_results = {}
        for band in ALL_BANDS:
            d = all_data[band]
            f_b = freqs_pp[band]
            m_b = evaluate_full(selected, d["G"], d["nodes"], d["demand_triples"])
            a_b = adequation_report(
                selected, f_b, d["nodes"], d["prox"],
                d["node_pot"], d["node_q_min"], d["node_q_max"])
            period_results[band] = {"metrics": m_b, "adeq": a_b, "freqs": f_b}

        # ----------------------------------------------------------------
        # Store results in session state
        # ----------------------------------------------------------------
        st.session_state["results"] = {
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

        progress.progress(100, text="Done!")
        status.update(label=f"Optimisation complete in {elapsed:.0f}s", state="complete")

    # ----------------------------------------------------------------
    # Display results (persistent across reruns)
    # ----------------------------------------------------------------
    results = st.session_state.get("results")
    if results is None:
        st.info("Click **Run Optimisation** to start.")
        st.stop()

    st.markdown("---")

    # Key metrics top bar
    m = results["peak_metrics"]
    a = results["peak_adeq"]
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("d0 (direct)", f"{m['d0']:.1f}%")
    c2.metric("d1 (1 transfer)", f"{m['d1']:.1f}%")
    c3.metric("d2+", f"{m.get('d2', 0) + m.get('d3+', 0):.1f}%")
    c4.metric("dun (unserved)", f"{m['dun']:.1f}%")
    c5.metric("ATT (min)", f"{m['ATT']:.2f}")
    c6.metric("Adequation", f"{a['pct_ok']:.1f}%")
    c7.metric("Routes", f"{results['n_routes']}")

    # Tabs for different views
    tab_map, tab_periods, tab_routes, tab_history = st.tabs([
        "Interactive Map", "Per-Period Results", "Route Details", "CG History",
    ])

    with tab_map:
        st.subheader("Interactive Route Map")
        html_str = generate_dashboard_html(
            results["selected"], results["freqs_pp"]["pot_hp"],
            results["node_coords"], results["G"],
            results["peak_metrics"], results["peak_adeq"],
            results["adeq_node_status"], results["stop_names"],
            title=f"GeoCapt Demo — {results['n_routes']} routes (pot_hp)",
        )
        st.components.v1.html(html_str, height=700, scrolling=True)

        # Download button for the HTML
        st.download_button(
            "Download HTML Map",
            html_str,
            file_name="geocapt_demo_result.html",
            mime="text/html",
        )

    with tab_periods:
        st.subheader("Per-Period Evaluation")

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
                "Period": band,
                "d0 %": round(pm["d0"], 1),
                "d1 %": round(pm["d1"], 1),
                "d2+ %": round(pm.get("d2", 0) + pm.get("d3+", 0), 1),
                "dun %": round(pm["dun"], 2),
                "ATT (min)": round(pm["ATT"], 2),
                "Adeq %": round(pa["pct_ok"], 1),
                "Over": pa["n_over"],
                "Under": pa["n_under"],
                "Avg Freq": round(avg_f, 2),
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Frequency table
        st.subheader("Per-Period Frequency Table")
        freq_rows = []
        for idx, route in enumerate(results["selected"]):
            row = {"Route": f"R{idx+1}", "Stops": len(route)}
            for band in ["pot_hp", "pot_hc", "pot_soir", "pot_nuit"]:
                row[band] = round(results["freqs_pp"][band][idx], 2)
            freq_rows.append(row)
        st.dataframe(pd.DataFrame(freq_rows), use_container_width=True, hide_index=True)

    with tab_routes:
        st.subheader("Selected Routes")
        for idx, route in enumerate(results["selected"]):
            freq = results["freqs_pp"]["pot_hp"][idx]
            color = _ROUTE_COLOURS[idx % len(_ROUTE_COLOURS)]
            names = [results["stop_names"].get(n, f"#{n}") for n in route]
            with st.expander(
                f"Route {idx+1} — {len(route)} stops, f={freq:.1f}/h",
                expanded=False,
            ):
                st.markdown(f"**Itinerary:** {' -> '.join(names)}")
                st.markdown(f"**Node IDs:** {' - '.join(map(str, route))}")

        # Coverage stats
        covered = set()
        for r in results["selected"]:
            covered.update(r)
        total_nodes = len(results["node_coords"])
        st.metric("Node Coverage", f"{len(covered)}/{total_nodes} ({100*len(covered)/total_nodes:.0f}%)")

    with tab_history:
        st.subheader("Column Generation History")
        hist = results.get("history", [])
        if hist:
            hist_rows = []
            for h in hist:
                impr = f"{h['improvement']:.2%}" if h["iteration"] > 1 else "---"
                hist_rows.append({
                    "Iteration": h["iteration"],
                    "Pool Size": h["pool_size"],
                    "Objective": round(h["obj_value"], 2),
                    "Improvement": impr,
                    "Time (s)": round(h["time"], 0),
                })
            st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No history available.")

        st.metric("Total Time", f"{results['elapsed']:.0f}s")

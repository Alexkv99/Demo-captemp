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
# SIDEBAR
# ============================================================================

st.sidebar.image(str(DEMO_DIR / "Transdev_logo_2018.png"), width=180)
st.sidebar.title("GeoCapt-UTRP Demo")
st.sidebar.markdown("---")

# Step indicator
step = st.sidebar.radio(
    "Navigation",
    ["1 - Données GTFS", "2 - Cadre POT", "3 - Priorités", "4 - Exécution & Résultats"],
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

        # ----------------------------------------------------------------
        # Import Capt-Temp pipeline modules
        # ----------------------------------------------------------------
        progress = st.progress(0, text="Importation des modules du pipeline...")

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
            st.error(f"Échec de l'importation des modules du pipeline : {e}")
            st.info(f"Assurez-vous que Capt-Temp se trouve à : `{CAPT_TEMP_DIR}`")
            st.stop()

        # ----------------------------------------------------------------
        # Phase 1: Load data for all periods
        # ----------------------------------------------------------------
        progress.progress(5, text="Phase 1 : Chargement des données pour toutes les périodes...")
        status = st.status("Exécution de l'optimisation conjointe multi-période...", expanded=True)

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
            status.write(f"Chargement de {band}...")
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
            status.write(f"  {data['n']} nœuds, {len(data['demand_triples'])} paires OD")

        progress.progress(20, text="Phase 1 terminée. Démarrage de l'optimisation...")

        sample_data = all_data[ALL_BANDS[0]]
        is_gtfs = sample_data.get("mode") == "gtfs_osm"
        route_gen_G = sample_data.get("route_gen_G")
        min_rl = cfg.GTFS_MIN_ROUTE_NODES if is_gtfs else cfg.MIN_ROUTE_NODES
        max_rl = cfg.GTFS_MAX_ROUTE_NODES if is_gtfs else cfg.MAX_ROUTE_NODES

        # ----------------------------------------------------------------
        # Phase 2-3: Joint column generation
        # ----------------------------------------------------------------
        progress.progress(25, text="Phase 2-3 : Génération conjointe de colonnes...")
        status.write("Exécution de la génération de colonnes (peut prendre plusieurs minutes)...")

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
            st.error("Aucune solution réalisable trouvée. Essayez d'ajuster les paramètres.")
            st.stop()

        progress.progress(70, text="Génération de colonnes terminée. Exécution de la recherche locale...")

        # ----------------------------------------------------------------
        # Phase 4: Local search refinement
        # ----------------------------------------------------------------
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

        # ----------------------------------------------------------------
        # Phase 4b: Final axis-sort on selected routes
        # ----------------------------------------------------------------
        # The solver + local search optimise for passenger cost and adequation,
        # not visual coherence.  Reorder each final route's stops monotonically
        # along its principal axis so the rendered polyline follows a clean
        # corridor direction without zigzag.  This preserves the SET of stops
        # on each route (so direct OD coverage is unchanged), only the order.
        status.write("Tri par axe des lignes finales pour une meilleure lisibilité...")
        from geocapt_routes import _axis_sort_route
        points_for_sort = all_data[ALL_BANDS[0]].get("points")
        nc_sort = build_node_coords(points_for_sort) if points_for_sort else {}
        if nc_sort:
            selected = [_axis_sort_route(r, nc_sort) for r in selected]

        progress.progress(85, text="Évaluation des résultats...")

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
        progress.progress(95, text="Génération de la carte...")

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

        progress.progress(100, text="Terminé !")
        status.update(label=f"Optimisation terminée en {elapsed:.0f} s", state="complete")

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

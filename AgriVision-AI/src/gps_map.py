"""
AgriVision AI - GPS & Real-Time Map Module
===========================================
Provides:
  - Live GPS location detection (browser-based)
  - Nearest agricultural district finder
  - Folium interactive field map with crop overlays
  - GPS-to-prediction pipeline (auto-fills predict form)
  - Field boundary drawing & area calculation
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import (
    LocateControl, MeasureControl, HeatMap,
    MarkerCluster, Fullscreen, MiniMap
)
from geopy.distance import geodesic
import os
from typing import Optional


# ── Indian state centroids (lat, lon) ─────────────────────────────
STATE_CENTROIDS = {
    "Andhra Pradesh":        (15.9129,  79.7400),
    "Arunachal Pradesh":     (28.2180,  94.7278),
    "Assam":                 (26.2006,  92.9376),
    "Bihar":                 (25.0961,  85.3131),
    "Chhattisgarh":          (21.2787,  81.8661),
    "Goa":                   (15.2993,  74.1240),
    "Gujarat":               (22.2587,  71.1924),
    "Haryana":               (29.0588,  76.0856),
    "Himachal Pradesh":      (31.1048,  77.1734),
    "Jharkhand":             (23.6102,  85.2799),
    "Karnataka":             (15.3173,  75.7139),
    "Kerala":                (10.8505,  76.2711),
    "Madhya Pradesh":        (22.9734,  78.6569),
    "Maharashtra":           (19.7515,  75.7139),
    "Manipur":               (24.6637,  93.9063),
    "Meghalaya":             (25.4670,  91.3662),
    "Mizoram":               (23.1645,  92.9376),
    "Nagaland":              (26.1584,  94.5624),
    "Odisha":                (20.9517,  85.0985),
    "Punjab":                (31.1471,  75.3412),
    "Rajasthan":             (27.0238,  74.2179),
    "Sikkim":                (27.5330,  88.5122),
    "Tamil Nadu":            (11.1271,  78.6569),
    "Telangana":             (18.1124,  79.0193),
    "Tripura":               (23.9408,  91.9882),
    "Uttar Pradesh":         (26.8467,  80.9462),
    "Uttarakhand":           (30.0668,  79.0193),
    "West Bengal":           (22.9868,  87.8550),
    "Andaman and Nicobar Islands": (11.7401, 92.6586),
    "Chandigarh":            (30.7333,  76.7794),
    "Dadra and Nagar Haveli": (20.1809, 73.0169),
    "Daman and Diu":         (20.3974,  72.8328),
    "Delhi":                 (28.7041,  77.1025),
    "Jammu and Kashmir":     (33.7782,  76.5762),
    "Lakshadweep":           (10.5667,  72.6417),
    "Puducherry":            (11.9416,  79.8083),
}


class GPSMapEngine:
    """
    Handles all GPS, location, and interactive map operations
    for the AgriVision AI platform.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._state_stats = self._precompute_state_stats()

    # ── Pre-compute state-level stats for map overlays ────────────
    def _precompute_state_stats(self) -> pd.DataFrame:
        agg = self.df.groupby("State_Name").agg(
            Total_Production=("Production", "sum"),
            Total_Area=("Area", "sum"),
            Avg_Yield=("Yield", "mean"),
            Top_Crop=("Crop", lambda x: x.value_counts().idxmax()),
            Num_Crops=("Crop", "nunique"),
            Num_Districts=("District_Name", "nunique"),
        ).reset_index()

        # Add coordinates
        agg["Lat"] = agg["State_Name"].map(lambda s: STATE_CENTROIDS.get(s, (20.5937, 78.9629))[0])
        agg["Lon"] = agg["State_Name"].map(lambda s: STATE_CENTROIDS.get(s, (20.5937, 78.9629))[1])
        return agg

    # ── Find nearest state to GPS coordinates ────────────────────
    def nearest_state(self, lat: float, lon: float) -> dict:
        """Return the closest Indian state to given GPS coordinates."""
        user_loc = (lat, lon)
        best_state, best_dist = None, float("inf")

        for state, (s_lat, s_lon) in STATE_CENTROIDS.items():
            dist = geodesic(user_loc, (s_lat, s_lon)).km
            if dist < best_dist:
                best_dist = dist
                best_state = state

        # Get state stats
        stats_row = self._state_stats[self._state_stats["State_Name"] == best_state]
        top_crop = stats_row["Top_Crop"].values[0] if len(stats_row) else "N/A"
        total_prod = int(stats_row["Total_Production"].values[0]) if len(stats_row) else 0

        return {
            "state": best_state,
            "distance_km": round(best_dist, 1),
            "lat": STATE_CENTROIDS[best_state][0],
            "lon": STATE_CENTROIDS[best_state][1],
            "top_crop": top_crop,
            "total_production": total_prod,
        }

    # ── Main India production choropleth map ─────────────────────
    def build_india_production_map(
        self,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None,
        selected_crop: str = "All",
        selected_season: str = "All",
    ) -> folium.Map:
        """
        Build a full-featured Folium map:
        - State circle markers sized by production
        - Heatmap layer of production density
        - User GPS pin if coordinates provided
        - Crop/season filtering
        - Click-to-predict markers
        """
        # Filter data
        df_f = self.df.copy()
        if selected_crop != "All":
            df_f = df_f[df_f["Crop"] == selected_crop]
        if selected_season != "All":
            df_f = df_f[df_f["Season"] == selected_season]

        stats = df_f.groupby("State_Name").agg(
            Total_Production=("Production", "sum"),
            Avg_Yield=("Yield", "mean"),
            Top_Crop=("Crop", lambda x: x.value_counts().idxmax() if len(x) else "N/A"),
        ).reset_index()

        # Base map centred on India
        m = folium.Map(
            location=[22.5, 82.0],
            zoom_start=5,
            tiles=None,
        )

        # ── Tile layers ───────────────────────────────────────────
        folium.TileLayer(
            "CartoDB dark_matter", name="🌑 Dark (default)", control=True
        ).add_to(m)
        folium.TileLayer(
            "OpenStreetMap", name="🗺️ Street Map", control=True
        ).add_to(m)
        folium.TileLayer(
            "CartoDB positron", name="☁️ Light", control=True
        ).add_to(m)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="🛰️ Satellite",
            control=True,
        ).add_to(m)

        # ── Plugins ───────────────────────────────────────────────
        Fullscreen(position="topright").add_to(m)
        MeasureControl(
            position="bottomleft",
            primary_length_unit="kilometers",
            secondary_length_unit="miles",
            primary_area_unit="sqmeters",
        ).add_to(m)
        MiniMap(toggle_display=True, position="bottomright").add_to(m)
        LocateControl(
            position="topright",
            strings={"title": "📍 Find my location"},
            flyTo=True,
        ).add_to(m)

        # ── Production scale for circle radii ────────────────────
        max_prod = stats["Total_Production"].max() if len(stats) else 1

        # ── State markers ─────────────────────────────────────────
        marker_cluster = MarkerCluster(
            name="📍 District markers", show=False
        ).add_to(m)

        state_layer = folium.FeatureGroup(name="🌾 State Production Circles", show=True)

        for _, row in stats.iterrows():
            state_name = row["State_Name"]
            if state_name not in STATE_CENTROIDS:
                continue
            lat, lon = STATE_CENTROIDS[state_name]
            prod      = row["Total_Production"]
            yield_avg = round(row["Avg_Yield"], 2)
            top_crop  = row["Top_Crop"]

            # Circle size: 5k – 80k metres radius scaled by production
            radius = 5000 + (prod / max_prod) * 75000

            # Colour: green scale by production quartile
            pct = prod / max_prod
            if pct > 0.75:
                color = "#00c853"
            elif pct > 0.5:
                color = "#64dd17"
            elif pct > 0.25:
                color = "#ffd600"
            else:
                color = "#ff6d00"

            popup_html = f"""
            <div style="font-family:Arial;min-width:200px;padding:6px">
              <h4 style="margin:0 0 6px;color:#2e7d32">🌾 {state_name}</h4>
              <table style="font-size:13px;width:100%">
                <tr><td><b>Total Production</b></td><td>{int(prod):,} tons</td></tr>
                <tr><td><b>Avg Yield</b></td><td>{yield_avg} t/unit</td></tr>
                <tr><td><b>Top Crop</b></td><td>{top_crop}</td></tr>
              </table>
            </div>
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=20 + (prod / max_prod) * 35,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.55,
                weight=2,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{state_name}: {int(prod):,} tons | Top: {top_crop}",
            ).add_to(state_layer)

        state_layer.add_to(m)

        # ── Heatmap layer ─────────────────────────────────────────
        heat_data = []
        for _, row in stats.iterrows():
            if row["State_Name"] in STATE_CENTROIDS:
                lat, lon = STATE_CENTROIDS[row["State_Name"]]
                weight = float(row["Total_Production"]) / max_prod
                heat_data.append([lat, lon, weight])

        HeatMap(
            heat_data,
            name="🔥 Production Heatmap",
            min_opacity=0.3,
            max_zoom=18,
            radius=40,
            blur=25,
            gradient={0.2: "blue", 0.5: "yellow", 0.8: "orange", 1.0: "red"},
        ).add_to(m)

        # ── District-level markers (clustered) ────────────────────
        dist_stats = df_f.groupby(["State_Name", "District_Name"]).agg(
            Production=("Production", "sum"),
            Top_Crop=("Crop", lambda x: x.value_counts().idxmax() if len(x) else "N/A"),
        ).reset_index().nlargest(150, "Production")

        for _, drow in dist_stats.iterrows():
            sname = drow["State_Name"]
            if sname not in STATE_CENTROIDS:
                continue
            base_lat, base_lon = STATE_CENTROIDS[sname]
            # Jitter around state centroid so districts don't all overlap
            jlat = base_lat + np.random.uniform(-1.5, 1.5)
            jlon = base_lon + np.random.uniform(-1.5, 1.5)

            folium.Marker(
                location=[jlat, jlon],
                icon=folium.Icon(color="green", icon="leaf", prefix="fa"),
                tooltip=f"{drow['District_Name']}, {sname}\n{int(drow['Production']):,} tons",
                popup=folium.Popup(
                    f"<b>{drow['District_Name']}</b><br>{sname}<br>"
                    f"Production: {int(drow['Production']):,} tons<br>"
                    f"Top Crop: {drow['Top_Crop']}",
                    max_width=200,
                ),
            ).add_to(marker_cluster)

        # ── User GPS pin ──────────────────────────────────────────
        if user_lat is not None and user_lon is not None:
            nearest = self.nearest_state(user_lat, user_lon)

            # Pulsing user location marker
            folium.Marker(
                location=[user_lat, user_lon],
                icon=folium.Icon(color="red", icon="map-marker", prefix="fa"),
                popup=folium.Popup(
                    f"<b>📍 Your Location</b><br>"
                    f"Lat: {round(user_lat, 4)}, Lon: {round(user_lon, 4)}<br>"
                    f"Nearest state: <b>{nearest['state']}</b> ({nearest['distance_km']} km)<br>"
                    f"Top crop there: <b>{nearest['top_crop']}</b>",
                    max_width=250,
                ),
                tooltip=f"📍 You are here | Nearest: {nearest['state']}",
            ).add_to(m)

            # Line to nearest state
            folium.PolyLine(
                locations=[[user_lat, user_lon], [nearest["lat"], nearest["lon"]]],
                color="#ff5252",
                weight=2,
                dash_array="8 4",
                tooltip=f"Distance to {nearest['state']}: {nearest['distance_km']} km",
            ).add_to(m)

            # Zoom to user
            m.location = [user_lat, user_lon]
            m.zoom_start = 7

        # ── Layer control ─────────────────────────────────────────
        folium.LayerControl(position="topright", collapsed=False).add_to(m)

        return m

    # ── Field area map: draw and measure a field ──────────────────
    def build_field_mapper(
        self,
        center_lat: float = 20.5937,
        center_lon: float = 78.9629,
    ) -> folium.Map:
        """
        Interactive field boundary mapper.
        User can draw polygons → area is calculated automatically.
        """
        from folium.plugins import Draw

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri Satellite",
        )

        # Drawing tools
        Draw(
            export=True,
            filename="field_boundary.geojson",
            draw_options={
                "polygon":   {"allowIntersection": False, "showArea": True},
                "rectangle": {"showArea": True},
                "circle":    {"showRadius": True},
                "polyline":  False,
                "circlemarker": False,
                "marker":    True,
            },
            edit_options={"edit": True, "remove": True},
        ).add_to(m)

        MeasureControl(
            position="bottomleft",
            primary_length_unit="meters",
            primary_area_unit="sqmeters",
        ).add_to(m)

        Fullscreen().add_to(m)
        LocateControl(flyTo=True).add_to(m)

        # Instructions overlay
        instructions = """
        <div style="
            position:fixed; bottom:30px; left:50%; transform:translateX(-50%);
            background:rgba(0,0,0,0.75); color:white; padding:10px 18px;
            border-radius:20px; font-size:13px; z-index:9999;
            border:1px solid #52b788; pointer-events:none;">
            ✏️ Draw your field boundary → area calculated automatically
        </div>
        """
        m.get_root().html.add_child(folium.Element(instructions))

        return m

    # ── Save map to HTML ──────────────────────────────────────────
    def save_map(self, fmap: folium.Map, path: str) -> str:
        """Save a Folium map to an HTML file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        fmap.save(path)
        return path

    # ── Crop recommendation by GPS location ──────────────────────
    def recommend_crops_for_location(
        self, lat: float, lon: float, season: str = "All", top_n: int = 5
    ) -> pd.DataFrame:
        """
        Suggest top N crops for the detected state based on historical yield.
        """
        nearest = self.nearest_state(lat, lon)
        state   = nearest["state"]

        df_s = self.df[self.df["State_Name"] == state]
        if season != "All":
            df_s = df_s[df_s["Season"] == season]

        if df_s.empty:
            return pd.DataFrame()

        recs = (
            df_s.groupby("Crop")
            .agg(
                Avg_Yield=("Yield", "mean"),
                Total_Production=("Production", "sum"),
                Seasons=("Season", lambda x: ", ".join(sorted(x.unique()))),
            )
            .sort_values("Avg_Yield", ascending=False)
            .head(top_n)
            .reset_index()
        )
        recs["State"]        = state
        recs["Distance_km"]  = nearest["distance_km"]
        recs["Avg_Yield"]    = recs["Avg_Yield"].round(2)
        return recs[["Crop", "Avg_Yield", "Total_Production", "Seasons", "State", "Distance_km"]]

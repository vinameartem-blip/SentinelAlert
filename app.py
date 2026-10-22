"""
app.py — Веб-интерфейс SentinelAlert для космического мониторинга угодий.
"""

from datetime import datetime, timezone, timedelta
import base64
import streamlit as st
import folium
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium

from core.stac_loader import get_sentinel_item
from core.ndvi_calc import (
    compute_health_index,
    analyze_field_metrics,
    render_heatmap,
    extract_true_color_rgb,
    get_bounding_box_4326,
)

st.set_page_config(
    page_title="SentinelAlert — Космический мониторинг угодий",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    .main-sub {
        background-color: #1e293b;
        color: #93c5fd;
        padding: 0.4rem 0.9rem;
        border-radius: 6px;
        display: inline-block;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 1.2rem;
        border: 1px solid #3b82f6;
    }
    .stAlert {
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_LAT = 43.7150
DEFAULT_LON = 39.6550

if "lat" not in st.session_state:
    st.session_state.lat = DEFAULT_LAT
if "lon" not in st.session_state:
    st.session_state.lon = DEFAULT_LON
if "last_click" not in st.session_state:
    st.session_state.last_click = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

def run_pipeline(lat: float, lon: float, max_cloud: int, box_size: int, end_date=None):
    item = get_sentinel_item(lat=lat, lon=lon, max_cloud=max_cloud, days_back=14, end_date=end_date)
    health = compute_health_index(item["urls"], lat, lon, box_size_meters=box_size)
    metrics = analyze_field_metrics(health)
    heatmap_buf = render_heatmap(health)

    rgb_bytes = None
    if "visual" in item["urls"]:
        try:
            rgb_buf = extract_true_color_rgb(item["urls"]["visual"], lat, lon, box_size_meters=box_size)
            rgb_bytes = rgb_buf.getvalue()
        except Exception:
            rgb_bytes = None

    return {
        "meta": item["meta"],
        "metrics": metrics,
        "image_bytes": heatmap_buf.getvalue(),
        "rgb_bytes": rgb_bytes,
    }

st.sidebar.header("Параметры мониторинга")

lat_val = st.sidebar.number_input(
    "Широта (Latitude, °N)",
    min_value=-85.0,
    max_value=85.0,
    value=float(st.session_state.lat),
    step=0.001,
    format="%.4f",
)

lon_val = st.sidebar.number_input(
    "Долгота (Longitude, °E)",
    min_value=-180.0,
    max_value=180.0,
    value=float(st.session_state.lon),
    step=0.001,
    format="%.4f",
)

if lat_val != st.session_state.lat or lon_val != st.session_state.lon:
    st.session_state.lat = lat_val
    st.session_state.lon = lon_val
    st.session_state.analysis_results = None

selected_date = st.sidebar.date_input(
    "Дата актуальности",
    value=datetime.now().date(),
    max_value=datetime.now().date(),
)

cloud_val = st.sidebar.slider(
    "Допустимая облачность (%)",
    min_value=5,
    max_value=40,
    value=20,
    step=5,
)

box_val = st.sidebar.slider(
    "Охват поля (м)",
    min_value=500,
    max_value=3000,
    value=1500,
    step=100,
)

run_clicked = st.sidebar.button(
    "Запустить космический анализ",
    type="primary",
)

st.markdown('<div class="main-title">SentinelAlert — Космический мониторинг угодий</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-sub">Высокоточный ансамбль EVI + MSAVI2 с фильтрацией облаков и теней SCL (разрешение 10×10 м)</div>',
    unsafe_allow_html=True,
)

if run_clicked:
    with st.spinner("Поиск свежих снимков Sentinel-2 и расчет индексов..."):
        try:
            st.session_state.analysis_results = run_pipeline(
                lat=st.session_state.lat,
                lon=st.session_state.lon,
                max_cloud=cloud_val,
                box_size=box_val,
                end_date=selected_date,
            )
        except Exception as e:
            st.error(f"Ошибка при анализе данных: {str(e)}")
            st.session_state.analysis_results = None

col1, col2 = st.columns(2)

min_lon, min_lat, max_lon, max_lat = get_bounding_box_4326(
    st.session_state.lat, st.session_state.lon, box_val
)
bounds = [[min_lat, min_lon], [max_lat, max_lon]]

with col1:
    st.subheader("Интерактивная карта")

    m = folium.Map(
        location=[st.session_state.lat, st.session_state.lon],
        zoom_start=13,
        tiles=None,
    )

    gibs_date_str = selected_date.strftime("%Y-%m-%d")
    gibs_tile_url = f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_SNPP_CorrectedReflectance_TrueColor/default/{gibs_date_str}/GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.jpg"
    folium.TileLayer(
        tiles=gibs_tile_url,
        name=f"Спутниковая мозаика ({selected_date.strftime('%d.%m.%Y')})",
        attr="NASA GIBS / EOSDIS",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        name="Физический рельеф (OpenTopoMap)",
        attr="OpenTopoMap",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        name="Схема дорог (OpenStreetMap)",
        attr='&copy; OpenStreetMap',
        overlay=False,
        control=True,
    ).add_to(m)

    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        meta = res["meta"]

        if meta.get("tile_url"):
            folium.TileLayer(
                tiles=meta["tile_url"],
                name=f"Спутник Sentinel-2 ({meta['date']})",
                attr="Copernicus Sentinel-2",
                overlay=True,
                control=True,
                opacity=0.95,
            ).add_to(m)

        if res.get("rgb_bytes"):
            b64_rgb = base64.b64encode(res["rgb_bytes"]).decode("utf-8")
            ImageOverlay(
                image=f"data:image/png;base64,{b64_rgb}",
                bounds=bounds,
                opacity=0.9,
                name="Реальный снимок поля Sentinel-2 (RGB)",
                zindex=2,
            ).add_to(m)

        if res.get("image_bytes"):
            b64_heat = base64.b64encode(res["image_bytes"]).decode("utf-8")
            ImageOverlay(
                image=f"data:image/png;base64,{b64_heat}",
                bounds=bounds,
                opacity=0.8,
                name="Тепловая карта поля (EVI+MSAVI2)",
                zindex=3,
            ).add_to(m)

    folium.Marker(
        [st.session_state.lat, st.session_state.lon],
        tooltip="Координаты поля",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    folium.Rectangle(
        bounds=bounds,
        color="#2563eb",
        weight=2,
        fill=False,
    ).add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    map_data = st_folium(
        m,
        height=480,
        width=None,
        returned_objects=["last_clicked"],
        key="map",
    )

    if map_data and map_data.get("last_clicked"):
        c_lat = round(map_data["last_clicked"]["lat"], 4)
        c_lon = round(map_data["last_clicked"]["lng"], 4)
        if st.session_state.last_click != (c_lat, c_lon):
            st.session_state.last_click = (c_lat, c_lon)
            st.session_state.lat = c_lat
            st.session_state.lon = c_lon
            st.session_state.analysis_results = None
            st.rerun()

    st.caption(f"Координаты поля: {st.session_state.lat:.4f}° N, {st.session_state.lon:.4f}° E (кликните по карте для изменения)")

with col2:
    st.subheader("Тепловая карта здоровья биомассы")
    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        meta = res["meta"]

        caption_text = f"Снимок от {meta['date']} ({meta['days_ago']} дн. назад), облачность {meta['cloud_pct']}% | ID сцены: {meta['scene_id']}"

        if res.get("rgb_bytes"):
            tab_heat, tab_rgb = st.tabs(["Тепловая карта", "Оптический снимок (RGB)"])
            with tab_heat:
                st.image(res["image_bytes"], caption=caption_text, width="stretch")
            with tab_rgb:
                st.image(res["rgb_bytes"], caption=caption_text, width="stretch")
        else:
            st.image(res["image_bytes"], caption=caption_text, width="stretch")
    else:
        st.info("Выберите точку на карте и нажмите кнопку «Запустить космический анализ» в боковом меню.")

if st.session_state.analysis_results:
    res = st.session_state.analysis_results
    meta = res["meta"]
    metrics = res["metrics"]

    st.markdown("---")
    st.subheader("Агрономические показатели участка")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Средний индекс здоровья",
            value=f"{metrics['mean_health']:.3f}",
            help="Ансамбль 0.6*EVI + 0.4*MSAVI2",
        )
    with m2:
        st.metric(
            label="Здоровая площадь",
            value=f"{metrics['healthy_pct']:.1f}%",
            delta=f"{metrics['healthy_ha']:.1f} га",
            delta_color="normal",
            help="Индекс >= 0.45",
        )
    with m3:
        st.metric(
            label="Зона риска",
            value=f"{metrics['risk_ha']:.1f} га",
            delta=f"{metrics['risk_pct']:.1f}%",
            delta_color="off",
            help="Индекс 0.20 - 0.45",
        )
    with m4:
        st.metric(
            label="Критический стресс",
            value=f"{metrics['stress_ha']:.1f} га",
            delta=f"{metrics['stress_pct']:.1f}%",
            delta_color="inverse",
            help="Индекс < 0.20",
        )

    st.markdown("#### Рекомендация агронома")
    if metrics["stress_pct"] > 35.0 or metrics["mean_health"] < 0.25:
        st.error(
            f"Критическое состояние поля. Доля угнетенной биомассы составляет "
            f"{metrics['stress_pct']}% ({metrics['stress_ha']} га). "
            f"Рекомендуется экстренный выезд агронома: проверка на корневые гнили, дефицит питания или засуху."
        )
    elif metrics["risk_pct"] > 30.0 or metrics["mean_health"] < 0.42:
        st.warning(
            f"Развитие в норме. Зона риска охватывает "
            f"{metrics['risk_pct']}% угодий ({metrics['risk_ha']} га), "
            f"критических отклонений не выявлено, но требуется увеличить контроль за состоянием поля."
        )
    else:
        st.success(
            f"Оптимальное состояние посевов. Доля здоровой биомассы составляет "
            f"{metrics['healthy_pct']}% ({metrics['healthy_ha']} га), средний индекс поля {metrics['mean_health']:.3f}. "
            f"Вегетация протекает без существенных аномалий."
        )

    report_text = f"""# Агрономический отчет SentinelAlert
Дата формирования: {datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M:%S UTC")}

### Метаданные космической съемки
- Координаты центра: {st.session_state.lat:.4f}° N, {st.session_state.lon:.4f}° E
- Охват поля: {box_val} x {box_val} м ({metrics['total_ha']} га)
- ID сцены: {meta['scene_id']}
- Дата пролета спутника: {meta['date']} ({meta['days_ago']} дн. назад)
- Облачность сцены: {meta['cloud_pct']}%
- Отфильтровано помех SCL (облака/тени/вода): {metrics['cloud_mask_pct']}%

### Методология
- Каналы Sentinel-2 L2A: B02 (Blue), B04 (Red), B08 (NIR)
- Фильтрация помех: SCL (классы 4 и 5)
- Индекс здоровья: 0.6 * EVI + 0.4 * MSAVI2

### Площади зон
- Здоровая биомасса (>= 0.45): {metrics['healthy_ha']} га ({metrics['healthy_pct']}%)
- Зона риска (0.20 - 0.45): {metrics['risk_ha']} га ({metrics['risk_pct']}%)
- Критический стресс (< 0.20): {metrics['stress_ha']} га ({metrics['stress_pct']}%)
- Валидная площадь: {metrics['valid_ha']} га
- Средний индекс здоровья поля: {metrics['mean_health']:.3f}

### Заключение
{"Критический стресс: необходим полевой осмотр." if (metrics["stress_pct"] > 35.0 or metrics["mean_health"] < 0.25) else ("Умеренный риск: рекомендуется контроль влаги и подкормка." if (metrics["risk_pct"] > 30.0 or metrics["mean_health"] < 0.42) else "Норма: оптимальное развитие посевов.")}
"""

    st.download_button(
        label="Скачать агро-отчет (.md)",
        data=report_text,
        file_name=f"sentinel_report_{st.session_state.lat:.4f}_{st.session_state.lon:.4f}.md",
        mime="text/markdown",
    )

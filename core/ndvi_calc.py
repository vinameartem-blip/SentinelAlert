"""
core/ndvi_calc.py — Высокоточный расчет индексов здоровья растительности
(EVI + MSAVI2), фильтрация облаков/теней через SCL и рендеринг тепловой карты.
"""

import io
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

def get_bounding_box_4326(center_lat: float, center_lon: float, box_size_meters: float = 1500.0):
    """
    Вычисляет Bounding Box в EPSG:4326 (min_lon, min_lat, max_lon, max_lat)
    вокруг заданной точки.
    """
    half_size = box_size_meters / 2.0

    meters_per_deg_lat = 111320.0
    delta_lat = half_size / meters_per_deg_lat

    cos_lat = math.cos(math.radians(center_lat))
    if abs(cos_lat) < 1e-6:
        cos_lat = 1e-6
    meters_per_deg_lon = meters_per_deg_lat * abs(cos_lat)
    delta_lon = half_size / meters_per_deg_lon

    min_lon = center_lon - delta_lon
    max_lon = center_lon + delta_lon
    min_lat = center_lat - delta_lat
    max_lat = center_lat + delta_lat

    return min_lon, min_lat, max_lon, max_lat

def compute_health_index(
    urls: dict,
    center_lat: float,
    center_lon: float,
    box_size_meters: float = 1500.0,
) -> np.ndarray:
    """
    Читает COG-окна каналов B02, B04, B08, SCL через rasterio,
    применяет маску SCL и рассчитывает ансамблевый индекс здоровья
    Health_Index = 0.6 * EVI + 0.4 * MSAVI2.

    Возвращает 2D float32 матрицу с np.nan на отбракованных пикселях.
    """
    min_lon, min_lat, max_lon, max_lat = get_bounding_box_4326(
        center_lat, center_lon, box_size_meters
    )

    with rasterio.open(urls["B04"]) as src_b04:

        left, bottom, right, top = transform_bounds(
            "EPSG:4326", src_b04.crs, min_lon, min_lat, max_lon, max_lat
        )
        window_b04 = from_bounds(left, bottom, right, top, transform=src_b04.transform)
        b04_raw = src_b04.read(1, window=window_b04)
        b04 = b04_raw.astype(np.float32) / 10000.0
        target_shape = b04.shape

    with rasterio.open(urls["B02"]) as src_b02:
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", src_b02.crs, min_lon, min_lat, max_lon, max_lat
        )
        window = from_bounds(left, bottom, right, top, transform=src_b02.transform)
        b02 = src_b02.read(1, window=window, out_shape=target_shape).astype(np.float32) / 10000.0

    with rasterio.open(urls["B08"]) as src_b08:
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", src_b08.crs, min_lon, min_lat, max_lon, max_lat
        )
        window = from_bounds(left, bottom, right, top, transform=src_b08.transform)
        b08 = src_b08.read(1, window=window, out_shape=target_shape).astype(np.float32) / 10000.0

    with rasterio.open(urls["SCL"]) as src_scl:
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", src_scl.crs, min_lon, min_lat, max_lon, max_lat
        )
        window_scl = from_bounds(left, bottom, right, top, transform=src_scl.transform)
        scl = src_scl.read(
            1,
            window=window_scl,
            out_shape=target_shape,
            resampling=Resampling.nearest,
        )

    valid_mask = (scl == 4) | (scl == 5)

    term1 = 2.0 * b08 + 1.0
    sqrt_arg = np.maximum(0.0, (term1 ** 2) - 8.0 * (b08 - b04))
    msavi2 = (term1 - np.sqrt(sqrt_arg)) / 2.0

    denom = b08 + 6.0 * b04 - 7.5 * b02 + 1.0
    safe_denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    evi = 2.5 * ((b08 - b04) / safe_denom)

    health_index = 0.6 * evi + 0.4 * msavi2
    health_index = np.clip(health_index, -1.0, 1.0)

    health_index[~valid_mask] = np.nan

    return health_index

def analyze_field_metrics(health_matrix: np.ndarray) -> dict:
    """
    Рассчитывает агрономические метрики поля по матрице health_index:
      - Средний индекс здоровья
      - Площади (га) и доли (%) по 3 зонам:
        * Стресс / деградация: < 0.20
        * Риск / дефицит влаги: [0.20, 0.45)
        * Здоровая биомасса: >= 0.45
      - Процент отфильтрованных помех (облака/тени/вода)
      - Общая площадь
    """
    total_pixels = health_matrix.size
    if total_pixels == 0:
        return {
            "mean_health": 0.0,
            "total_ha": 0.0,
            "valid_ha": 0.0,
            "stress_pct": 0.0,
            "stress_ha": 0.0,
            "risk_pct": 0.0,
            "risk_ha": 0.0,
            "healthy_pct": 0.0,
            "healthy_ha": 0.0,
            "cloud_mask_pct": 0.0,
        }

    PIXEL_AREA_HA = 0.01

    nan_mask = np.isnan(health_matrix)
    nan_count = int(np.sum(nan_mask))
    valid_count = total_pixels - nan_count

    cloud_mask_pct = (nan_count / total_pixels) * 100.0

    if valid_count == 0:
        return {
            "mean_health": float("nan"),
            "total_ha": round(total_pixels * PIXEL_AREA_HA, 2),
            "valid_ha": 0.0,
            "stress_pct": 0.0,
            "stress_ha": 0.0,
            "risk_pct": 0.0,
            "risk_ha": 0.0,
            "healthy_pct": 0.0,
            "healthy_ha": 0.0,
            "cloud_mask_pct": round(cloud_mask_pct, 1),
        }

    valid_vals = health_matrix[~nan_mask]
    mean_health = float(np.mean(valid_vals))

    stress_count = int(np.sum(valid_vals < 0.20))
    risk_count = int(np.sum((valid_vals >= 0.20) & (valid_vals < 0.45)))
    healthy_count = int(np.sum(valid_vals >= 0.45))

    stress_pct = (stress_count / valid_count) * 100.0
    risk_pct = (risk_count / valid_count) * 100.0
    healthy_pct = (healthy_count / valid_count) * 100.0

    stress_ha = stress_count * PIXEL_AREA_HA
    risk_ha = risk_count * PIXEL_AREA_HA
    healthy_ha = healthy_count * PIXEL_AREA_HA
    valid_ha = valid_count * PIXEL_AREA_HA
    total_ha = total_pixels * PIXEL_AREA_HA

    return {
        "mean_health": round(mean_health, 3),
        "total_ha": round(total_ha, 2),
        "valid_ha": round(valid_ha, 2),
        "stress_pct": round(stress_pct, 1),
        "stress_ha": round(stress_ha, 2),
        "risk_pct": round(risk_pct, 1),
        "risk_ha": round(risk_ha, 2),
        "healthy_pct": round(healthy_pct, 1),
        "healthy_ha": round(healthy_ha, 2),
        "cloud_mask_pct": round(cloud_mask_pct, 1),
    }

def render_heatmap(health_matrix: np.ndarray) -> io.BytesIO:
    """
    Строит тепловую карту матрицы health_matrix с помощью matplotlib:
      - Палитра RdYlGn (vmin=0.0, vmax=0.85)
      - nan отображаются нейтральным серым цветом (bad='lightgray')
      - Горизонтальный colorbar с метками зон
      - Красный контур вокруг зон стресса (< 0.25)
    Возвращает BytesIO с изображением PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1e222a")

    cmap = matplotlib.colormaps.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#3a3f4d")

    im = ax.imshow(
        health_matrix,
        cmap=cmap,
        vmin=0.0,
        vmax=0.85,
        interpolation="bilinear",
    )

    stress_mask = np.where(np.isnan(health_matrix), 0, (health_matrix < 0.25).astype(int))
    if np.any(stress_mask > 0):
        try:
            ax.contour(
                stress_mask,
                levels=[0.5],
                colors=["#ff2222"],
                linewidths=1.8,
                linestyles="dashed",
            )
        except Exception:
            pass

    ax.set_title(
        "Тепловая карта здоровья растительности (EVI + MSAVI2)",
        fontsize=13,
        fontweight="bold",
        color="white",
        pad=12,
    )
    ax.axis("off")

    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        fraction=0.05,
        pad=0.06,
        aspect=30,
    )
    cbar.set_ticks([0.10, 0.325, 0.65])
    cbar.set_ticklabels(["< 0.20 Стресс", "0.20–0.45 Риск", "> 0.45 Норма"], color="white", fontsize=10)
    cbar.ax.tick_params(color="white", length=4)
    cbar.outline.set_edgecolor("#555555")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf

def extract_true_color_rgb(
    visual_url: str,
    center_lat: float,
    center_lon: float,
    box_size_meters: float = 1500.0,
) -> io.BytesIO:
    """
    Извлекает реальный оптический снимок высокого разрешения (RGB, True Color)
    непосредственно с борта спутника Sentinel-2 для заданного окна поля.
    Исключает использование архивных карт прошлых лет.
    """
    from PIL import Image

    min_lon, min_lat, max_lon, max_lat = get_bounding_box_4326(
        center_lat, center_lon, box_size_meters
    )

    with rasterio.open(visual_url) as src:
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", src.crs, min_lon, min_lat, max_lon, max_lat
        )
        window = from_bounds(left, bottom, right, top, transform=src.transform)

        rgb_raw = src.read([1, 2, 3], window=window)

    rgb = np.moveaxis(rgb_raw, 0, -1)

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


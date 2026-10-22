"""
core/stac_loader.py — Поиск и получение спутниковых снимков Sentinel-2 L2A в реальном времени
через STAC API Microsoft Planetary Computer.

ВАЖНО:
- Запрещено использовать статичные растровые подложки или кэшированные карты прошлых лет.
- Запрос к STAC API выполняется в реальном времени.
- Динамический диапазон поиска: строго от (сегодня минус 14 дней) до текущего дня.
- Автоматический выбор кадра с минимальной облачностью в пределах актуального окна.
"""

from datetime import datetime, timezone, timedelta
from pystac_client import Client
import planetary_computer
import requests

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

REQUIRED_BANDS = ("B02", "B04", "B08", "SCL")

def get_sentinel_item(
    lat: float,
    lon: float,
    max_cloud: int = 20,
    days_back: int = 14,
    end_date=None,
) -> dict:
    """
    Выполняет запрос в реальном времени к STAC API Microsoft Planetary Computer
    и находит самый свежий снимок Sentinel-2 L2A с минимальной облачностью за последние 14 дней.

    Параметры
    ----------
    lat, lon : float
        Координаты центра интереса (WGS-84).
    max_cloud : int
        Максимально допустимый порог облачности (по умолчанию 20%).
    days_back : int
        Глубина поиска в прошлое (по умолчанию 14 дней).
    end_date : date | None
        Конечная дата. Если None, используется текущая дата (сегодня в UTC).

    Возвращает
    ----------
    dict  с метаданными реального времени и прямыми COG URL для каналов + динамический тайловый URL.
    """

    if end_date is None:
        today = datetime.now(timezone.utc).date()
    elif isinstance(end_date, str):
        today = datetime.strptime(end_date, "%Y-%m-%d").date()
    else:
        today = end_date

    start_date = today - timedelta(days=days_back)
    date_range = f"{start_date.isoformat()}/{today.isoformat()}"

    catalog = Client.open(STAC_API_URL)

    search = catalog.search(
        collections=[COLLECTION],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())

    if not items:
        search_all = catalog.search(
            collections=[COLLECTION],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            datetime=date_range,
        )
        items = list(search_all.items())

    if not items:
        raise RuntimeError(
            f"За последние {days_back} дней (с {start_date.strftime('%d.%m.%Y')} по {today.strftime('%d.%m.%Y')}) "
            f"спутник Sentinel-2 не зафиксировал пролетов над координатами ({lat:.4f}, {lon:.4f}). "
            f"Попробуйте выбрать соседнюю локацию или увеличить допустимую облачность."
        )

    items.sort(
        key=lambda it: (
            it.properties.get("eo:cloud_cover", 100.0),
            -it.datetime.timestamp() if it.datetime else 0,
        )
    )

    best = items[0]
    signed = planetary_computer.sign(best)

    urls = {}
    for band in REQUIRED_BANDS:
        asset = signed.assets.get(band) or signed.assets.get(band.lower())
        if asset is None:
            raise RuntimeError(f"В снимке {best.id} отсутствует канал {band}.")
        urls[band] = asset.href

    visual_asset = signed.assets.get("visual") or signed.assets.get("TCI")
    if visual_asset:
        urls["visual"] = visual_asset.href

    cloud_pct = best.properties.get("eo:cloud_cover", 0.0)
    scene_dt = best.datetime
    if scene_dt is None:
        dt_str = best.properties.get("datetime") or best.properties.get("start_datetime")
        scene_dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))

    scene_date = scene_dt.date()
    days_ago = max(0, (today - scene_date).days)
    platform = best.properties.get("platform", "Sentinel-2")

    tile_url = None
    tilejson_asset = signed.assets.get("tilejson")
    if tilejson_asset:
        try:
            r = requests.get(tilejson_asset.href, timeout=5)
            if r.status_code == 200:
                tiles = r.json().get("tiles")
                if tiles and len(tiles) > 0:
                    tile_url = tiles[0]
        except Exception:
            pass

    return {
        "meta": {
            "date": scene_date.strftime("%d.%m.%Y"),
            "datetime_iso": scene_dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "days_ago": days_ago,
            "cloud_pct": round(float(cloud_pct), 1),
            "scene_id": best.id,
            "platform": platform.upper(),
            "search_window": f"{start_date.strftime('%d.%m.%Y')} — {today.strftime('%d.%m.%Y')}",
            "tile_url": tile_url,
        },
        "urls": urls,
    }

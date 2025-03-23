from typing import List, Tuple

import msgspec
import asyncio
from rich import print
from rsi_download.auth import get_access_token
from rsi_download.download.product import download_products_data
from rsi_download.cli import (
    show_preview_urls,
    Preview,
    get_selected_products,
)
from rsi_download.download.search import search_odata
import math



async def download_core(
    x: str,
    y: str,
    z: str,
    date_min: str,
    date_max: str,
    username: str,
    password: str,
    api_key: str = None,
    max_: int = 100,
    cloud_coverage: float = 20.0,
    debug: bool = False,
    tci: bool = False,
    platform_name: str = "S2",
):
    """
    X tile x coordinate
    Y tile y coordinate 
    Z zoom level
    DATE_MIN start date in format YYYYMM
    DATE_MAX end date in format YYYYMM
    """
    lat, long = tile_to_latlon(float(x), float(y), float(z))
    time_gt = f"{date_min[:4]}-{date_min[4:6]}-01T00:00:00.000Z"
    year = int(date_max[:4])
    month = int(date_max[4:])
    if month == 12:
        next_year = year + 1
        next_month = 1
    else:
        next_year = year
        next_month = month + 1
    time_lt = f"{next_year}-{next_month:02d}-01T00:00:00.000Z"
    
    print(f"coordinates: lat: {lat:.4f}, long: {long:.4f}")
    print(f"maximum results: {max_}")
    print(f"cloud coverage percentage less then: {cloud_coverage:.2f}")
    print(f"time_gt: {time_gt}, time_lt: {time_lt}")
    search_data = await search_odata(long, lat, cloud_coverage, time_lt, time_gt, max_, platform_name)
    if debug:
        print("DEBUG: Search request data is saved to disk.")
        with open("search_data.json", "wb") as f:
            f.write(msgspec.json.encode(search_data))
    preview_urls: List[Preview] = show_preview_urls(search_data, platform_name)
    print("start downloading all data ...")
    products_to_download = get_selected_products(
        search_json=search_data, preview_urls=preview_urls, product_ids=list(range(len(preview_urls)))
    )
    tokens = get_access_token(username, password)

    try:
        for i, (product, preview) in enumerate(zip(products_to_download, preview_urls)):
            print(f"[{i+1}/{len(products_to_download)}] downloading {product.id} ...")
            await asyncio.shield(download_products_data(
                [product], [preview], tokens.access_token, tci_only=tci
            ))
    except asyncio.CancelledError:
        print("\nDownload cancelled, exiting...")
        return

def tile_to_latlon(x: int, y: int, z: int, get_center: bool = True) -> Tuple[float, float]:
    """
    Convert XYZ tile coordinates to latitude/longitude
    
    Args:
        x: Tile X coordinate
        y: Tile Y coordinate
        z: Zoom level
        get_center: If True, returns the center point coordinates. If False, returns the top-left corner.
        
    Returns:
        Tuple of (latitude, longitude)
    """
    n = 2.0 ** z
    if get_center:
        x += 0.5
        y += 0.5
    
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


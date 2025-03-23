from __future__ import annotations

from typing import List

import msgspec
import httpx

from rsi_download.exceptions import SearchException
from rsi_download.geo.geo_types import GeoJsonPolygon

ESA_SEARCH_URL = r"https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


class ContentData(msgspec.Struct, rename="pascal"):
    """Odata search result start and end date"""

    start: str
    end: str


class Asset(msgspec.Struct, rename="pascal"):
    """Odata search Asset"""

    type_: str = msgspec.field(name="Type")
    id: str
    download_link: str
    s3_path: str


class SearchResult(msgspec.Struct, rename="pascal"):
    """Odata search Result"""

    id: str
    name: str
    content_length: int
    origin_date: str
    s3_path: str
    content_date: ContentData
    geo_footprint: GeoJsonPolygon
    assets: List[Asset]


class SearchContent(msgspec.Struct):
    value: List[SearchResult]
    next_link: str | None = msgspec.field(default=None, name="@odata.nextLink")


async def search_odata(
    long: float,
    lat: float,
    cloud_coverage: float,
    time_lt: str,
    time_gt: str,
    max_: int,
    platform_name: str,
) -> SearchContent:
    # filter voor zoeken op cloudCover, Productype en orbitDirection.
    # lt = less then
    # eq = equal to
    # gt = greater then
    # sentinel-2
    if platform_name == "S2":
        search_filter = f"OData.CSC.Intersects(area=geography'SRID=4326;POINT ({long:.4f} {lat:.4f})') and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt {cloud_coverage:.2f}) and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and ContentDate/Start gt {time_gt} and ContentDate/Start lt {time_lt}"
    elif platform_name == "S1":
        search_filter = f"OData.CSC.Intersects(area=geography'SRID=4326;POINT ({long:.4f} {lat:.4f})') and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'IW_GRDH_1S') and ContentDate/Start gt {time_gt} and ContentDate/Start lt {time_lt}"
    elif platform_name == "WV3":
        search_filter = f"OData.CSC.Intersects(area=geography'SRID=4326;POINT ({long:.4f} {lat:.4f})') and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'platformName' and att/OData.CSC.StringAttribute/Value eq 'WorldView-3') and ContentDate/Start gt {time_gt} and ContentDate/Start lt {time_lt}"
    else:
        raise ValueError(f"Invalid platform name: {platform_name}")

    async with httpx.AsyncClient() as client:
        r = await client.get(
            url=f"{ESA_SEARCH_URL}?$filter={search_filter}&$top={max_}&$expand=Assets",
            timeout=60,
        )
        if not r.status_code == 200:
            raise SearchException(f"Error getting data: {r.text}")
        return msgspec.json.decode(r.content, type=SearchContent)

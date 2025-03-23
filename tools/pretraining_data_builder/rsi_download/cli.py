from typing import Tuple, List

from rich.table import Table
from rich.console import Console
import re
import msgspec
from datetime import datetime
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rsi_download.exceptions import InvalidWktPointArgument, InvalidDateRangeArgument
from rsi_download.download.search import SearchContent, SearchResult


class Preview(msgspec.Struct):
    id: str
    productid: str
    url: str
    origin_date: str
    name: str


progress = Progress(
    TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    DownloadColumn(),
    "•",
    TransferSpeedColumn(),
    "•",
    TimeRemainingColumn(),
)


# "2022-05-03T00:00:00.000Z"
ESA_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def convert_to_timestamp(datestring="", dateformat="%d-%m-%Y %H:%M:%S") -> str:
    if len(datestring) > 10:
        source = datetime.strptime(datestring, dateformat)
    else:
        source = datetime.strptime(datestring, "%d-%m-%Y")
    return source.strftime(ESA_DATE_FORMAT)


def daterange_to_timestamp(daterange: str) -> Tuple[str, str]:
    if "," not in daterange:
        raise InvalidDateRangeArgument(
            f'Give a valid daterange string. for example: "11-08-2023 00:00:00,11-09-2023 00:00:00" \n Daterange received: {daterange}'
        )
    gt, lt = daterange.split(",")
    try:
        time_gt = convert_to_timestamp(datestring=gt)
    except ValueError:
        raise InvalidDateRangeArgument(
            f"Invalid dateformat encountered for time_gt: {gt}. Dateformat expected: %d-%m-%Y or %d-%m-%Y %H:%M:%S"
        )
    try:
        time_lt = convert_to_timestamp(datestring=lt)
    except ValueError:
        raise InvalidDateRangeArgument(
            f"Invalid dateformat encountered for time_lt: {lt}. Dateformat expected: %d-%m-%Y or %d-%m-%Y %H:%M:%S"
        )
    return time_gt, time_lt


def wkt_to_point(wktstring: str) -> Tuple[float, ...]:
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", wktstring)
    if len(nums) != 2:
        raise InvalidWktPointArgument(
            f"Give a valid WKT string. for example: POINT(-9.1372 38.7000). WKT received: {wktstring}"
        )
    return tuple(float(n) for n in nums)


def show_preview_urls(search_json: SearchContent, platform_name: str) -> List[Preview]:
    """
    Show a list of preview urls for downloading in the terminal

    :param search_json: SearchContent object
    """
    # print(search_json.value)
    preview_urls = [
        Preview(
            id=str(i),
            productid=v.id,
            url=v.assets[0].download_link,
            origin_date=v.content_date.start,
            name=v.name,
        )
        for i, v in enumerate(search_json.value)
    ]
    table = Table(title=f"RSI Preview Url's")
    table.add_column("ID", justify="left", style="magenta")
    table.add_column("Acquisition Time", justify="left", style="blue")
    table.add_column("Name", justify="left", style="magenta")

    for entry in preview_urls:
        table.add_row(
            entry.id,
            f'[link={entry.url.replace("(", "%28").replace(")", "%29")}]{entry.origin_date}[/link]',
            entry.name,
        )

    console = Console()
    console.print(table)
    return preview_urls


def get_selected_products(
    search_json: SearchContent, preview_urls: List[Preview], product_ids: str
) -> List[SearchResult]:
    """
    Return the selected items from the search_json by the preview url id.

    :param search_json: SearchContent
    :param preview_urls: List[Preview]
    :param product_ids: string of preview ids
    :return: List[SearchResult]
    """
    download_product_ids = [
        item.productid
        for item in preview_urls
        if item.id in [str(n) for n in product_ids]
    ]
    return [x for x in search_json.value if x.id in download_product_ids]

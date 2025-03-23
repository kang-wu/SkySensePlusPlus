from typing import List

import msgspec


class Coordinate(msgspec.Struct):
    long: float
    lat: float


class GeoJsonPolygon(msgspec.Struct):
    type: str
    coordinates: List[List[List[float]]]

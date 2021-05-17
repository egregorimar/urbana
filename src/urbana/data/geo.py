"""Tools to manage geospatial data."""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


def harversine_distance_btw_geoseries_point(
    geoseries: gpd.GeoSeries, point: Point
) -> pd.Series:
    """Compute harversine distance between a geopandas series of points and a shapely point

    Args:
        geoseries (gpd.GeoSeries): The geopandas dataseries
        point (Point): the point to compute the distance from

    Returns:
        pd.Series: a pandas series with the distance between the point and the geoseries
    """
    lon1 = np.radians(geoseries.x)
    lon2 = np.radians(point.x)

    lat2 = np.radians(point.y)
    lat1 = np.radians(geoseries.y)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    R_earth = 6371
    return c * R_earth

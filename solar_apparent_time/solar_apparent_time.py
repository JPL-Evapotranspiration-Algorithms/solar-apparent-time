
from typing import Union
from datetime import datetime, timedelta

from geopandas import GeoSeries
import numpy as np
import pandas as pd
import rasters as rt
from rasters import SpatialGeometry


def _parse_time(time_UTC: Union[datetime, str, list, np.ndarray]) -> np.ndarray:
    """
    Convert a time or list/array of times to a numpy array of datetime64 objects.
    Accepts a single datetime, string, or a list/array of either.

    Parameters
    ----------
    time_UTC : datetime, str, list, or np.ndarray
        The UTC time(s) as datetime object(s), string(s), or array-like.

    Returns
    -------
    np.ndarray
        Array of datetime64 objects.
    """
    if isinstance(time_UTC, (str, datetime)):
        return np.array([time_UTC], dtype='datetime64[ns]')
    
    # If already array-like
    arr = np.asarray(time_UTC)
    
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr
    
    # Use numpy's datetime conversion for better performance
    return np.array(arr, dtype='datetime64[ns]')

def _broadcast_time_and_space(times: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Broadcast time and longitude arrays to compatible shapes for element-wise operations.

    Parameters
    ----------
    times : np.ndarray
        Array of times (datetime64).
    lons : np.ndarray
        Array of longitudes (degrees).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Broadcasted arrays of times and longitudes.
    """
    times = np.asarray(times)
    lons = np.asarray(lons)
    
    if times.shape == ():
        times = times[None]

    if lons.shape == ():
        lons = lons[None]

    # For 1D arrays, create proper broadcasting
    if times.ndim == 1 and lons.ndim == 1:
        if times.shape[0] == 1:
            # Single time, multiple lons: repeat time for each lon
            times_b = np.repeat(times, lons.shape[0])
            lons_b = lons
        elif lons.shape[0] == 1:
            # Multiple times, single lon: repeat lon for each time  
            times_b = times
            lons_b = np.repeat(lons, times.shape[0])
        else:
            # Both are multi-element: use meshgrid for full cross-product
            times_b, lons_b = np.meshgrid(times, lons, indexing='ij')
        return times_b, lons_b
    else:
        # Use numpy's broadcasting for higher dimensions
        return np.broadcast_arrays(times, lons)

def extract_lat_lon(geometry: Union[SpatialGeometry, GeoSeries]) -> SpatialGeometry:
    """
    Extract the SpatialGeometry from a RasterGeometry or GeoSeries.

    Parameters
    ----------
    geometry : SpatialGeometry or GeoSeries
        The geometry object to extract from.

    Returns
    -------
    SpatialGeometry
        The extracted SpatialGeometry.
    """
    if isinstance(geometry, SpatialGeometry):
        lat = geometry.lat
        lon = geometry.lon
    elif isinstance(geometry, GeoSeries):
        lat = geometry.y
        lon = geometry.x
    else:
        raise ValueError("geometry must be SpatialGeometry or GeoSeries")
    return lat, lon

def calculate_solar_hour_of_day(
    time_UTC: Union[datetime, str, list, np.ndarray],
    geometry: Union[SpatialGeometry, GeoSeries] = None,
    lat: Union[np.ndarray, float] = None,
    lon: Union[np.ndarray, float] = None
) -> np.ndarray:
    """
    Calculate the solar hour of day for given UTC time(s) and spatial information.

    Parameters
    ----------
    time_UTC : datetime, str, list, or np.ndarray
        UTC time(s) as datetime object(s), string(s), or array-like.
    geometry : SpatialGeometry, optional
        SpatialGeometry or RasterGeometry object with longitude attribute.
    lat : float or np.ndarray, optional
        Latitude(s) in degrees (not used, included for API compatibility).
    lon : float or np.ndarray, optional
        Longitude(s) in degrees. Required if geometry is not provided.

    Returns
    -------
    np.ndarray
        Array of solar hour of day values, same shape as broadcasted input.

    Notes
    -----
    The solar hour of day is the local solar time in hours, accounting for longitude offset.
    """
    times = _parse_time(time_UTC)

    if (lat is None or lon is None) and geometry is not None:
        lat, lon = extract_lat_lon(geometry)
    elif lon is None:
        raise ValueError("longitude is required when geometry is not provided")

    times = np.asarray(times)
    lon = np.asarray(lon)
    if times.ndim == 1 and lon.ndim == 1 and times.shape == lon.shape:
        times_b = times
        lons_b = lon
    else:
        times_b, lons_b = _broadcast_time_and_space(times, lon)

    hour_UTC = (
        times_b.astype('datetime64[h]').astype(int) % 24
        + (times_b.astype('datetime64[m]').astype(int) % 60) / 60
        + (times_b.astype('datetime64[s]').astype(int) % 60) / 3600
    )

    offset = np.radians(lons_b) / np.pi * 12
    hour_of_day = hour_UTC + offset
    hour_of_day = np.where(hour_of_day < 0, hour_of_day + 24, hour_of_day)
    hour_of_day = np.where(hour_of_day > 24, hour_of_day - 24, hour_of_day)

    return hour_of_day

def calculate_solar_day_of_year(
    time_UTC: Union[datetime, str, list, np.ndarray],
    geometry: Union[SpatialGeometry, GeoSeries] = None,
    lat: Union[np.ndarray, float] = None,
    lon: Union[np.ndarray, float] = None
) -> np.ndarray:
    """
    Calculate the solar day of year for given UTC time(s) and spatial information.

    Parameters
    ----------
    time_UTC : datetime, str, list, or np.ndarray
        UTC time(s) as datetime object(s), string(s), or array-like.
    geometry : SpatialGeometry, optional
        SpatialGeometry or RasterGeometry object with longitude attribute.
    lat : float or np.ndarray, optional
        Latitude(s) in degrees (not used, included for API compatibility).
    lon : float or np.ndarray, optional
        Longitude(s) in degrees. Required if geometry is not provided.

    Returns
    -------
    np.ndarray
        Array of solar day of year values, same shape as broadcasted input.

    Notes
    -----
    The solar day of year is the day of year at the local solar time, accounting for longitude offset.
    """
    times = _parse_time(time_UTC)

    if (lat is None or lon is None) and geometry is not None:
        lat, lon = extract_lat_lon(geometry)
    elif lon is None:
        raise ValueError("longitude is required when geometry is not provided")

    # Handle 1D time and lon inputs of the same length: pair element-wise
    times = np.asarray(times)
    lon = np.asarray(lon)
    if times.ndim == 1 and lon.ndim == 1 and times.shape == lon.shape:
        times_b = times
        lons_b = lon
    else:
        # Broadcast to 2D if not matching 1D
        times_b, lons_b = _broadcast_time_and_space(times, lon)

    # More efficient day of year extraction using numpy datetime operations
    # Convert to datetime64[D] (days since epoch) and then calculate day of year
    year_start = times_b.astype('datetime64[Y]')  # Start of each year
    doy_UTC = ((times_b.astype('datetime64[D]') - year_start.astype('datetime64[D]')) + 1).astype(int)

    hour_UTC = (
        times_b.astype('datetime64[h]').astype(int) % 24
        + (times_b.astype('datetime64[m]').astype(int) % 60) / 60
        + (times_b.astype('datetime64[s]').astype(int) % 60) / 3600
    )

    offset = np.radians(lons_b) / np.pi * 12
    hour_of_day = hour_UTC + offset
    day_of_year = doy_UTC.copy()
    
    # Adjust day of year for timezone offsets
    day_of_year = np.where(hour_of_day < 0, day_of_year - 1, day_of_year)
    day_of_year = np.where(hour_of_day > 24, day_of_year + 1, day_of_year)
    
    # Handle boundary conditions - clamp to valid day range for the year
    # Note: This assumes we want to stay within the same calendar year
    # Day 0 becomes day 1, day > 365/366 becomes last day of year
    day_of_year = np.maximum(day_of_year, 1)
    
    # Get the actual last day of the year for each time
    year_values = times_b.astype('datetime64[Y]').astype(int) + 1970
    # Check if leap year: divisible by 4, but not by 100 unless also by 400
    is_leap = ((year_values % 4 == 0) & (year_values % 100 != 0)) | (year_values % 400 == 0)
    max_day = np.where(is_leap, 366, 365)
    day_of_year = np.minimum(day_of_year, max_day)

    return day_of_year

def UTC_to_solar(time_UTC: datetime, lon: float) -> datetime:
    """
    Convert Coordinated Universal Time (UTC) to solar apparent time at a given longitude.

    Parameters
    ----------
    time_UTC : datetime
        The UTC time.
    lon : float
        The longitude in degrees.

    Returns
    -------
    datetime
        The solar time at the given longitude.
    """
    return time_UTC + timedelta(hours=(np.radians(lon) / np.pi * 12))

def solar_to_UTC(time_solar: datetime, lon: float) -> datetime:
    """
    Convert solar apparent time to Coordinated Universal Time (UTC) at a given longitude.

    Parameters
    ----------
    time_solar : datetime
        The solar time.
    lon : float
        The longitude in degrees.

    Returns
    -------
    datetime
        The UTC time at the given longitude.
    """
    return time_solar - timedelta(hours=(np.radians(lon) / np.pi * 12))

def UTC_offset_hours_for_longitude(lon: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate the offset in hours from UTC based on longitude.

    Parameters
    ----------
    lon : float or np.ndarray
        Longitude(s) in degrees.

    Returns
    -------
    float or np.ndarray
        The calculated offset in hours from UTC.
    """
    # Convert longitude to radians and calculate the offset in hours from UTC
    return np.radians(lon) / np.pi * 12

def UTC_offset_hours_for_area(geometry: rt.RasterGeometry) -> rt.Raster:
    """
    Calculate the UTC offset in hours for a given raster geometry.

    Parameters
    ----------
    geometry : rt.RasterGeometry
        The raster geometry object with longitude information.

    Returns
    -------
    rt.Raster
        The UTC offset in hours as a raster.
    """
    return rt.Raster(np.radians(geometry.lon) / np.pi * 12, geometry=geometry)

def solar_day_of_year_for_area(time_UTC: datetime, geometry: rt.RasterGeometry) -> rt.Raster:
    """
    Calculate the solar day of year for a given UTC time and raster geometry.

    Parameters
    ----------
    time_UTC : datetime
        The UTC time.
    geometry : rt.RasterGeometry
        The raster geometry object with longitude information.

    Returns
    -------
    rt.Raster
        The day of the year as a raster.
    """
    doy_UTC = time_UTC.timetuple().tm_yday
    hour_UTC = time_UTC.hour + time_UTC.minute / 60 + time_UTC.second / 3600
    UTC_offset_hours = UTC_offset_hours_for_area(geometry=geometry)
    hour_of_day = hour_UTC + UTC_offset_hours
    doy = doy_UTC
    doy = rt.where(hour_of_day < 0, doy - 1, doy)
    doy = rt.where(hour_of_day > 24, doy + 1, doy)

    return doy

def solar_day_of_year_for_longitude(
    time_UTC: datetime, 
    lon: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate the solar day of year for a given UTC time and longitude(s).

    Parameters
    ----------
    time_UTC : datetime
        The UTC time to calculate the day of year for.
    lon : float or np.ndarray
        Longitude(s) in degrees.

    Returns
    -------
    float or np.ndarray
        The calculated day of year.
    """
    # Support single datetime or list/array/Series of datetimes
    import numpy as np
    import pandas as pd

    def process_single_time(single_time, lon):
        DOY_UTC = single_time.timetuple().tm_yday
        hour_UTC = single_time.hour + single_time.minute / 60 + single_time.second / 3600
        offset = UTC_offset_hours_for_longitude(lon)
        hour_of_day = hour_UTC + offset
        DOY = DOY_UTC
        # Adjust the day of year if the hour of day is outside the range [0, 24]
        if hour_of_day < 0:
            DOY -= 1
        if hour_of_day > 24:
            DOY += 1
        return DOY

    # Handle list, np.ndarray, pd.Series
    if isinstance(time_UTC, (list, np.ndarray, pd.Series)):
        # If lon is array-like, broadcast
        if isinstance(lon, (list, np.ndarray, pd.Series)):
            return np.array([process_single_time(t, l) for t, l in zip(time_UTC, lon)])
        else:
            return np.array([process_single_time(t, lon) for t in time_UTC])
    else:
        return process_single_time(time_UTC, lon)

def solar_hour_of_day_for_area(time_UTC: datetime, geometry: rt.RasterGeometry) -> rt.Raster:
    """
    Calculate the solar hour of day for a given UTC time and raster geometry.

    Parameters
    ----------
    time_UTC : datetime
        The UTC time.
    geometry : rt.RasterGeometry
        The raster geometry object with longitude information.

    Returns
    -------
    rt.Raster
        The hour of the day as a raster.
    """
    hour_UTC = time_UTC.hour + time_UTC.minute / 60 + time_UTC.second / 3600
    UTC_offset_hours = UTC_offset_hours_for_area(geometry=geometry)
    hour_of_day = hour_UTC + UTC_offset_hours
    hour_of_day = rt.where(hour_of_day < 0, hour_of_day + 24, hour_of_day)
    hour_of_day = rt.where(hour_of_day > 24, hour_of_day - 24, hour_of_day)

    return hour_of_day

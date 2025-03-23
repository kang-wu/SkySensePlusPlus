import numpy as np
import mercantile
from pyresample import bilinear, kd_tree, geometry

TILE_SIZE = 256

class LngLatTransfer():

    def __init__(self):
        self.x_pi = 3.14159265358979324 * 3000.0 / 180.0
        self.pi = np.pi  # π
        self.a = 6378245.0 
        self.es = 0.00669342162296594323  
        pass

    def GCJ02_to_BD09(self, gcj_lng, gcj_lat):
        """
        Convert coordinates from GCJ02 to BD09 coordinate system
        :param lng: Longitude in GCJ02 coordinate system
        :param lat: Latitude in GCJ02 coordinate system
        :return: Converted longitude and latitude in BD09
        """
        z = np.sqrt(gcj_lng * gcj_lng + gcj_lat * gcj_lat) + 0.00002 * np.sin(gcj_lat * self.x_pi)
        theta = np.arctan2(gcj_lat, gcj_lng) + 0.000003 * np.cos(gcj_lng * self.x_pi)
        bd_lng = z * np.cos(theta) + 0.0065
        bd_lat = z * np.sin(theta) + 0.006
        return bd_lng, bd_lat


    def BD09_to_GCJ02(self, bd_lng, bd_lat):
        '''
        Convert coordinates from BD09 to GCJ02 coordinate system
        :param bd_lng: Longitude in BD09 coordinate system
        :param bd_lat: Latitude in BD09 coordinate system
        :return: Converted longitude and latitude in GCJ02
        '''
        x = bd_lng - 0.0065
        y = bd_lat - 0.006
        z = np.sqrt(x * x + y * y) - 0.00002 * np.sin(y * self.x_pi)
        theta = np.arctan2(y, x) - 0.000003 * np.cos(x * self.x_pi)
        gcj_lng = z * np.cos(theta)
        gcj_lat = z * np.sin(theta)
        return gcj_lng, gcj_lat


    def WGS84_to_GCJ02(self, lng, lat):
        '''
        Convert coordinates from WGS84 to GCJ02 coordinate system
        :param lng: Longitude in WGS84 coordinate system
        :param lat: Latitude in WGS84 coordinate system
        :return: Converted longitude and latitude in GCJ02
        '''
        dlat = self._transformlat(lng - 105.0, lat - 35.0)
        dlng = self._transformlng(lng - 105.0, lat - 35.0)
        radlat = lat / 180.0 * self.pi
        magic = np.sin(radlat)
        magic = 1 - self.es * magic * magic
        sqrtmagic = np.sqrt(magic)
        dlat = (dlat * 180.0) / ((self.a * (1 - self.es)) / (magic * sqrtmagic) * self.pi)
        dlng = (dlng * 180.0) / (self.a / sqrtmagic * np.cos(radlat) * self.pi)
        gcj_lng = lng + dlng
        gcj_lat = lat + dlat
        return gcj_lng, gcj_lat


    def GCJ02_to_WGS84(self, gcj_lng, gcj_lat):
        '''
        Convert coordinates from GCJ02 to WGS84 coordinate system
        :param gcj_lng: Longitude in GCJ02 coordinate system
        :param gcj_lat: Latitude in GCJ02 coordinate system
        :return: Converted longitude and latitude in WGS84
        '''
        dlat = self._transformlat(gcj_lng - 105.0, gcj_lat - 35.0)
        dlng = self._transformlng(gcj_lng - 105.0, gcj_lat - 35.0)
        radlat = gcj_lat / 180.0 * self.pi
        magic = np.sin(radlat)
        magic = 1 - self.es * magic * magic
        sqrtmagic = np.sqrt(magic)
        dlat = (dlat * 180.0) / ((self.a * (1 - self.es)) / (magic * sqrtmagic) * self.pi)
        dlng = (dlng * 180.0) / (self.a / sqrtmagic * np.cos(radlat) * self.pi)
        mglat = gcj_lat + dlat
        mglng = gcj_lng + dlng
        lng = gcj_lng * 2 - mglng
        lat = gcj_lat * 2 - mglat
        return lng, lat


    def BD09_to_WGS84(self, bd_lng, bd_lat):
        '''
        Convert coordinates from BD09 to WGS84 coordinate system
        :param bd_lng: Longitude in BD09 coordinate system
        :param bd_lat: Latitude in BD09 coordinate system
        :return: Converted longitude and latitude in WGS84
        '''
        lng, lat = self.BD09_to_GCJ02(bd_lng, bd_lat)
        return self.GCJ02_to_WGS84(lng, lat)


    def WGS84_to_BD09(self, lng, lat):
        '''
        Convert coordinates from WGS84 to BD09 coordinate system
        :param lng: Longitude in WGS84 coordinate system
        :param lat: Latitude in WGS84 coordinate system
        :return: Converted longitude and latitude in BD09
        '''
        lng, lat = self.WGS84_to_GCJ02(lng, lat)
        return self.GCJ02_to_BD09(lng, lat)


    def _transformlat(self, lng, lat):
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
              0.1 * lng * lat + 0.2 * np.sqrt(np.fabs(lng))
        ret += (20.0 * np.sin(6.0 * lng * self.pi) + 20.0 *
                np.sin(2.0 * lng * self.pi)) * 2.0 / 3.0
        ret += (20.0 * np.sin(lat * self.pi) + 40.0 *
                np.sin(lat / 3.0 * self.pi)) * 2.0 / 3.0
        ret += (160.0 * np.sin(lat / 12.0 * self.pi) + 320 *
                np.sin(lat * self.pi / 30.0)) * 2.0 / 3.0
        return ret


    def _transformlng(self, lng, lat):
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
              0.1 * lng * lat + 0.1 * np.sqrt(np.fabs(lng))
        ret += (20.0 * np.sin(6.0 * lng * self.pi) + 20.0 *
                np.sin(2.0 * lng * self.pi)) * 2.0 / 3.0
        ret += (20.0 * np.sin(lng * self.pi) + 40.0 *
                np.sin(lng / 3.0 * self.pi)) * 2.0 / 3.0
        ret += (150.0 * np.sin(lng / 12.0 * self.pi) + 300.0 *
                np.sin(lng / 30.0 * self.pi)) * 2.0 / 3.0
        return ret

    def WGS84_to_WebMercator(self, lng, lat):
        '''
        Convert coordinates from WGS84 to Web Mercator
        :param lng: Longitude in WGS84
        :param lat: Latitude in WGS84
        :return: Converted Web Mercator coordinates
        '''
        x = lng * 20037508.342789 / 180
        y = np.log(np.tan((90 + lat) * self.pi / 360)) / (self.pi / 180)
        y = y * 20037508.34789 / 180
        return x, y

    def WebMercator_to_WGS84(self, x, y):
        '''
        Convert coordinates from Web Mercator to WGS84
        :param x: Web Mercator x coordinate
        :param y: Web Mercator y coordinate
        :return: Converted longitude and latitude in WGS84
        '''
        lng = x / 20037508.34 * 180
        lat = y / 20037508.34 * 180
        lat = 180 / self.pi * (2 * np.arctan(np.exp(lat * self.pi / 180)) - self.pi / 2)
        return lng, lat


transfer = LngLatTransfer()
def get_tile_array(x, y, z, method='nearest', func_source=None, radius=2, fill_value=0, use_gc02=True):
    """Resample source image data to map tile
    
    Args:
        x, y, z: Tile coordinates
        method: Resampling method ('nearest' or 'bilinear')
        func_source: Function to get source image data
        radius: Search radius in pixels
        fill_value: Value for no data areas
        gc02: Whether the coordinates are in GCJ02 system (True) or WGS84 (False)
        
    Returns:
        ndarray: Resampled tile data
    """
    bounds = mercantile.bounds(x, y, z)
    
    if use_gc02:
        # Convert coordinates from GCJ02 to WGS84
        wgs84_lngs, wgs84_lats = transfer.GCJ02_to_WGS84(
            gcj_lng=np.array([bounds.west, bounds.west, bounds.east, bounds.east]),
            gcj_lat=np.array([bounds.north, bounds.south, bounds.south, bounds.north])
        )
        boundary = list(zip(wgs84_lngs, wgs84_lats))
    else:
        boundary = list(zip(
            [bounds.west, bounds.west, bounds.east, bounds.east],
            [bounds.north, bounds.south, bounds.south, bounds.north]
        ))
    
    source_data = func_source(boundary)
    
    if source_data is None:
        return None
        
    arr_image, arr_lngs, arr_lats = source_data
    
    if use_gc02:
        gcj02_lngs, gcj02_lats = transfer.WGS84_to_GCJ02(arr_lngs, arr_lats)
    else:
        gcj02_lngs, gcj02_lats = arr_lngs, arr_lats
    
    # Define source and target geometries
    source_def = geometry.SwathDefinition(lons=gcj02_lngs, lats=gcj02_lats)
    
    xy_bounds = mercantile.xy_bounds(x, y, z)
    target_def = geometry.AreaDefinition(
        'tile', 'tile', 'tile',
        'EPSG:3857',
        TILE_SIZE, TILE_SIZE,
        (xy_bounds.left, xy_bounds.bottom, xy_bounds.right, xy_bounds.top)
    )
    
    # Resample
    pixel_size = mercantile.CE / 2 ** z / TILE_SIZE
    if method == 'nearest':
        result = kd_tree.resample_nearest(
            source_def, arr_image, target_def,
            radius_of_influence=radius * pixel_size,
            fill_value=fill_value
        )
    elif method == 'bilinear':
        resampler = bilinear.NumpyBilinearResampler(
            source_def, target_def,
            radius_of_influence=radius * pixel_size,
            neighbours=8
        )
        result = resampler.resample(arr_image).astype(arr_image.dtype)
    else:
        raise ValueError(f'Unknown resampling method: {method}')
        
    return result


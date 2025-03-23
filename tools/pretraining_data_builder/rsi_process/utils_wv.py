import os
from osgeo import gdal
import numpy as np
from datetime import datetime
import xml.etree.ElementTree as ET

def parse_metadata(meta_xml_file):
    """Parse the WorldView metadata XML file
    
    Args:
        meta_xml_file: Metadata XML file path
        
    Returns:
        dict: Dictionary containing key metadata information
    """
    record = {}
    
    try:
        tree = ET.parse(meta_xml_file)
        root = tree.getroot()
        
        ns = {'imd': root.tag.split('}')[0].strip('{')}
        
        # Get basic information
        record['satellite_id'] = root.find('.//imd:satelliteID', ns).text
        record['product_type'] = root.find('.//imd:productType', ns).text
        
        # Get acquisition time
        acq_time = root.find('.//imd:firstLineTime', ns).text
        record['sensing_start'] = datetime.strptime(acq_time, '%Y-%m-%dT%H:%M:%S.%fZ')
        
        # Get solar angle
        record['sun_azimuth'] = float(root.find('.//imd:meanSunAz', ns).text)
        record['sun_elevation'] = float(root.find('.//imd:meanSunEl', ns).text)
        
        # Get satellite angle
        record['satellite_azimuth'] = float(root.find('.//imd:meanSatAz', ns).text)
        record['satellite_elevation'] = float(root.find('.//imd:meanSatEl', ns).text)
        
        # Get cloud cover
        cloud_cover = root.find('.//imd:cloudCover', ns)
        record['cloud_cover'] = float(cloud_cover.text) if cloud_cover is not None else None
        
        # Get image range
        record['ul_lon'] = float(root.find('.//imd:ULLon', ns).text)
        record['ul_lat'] = float(root.find('.//imd:ULLat', ns).text)
        record['ur_lon'] = float(root.find('.//imd:URLon', ns).text)
        record['ur_lat'] = float(root.find('.//imd:URLat', ns).text)
        record['ll_lon'] = float(root.find('.//imd:LLLon', ns).text)
        record['ll_lat'] = float(root.find('.//imd:LLLat', ns).text)
        record['lr_lon'] = float(root.find('.//imd:LRLon', ns).text)
        record['lr_lat'] = float(root.find('.//imd:LRLat', ns).text)
        
        # Build WKT format geometry information
        record['geom_wkt'] = create_footprint_wkt(record)
        
    except Exception as e:
        print(f"Error parsing metadata: {str(e)}")
        return None
        
    return record

def create_footprint_wkt(record):
    """Create a WKT format polygon based on corner coordinates
    
    Args:
        record: Dictionary containing corner coordinates
        
    Returns:
        str: WKT format polygon string
    """
    coords = [
        (record['ul_lon'], record['ul_lat']),
        (record['ur_lon'], record['ur_lat']),
        (record['lr_lon'], record['lr_lat']),
        (record['ll_lon'], record['ll_lat']),
        (record['ul_lon'], record['ul_lat']) 
    ]
    
    coord_str = ', '.join([f"{lon} {lat}" for lon, lat in coords])
    return f"POLYGON(({coord_str}))"

def get_band_info(ds):
    """Get the band information of the image
    
    Args:
        ds: GDAL dataset
        
    Returns:
        list: Band information list
    """
    bands = []
    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        band_info = {
            'band_number': i,
            'data_type': gdal.GetDataTypeName(band.DataType),
            'nodata_value': band.GetNoDataValue()
        }
        bands.append(band_info)
    return bands

def read_as_array(ds, window=None):
    """Read image data as a numpy array
    
    Args:
        ds: GDAL dataset
        window: Read window, format as (xoff, yoff, xsize, ysize)
        
    Returns:
        numpy.ndarray: Image data array
    """
    if window is None:
        return ds.ReadAsArray()
    else:
        xoff, yoff, xsize, ysize = window
        return ds.ReadAsArray(xoff, yoff, xsize, ysize)

def get_image_info(fn_img):
    """Get basic information of WorldView image
    
    Args:
        fn_img: Image file path
        
    Returns:
        dict: Image information dictionary
    """
    ds = gdal.Open(fn_img)
    if ds is None:
        raise Exception(f"Cannot open {fn_img}")
    
    info = {
        'width': ds.RasterXSize,
        'height': ds.RasterYSize,
        'bands': ds.RasterCount,
        'projection': ds.GetProjection(),
        'geotransform': ds.GetGeoTransform(),
        'band_info': get_band_info(ds)
    }
    
    xml_file = fn_img.replace('.tif', '.xml')
    if os.path.exists(xml_file):
        metadata = parse_metadata(xml_file)
        if metadata:
            info.update(metadata)
    
    ds = None
    return info

def calculate_stats(fn_img, percentiles=[2, 98]):
    """Calculate the statistics of the image
    
    Args:
        fn_img: Image file path
        percentiles: List of percentiles
        
    Returns:
        dict: Statistics dictionary
    """
    ds = gdal.Open(fn_img)
    stats = {}
    
    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        array = band.ReadAsArray()
        valid_data = array[array != band.GetNoDataValue()]
        
        stats[f'band_{i}'] = {
            'min': np.min(valid_data),
            'max': np.max(valid_data),
            'mean': np.mean(valid_data),
            'std': np.std(valid_data),
            'percentiles': {
                p: np.percentile(valid_data, p)
                for p in percentiles
            }
        }
    
    ds = None
    return stats

def create_quicklook(fn_img, output_file, size=(1024, 1024)):
    """Create a thumbnail
    
    Args:
        fn_img: Image file path
        output_file: Output file path
        size: Output image size
    """
    ds = gdal.Open(fn_img)
    
    if ds.RasterCount >= 3:
        r = ds.GetRasterBand(1).ReadAsArray()
        g = ds.GetRasterBand(2).ReadAsArray()
        b = ds.GetRasterBand(3).ReadAsArray()
        
        def stretch(arr):
            p2, p98 = np.percentile(arr[arr > 0], (2, 98))
            return np.clip((arr - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
        
        rgb = np.dstack([stretch(r), stretch(g), stretch(b)])
        
        from PIL import Image
        img = Image.fromarray(rgb)
        img.thumbnail(size)
        img.save(output_file)
    
    ds = None

def warp(ds, outputBounds, 
         outputBoundsSRS='EPSG:4326',
         xRes=2, yRes=2, 
         targetAlignedPixels=True,
         **kwargs):
    """Reprojection and resampling
    
    Args:
        ds: GDAL dataset
        outputBounds: Output range
        outputBoundsSRS: Output coordinate system
        xRes, yRes: Output resolution
        targetAlignedPixels: Whether to align pixels
        **kwargs: Other GDAL.Warp parameters
    
    Returns:
        GDAL dataset
    """
    options_warp = gdal.WarpOptions(
        format="MEM",
        outputBounds=outputBounds, 
        outputBoundsSRS=outputBoundsSRS,
        xRes=xRes, yRes=yRes, 
        targetAlignedPixels=targetAlignedPixels,
        **kwargs
    )
    ds_warp = gdal.Warp('', ds, options=options_warp)
    return ds_warp 
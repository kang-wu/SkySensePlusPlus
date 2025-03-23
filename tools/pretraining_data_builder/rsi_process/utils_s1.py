import xml.dom.minidom
import os
from glob import glob
import zipfile
from shapely import wkt
import geopandas as gpd
from osgeo import gdal
import imageio.v2 as iio

def parse_metadata(meta_xml_file):
    """Parse Sentinel-1 metadata XML file
    
    Args:
        meta_xml_file: Metadata XML file path
        
    Returns:
        dict: Dictionary containing key metadata information
    """
    record = {}

    dom = xml.dom.minidom.parse(meta_xml_file)    # Get sensing start time
    sensing_start = dom.getElementsByTagName('startTime')[0].firstChild.data
    
    product_uri = meta_xml_file.name.split('/')[0]
    
    record.update({
        'product_uri': product_uri,
        'sensing_start': sensing_start,
    })
        
        
    return record

def convert_footprint_to_wkt(footprint):
    """Convert footprint string to WKT format"""
    coords = footprint.strip().split(' ')
    wkt_coords = []
    for coord in coords:
        lat, lon = coord.split(',')
        wkt_coords.append(f"{lon} {lat}")
    return f"MULTIPOLYGON ((({','.join(wkt_coords)})))"

def zip2rec(fn_zip):
    id_img = os.path.splitext(os.path.basename(fn_zip))[0]
    archive = zipfile.ZipFile(fn_zip, 'r')
    xml_files = [f for f in archive.namelist() if f.endswith('-001.xml')]
    if not xml_files:
        raise FileNotFoundError(f"No XML file ending with '-001.xml' found in {fn_zip}")
    fn_xml = archive.open(xml_files[0])
    rec = parse_metadata(fn_xml)
    import pdb; pdb.set_trace()
    # rec['geometry'] = wkt.loads(rec['geom_wkt'])
    thumb = archive.open(os.path.join(f'{id_img}.SAFE', 'preview', 'quick-look.png'))
    thumb = iio.imread(thumb)
    rec['thumb'] = thumb
    return rec

def build_catalog(path, fn='catalog'):
    '''
    fn: filename or None
    '''
    list_fnames = glob(os.path.join(path, 'S2*.zip'))

    list_rec = []
    for fn_zip in list_fnames:
        rec = zip2rec(fn_zip)
        list_rec.append(rec)

    gdf = gpd.GeoDataFrame(list_rec, crs='EPSG:4326').drop(columns='geom_wkt')
    if fn is not None:
        fn_geojson = os.path.join(path, f"{fn}.geojson")
        gdf.to_file(fn_geojson, driver='GeoJSON')
        return fn_geojson
    else:
        return gdf

def make_full_name(rec, band):
    dict_bands = {
        'VV': '001',
        'VH': '002',
    }
    parts = rec['product_uri'].split('_')
    
    satellite = parts[0].lower()  # S1A -> s1a
    mode = parts[1].lower()       # IW -> iw
    product_type = parts[2][:3].lower()  # GRDH -> grd
    polarization = band.lower()         # Assume polarization mode is VV
    start_time = parts[4].lower() # Start time
    end_time = parts[5].lower()   # End time
    id1 = parts[6].lower()        # 058175
    id2 = parts[7].lower()        # 072FF2
    fixed_part = dict_bands[band]          # Replace fixed part with 001
    
    # Concatenate to target format
    file_name = f"{satellite}-{mode}-{product_type}-{polarization}-{start_time}-{end_time}-{id1}-{id2}-{fixed_part}.tiff"

    fn_template = os.path.join(
        rec['product_uri'], 'measurement', file_name
    )
    return fn_template

def warp(
    ds, outputBounds, 
    outputBoundsSRS='EPSG:4326',
    xRes=10, yRes=10, targetAlignedPixels=True,
    **kwargs,
):
    options_warp = gdal.WarpOptions(
        format="MEM",
        outputBounds=outputBounds, 
        outputBoundsSRS=outputBoundsSRS,
        xRes=xRes, yRes=yRes, targetAlignedPixels=targetAlignedPixels,
        **kwargs,
    )
    ds_warp = gdal.Warp('', ds, options=options_warp)
    return ds_warp

def get_ndarray(
    ds, outputBounds, 
    outputBoundsSRS='EPSG:4326',
    xRes=10, yRes=10, targetAlignedPixels=True,
    **kwargs,
):
    ds_warp = warp(
        ds, outputBounds, 
        outputBoundsSRS='EPSG:4326',
        xRes=10, yRes=10, targetAlignedPixels=True,
        **kwargs
    )
    arr = ds_warp.ReadAsArray()
    ds_warp = None
    return arr


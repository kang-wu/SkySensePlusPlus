import xml.dom.minidom
import os
from glob import glob
import zipfile
from shapely import wkt
import geopandas as gpd
from osgeo import gdal
import imageio.v2 as iio

def parse_metadata(meta_xml_file):
    """Parse Sentinel-2 metadata XML file
    
    Args:
        meta_xml_file: Path to metadata XML file
        
    Returns:
        dict: Metadata information including sensing time, product URI, etc.
    """
    record = {}
    try:
        dom = xml.dom.minidom.parse(meta_xml_file)
        
        # Get sensing start time
        sensing_start = dom.getElementsByTagName('DATATAKE_SENSING_START')[0].firstChild.data[0:10]
        
        # Get product URI and image paths
        product_uri = dom.getElementsByTagName('PRODUCT_URI')[0].firstChild.data
        
        image_file = dom.getElementsByTagName('IMAGE_FILE')[0].firstChild.data
        items = image_file.split('/')
        granule_path = items[1]
        img_name = items[4].split('_')[0] + '_' + items[4].split('_')[1]
        
        # Get footprint
        footprint = dom.getElementsByTagName('EXT_POS_LIST')[0].firstChild.data
        geom_wkt = convert_footprint_to_wkt(footprint)
        
        # Get cloud coverage info
        cloud_coverage = float(dom.getElementsByTagName('Cloud_Coverage_Assessment')[0].firstChild.data)
        cloud_shadow = float(dom.getElementsByTagName('CLOUD_SHADOW_PERCENTAGE')[0].firstChild.data)
        medium_clouds = float(dom.getElementsByTagName('MEDIUM_PROBA_CLOUDS_PERCENTAGE')[0].firstChild.data)
        high_clouds = float(dom.getElementsByTagName('HIGH_PROBA_CLOUDS_PERCENTAGE')[0].firstChild.data)
        
        record.update({
            'product_uri': product_uri,
            'sensing_start': sensing_start,
            'granule_path': granule_path,
            'img_name': img_name,
            'cloud_cover': cloud_coverage,
            'cloud_shadow': cloud_shadow,
            'medium_clouds': medium_clouds,
            'high_clouds': high_clouds,
            'geom_wkt': geom_wkt
        })
        
    except Exception as e:
        print(f'Failed to parse XML: {e}')
        
    return record

def convert_footprint_to_wkt(footprint):
    """Convert footprint string to WKT format"""
    coords = footprint.strip().split(' ')
    wkt_coords = []
    for i in range(0, len(coords), 2):
        wkt_coords.append(f"{coords[i+1]} {coords[i]}")
    return f"MULTIPOLYGON ((({','.join(wkt_coords)})))"

def zip2rec(fn_zip):
    id_img = os.path.splitext(os.path.basename(fn_zip))[0]
    archive = zipfile.ZipFile(fn_zip, 'r')
    fn_xml = archive.open(os.path.join(f'{id_img}.SAFE', 'MTD_MSIL2A.xml'))
    rec = parse_metadata(fn_xml)
    rec['geometry'] = wkt.loads(rec['geom_wkt'])
    thumb = archive.open(os.path.join(f'{id_img}.SAFE', f'{id_img}-ql.jpg'))
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
        'B2': ['B02', '10m'],
        'B3': ['B03', '10m'],
        'B4': ['B04', '10m'],
        'B8': ['B08', '10m'],
        'B5': ['B05', '20m'],
        'B6': ['B06', '20m'],
        'B7': ['B07', '20m'],
        'B8A': ['B8A', '20m'],
        'B11': ['B11', '20m'],
        'B12': ['B12', '20m'],
        'SCL': ['SCL', '20m'],
    }
    fn_template = os.path.join(
        '{p0}', 'GRANULE',
        '{p1}', 'IMG_DATA', "R{p2}",
        '{p3}_{p4}_{p2}.jp2'
    )
    return fn_template.format(**{
        'p0': rec['product_uri'], 
        'p0b': rec['product_uri'].split('.')[0], 
        'p1': rec['granule_path'],
        'p2': dict_bands[band][1],
        'p3': rec['img_name'], 
        'p4': dict_bands[band][0],
    })

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


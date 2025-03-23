import os
import uuid
import numpy as np
import pyproj as prj
from osgeo import gdal
from time import time
import mercantile
from PIL import Image
import imageio.v2 as iio
from tile_resample import (
    get_tile_array,
    transfer
)
import argparse
from rich import print
from rich.progress import track

def get_args_parser():
    parser = argparse.ArgumentParser(description='WorldView to tiles')
    parser.add_argument('--fn_img', help='input file of WorldView image')
    parser.add_argument('--save_dir', default='output_wv/', help='output directory')
    parser.add_argument('--zoom', type=int, default=16, help='zoom level')
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--use_gcj02', action='store_true', default=False)
    return parser.parse_args()

def get_image_by_approximate_boundary(ds_list, boundary, tr, buf=1):
    '''Get image data within a specified boundary
    
    Args:
        ds_list: List of GDAL datasets
        boundary: List of (lng, lat) coordinates
        tr: Geotransformation parameters
        buf: Buffer size
    '''
    arr_lnglat = np.array(boundary)
    tr_from_4326 = prj.Transformer.from_crs(4326, ds_list[0].GetProjection(), always_xy=True)
    
    xx, yy = tr_from_4326.transform(arr_lnglat[:, 0], arr_lnglat[:, 1])
    
    nx = ds_list[0].RasterXSize
    ny = ds_list[0].RasterYSize
    xres = tr[1]
    yres = -tr[5]
    
    row_min = int((tr[3] - yy.max()) / yres)
    row_max = int((tr[3] - yy.min()) / yres)
    col_min = int((xx.min() - tr[0]) / xres)
    col_max = int((xx.max() - tr[0]) / xres)
    
    row_min = max(0, row_min - buf)
    row_max = min(ny - 1, row_max + buf)
    col_min = max(0, col_min - buf)
    col_max = min(nx - 1, col_max + buf)
    
    if row_min > row_max or col_min > col_max:
        return None
    
    arr_image = np.stack([
        ds.ReadAsArray(col_min, row_min, col_max - col_min + 1, row_max - row_min + 1)
        for ds in ds_list
    ])
    
    if np.all(arr_image == 0):
        return None
        
    arr_image = arr_image.transpose((1, 2, 0))
    
    arr_xx = tr[0] + np.arange(col_min, col_max + 1) * xres
    arr_yy = tr[3] - np.arange(row_min, row_max + 1) * yres
    arr_xx, arr_yy = np.meshgrid(arr_xx, arr_yy)
    
    tr_to_4326 = prj.Transformer.from_crs(ds_list[0].GetProjection(), 4326, always_xy=True)
    arr_lngs, arr_lats = tr_to_4326.transform(arr_xx, arr_yy)
    
    return arr_image, arr_lngs, arr_lats

def process_wv(args):
    t_start = time()
    
    fn_img = args.fn_img
    save_dir = args.save_dir
    z = args.zoom
    verbose = args.verbose
    
    os.makedirs(save_dir, exist_ok=True)
    
    ds = gdal.Open(fn_img)
    if ds is None:
        raise Exception(f"Cannot open {fn_img}")
        
    bands = [ds.GetRasterBand(i+1) for i in range(ds.RasterCount)]
    list_arr = [ds]
    
    nx, ny = ds.RasterXSize, ds.RasterYSize
    tr = ds.GetGeoTransform()
    
    if verbose:
        print('Input size:', nx, ny)
        print(gdal.Info(ds, format='json'))
    
    # Calculate the image range
    size_pixel = mercantile.CE / 2 ** z / 256
    radius = np.ceil(max(tr[1], -tr[5]) / size_pixel * 1.5)
    
    buf_ext = 1
    xmin = tr[0] - buf_ext * tr[1]
    ymin = tr[3] + (ny + buf_ext) * tr[5]
    xmax = tr[0] + (nx + buf_ext) * tr[1]
    ymax = tr[3] - buf_ext * tr[5]
    
    tr_to_4326 = prj.Transformer.from_crs(ds.GetProjection(), 4326, always_xy=True)
    arr_lng, arr_lat = tr_to_4326.transform(
        np.array([xmin, xmin, xmax, xmax]),
        np.array([ymax, ymin, ymin, ymax])
    )
    
    if args.use_gcj02:
        arr_lng_final, arr_lat_final = transfer.WGS84_to_GCJ02(arr_lng, arr_lat)
    else:
        arr_lng_final, arr_lat_final = arr_lng, arr_lat
        
    box = (
        arr_lng_final.min(),
        arr_lat_final.min(),
        arr_lng_final.max(),
        arr_lat_final.max()
    )
    
    if verbose:
        coord_system = "GCJ02" if args.use_gcj02 else "WGS84"
        print(f'Input extent, {coord_system}: {box}')
    
    # Calculate the tile range to be processed
    tile_ul = mercantile.tile(box[0], box[3], z)
    tile_lr = mercantile.tile(box[2], box[1], z)
    
    if verbose:
        print('Upperleft  ', str(tile_ul))
        print('Lowerright ', str(tile_lr))
    
    def work(x, y, z):
        arr_tile = get_tile_array(
            x, y, z,
            method='nearest',
            func_source=lambda boundary: get_image_by_approximate_boundary(list_arr, boundary, tr),
            radius=radius,
            use_gc02=args.use_gcj02
        )
        
        if arr_tile is not None:
            save_path = os.path.join(save_dir, str(z), str(x))
            os.makedirs(save_path, exist_ok=True)
            
            # Save as PNG
            if arr_tile.shape[2] >= 3:
                arr_rgb = arr_tile[:, :, :3]
                arr_rgb = np.clip(arr_rgb / 2000. * 255, 0, 255).astype(np.uint8)
                image_tile = Image.fromarray(arr_rgb)
                png_filename = os.path.join(save_path, f'{y}.png')
                image_tile.save(png_filename, format='png')
            
            # Save as NPZ
            dict_arr = {f'B{i+1}': arr_tile[:, :, i] for i in range(arr_tile.shape[2])}
            npz_filename = os.path.join(save_path, f'{y}.npz')
            np.savez_compressed(npz_filename, **dict_arr)
    
    tasks = [
        (x, y) for x in range(tile_ul.x, tile_lr.x + 1)
        for y in range(tile_ul.y, tile_lr.y + 1)
    ]
    
    for x, y in track(tasks, description="Converting tiles..."):
        work(x, y, z)
    
    print("Time cost:", time() - t_start)

def main():
    args = get_args_parser()
    process_wv(args)

if __name__ == '__main__':
    main()
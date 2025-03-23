import os
import uuid
import numpy as np
import pyproj as prj
from osgeo import gdal
from time import time
import mercantile
from PIL import Image
import utils_s1
import imageio.v2 as iio
from tile_resample import (
    get_tile_array,
    transfer
)

import argparse
from rich import print
from rich.progress import track

def get_args_parser():
    parser = argparse.ArgumentParser(description='Sentinel-1 to GCJ02 tiles')
    parser.add_argument('--fn_img', help='input zip file of Sentinel-1 L1C')
    parser.add_argument('--save_dir', default='output_s1/', help='prefix on oss bucket')
    parser.add_argument('--verbose', action='store_true', default=True, help='whether to print info')
    parser.add_argument('--use_gcj02', action='store_true', default=False, help='whether to use GCJ02 coordinate system')
    return parser

def process_s1(args):
    t_start = time()
    fn_img = args.fn_img
    max_target_file = fn_img.split('_')[2][0:8]
    verbose = args.verbose
    save_rgb = True
    nodata = 0

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    thumb_save_dir = os.path.join(save_dir, 'thumb')
    os.makedirs(thumb_save_dir, exist_ok=True)

    print(f"converting {fn_img}...")

    z = 14
    bands = ['VV', 'VH']
    buf = 1

    def get_image_by_approximate_boundary(boundary):
        '''
        boundary: iterable of (lng, lat) in wgs84
        '''
        arr_lnglat = np.array(boundary)
        xx, yy = tr_from_4326.transform(arr_lnglat[:, 0], arr_lnglat[:, 1])
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
            for ds in list_arr
        ])

        for iband in range(arr_image.shape[0]):
            if np.any(arr_image[iband] != nodata):
                break
        else:
            return None
        arr_image = arr_image.transpose((1, 2, 0))
        if arr_image.shape[2] == 1:
            arr_image = arr_image[:, :, 0]
        arr_xx = tr[0] + np.arange(col_min, col_max + 1) * xres
        arr_yy = tr[3] - np.arange(row_min, row_max + 1) * yres
        arr_xx, arr_yy = np.meshgrid(arr_xx, arr_yy)
        arr_lngs, arr_lats = tr_to_4326.transform(arr_xx, arr_yy)
        return arr_image, arr_lngs, arr_lats


    rec = utils_s1.zip2rec(fn_img)
    # import pdb; pdb.set_trace()
    os.makedirs(os.path.join(thumb_save_dir, rec['sensing_start'].replace('-', '')), exist_ok=True)
    thumb_save_path = os.path.join(thumb_save_dir, rec['sensing_start'].replace('-', ''), rec['product_uri'].replace('SAFE', 'png'))
    iio.imwrite(thumb_save_path, rec['thumb'])

    list_arr = []
    for band in bands:
        fn_jp2 = utils_s1.make_full_name(rec, band=band)
        # import pdb; pdb.set_trace()
        fn_jp2 = '/vsizip/' + os.path.join(fn_img, fn_jp2)
        ds = gdal.Open(fn_jp2)
        list_arr.append(ds)
        if band == bands[0]:
            nx, ny = ds.RasterXSize, ds.RasterYSize
            if verbose: print('input size:', nx, ny)
            tr = ds.GetGeoTransform()
            if verbose:
                print(gdal.Info(ds, format='json'))
            # import pdb; pdb.set_trace()
            try:
                proj_wkt = ds.GetProjectionRef()
                if proj_wkt:
                    srs = prj.CRS.from_wkt(proj_wkt)
                    epsg = int(srs.to_epsg())
                else:
                    proj_wkt = ds.GetGCPProjection()
                    if proj_wkt:
                        srs = prj.CRS.from_wkt(proj_wkt)
                        epsg = int(srs.to_epsg())
                    else:
                        print("Warning: No projection information found, using default value 4326 (WGS84)")
                        epsg = 4326
            except Exception as e:
                print(f"Warning: Unable to get EPSG code, using default value 4326 (WGS84). Error: {e}")
                epsg = 4326

            if verbose:
                print(f"Used EPSG code: {epsg}")

    size_pixel = mercantile.CE / 2 ** z / 256
    radius = np.ceil(max(tr[1], -tr[5]) / size_pixel * 1.5)

    buf_ext = buf
    xmin = tr[0] - buf_ext * tr[1]
    ymin = tr[3] + (ny + buf_ext) * tr[5]
    xmax = tr[0] + (nx + buf_ext) * tr[1]
    ymax = tr[3] - buf_ext * tr[5]
    xres = tr[1]
    yres = - tr[5]
    if verbose:
        print(
            f'input extent, WGS84, buffered by {buf_ext} pixels: {xmin}, {ymin}, {xmax}, {ymax}'
        )

    tr_to_4326 = prj.Transformer.from_crs(epsg, 4326, always_xy=True)
    tr_from_4326 = prj.Transformer.from_crs(4326, epsg, always_xy=True)
    arr_lng, arr_lat = tr_to_4326.transform(
        np.array([xmin, xmin, xmax, xmax]),
        np.array([ymax, ymin, ymin, ymax])
    )
    # import pdb; pdb.set_trace()
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
        print(f'input extent, {coord_system}: {box}')

    tile_ul = mercantile.tile(box[0], box[3], z)
    tile_lr = mercantile.tile(box[2], box[1], z)

    if verbose:
        print('Upperleft  ', str(tile_ul))
        print('Lowerright ', str(tile_lr))

    def work(x, y, z, save_rgb):
        arr_tile = get_tile_array(
            x, y, z,
            method='nearest',
            func_source=get_image_by_approximate_boundary,
            radius=radius,
            use_gc02=args.use_gcj02
        )
        y_str = str(y)
        if arr_tile is not None:
            indi_gap = arr_tile[:, :, 0] == 0

            dict_arr = {
                band: arr_tile[:, :, i_band]
                for i_band, band in enumerate(bands)
            }
            save_path = os.path.join(save_dir, str(z), str(x))
            os.makedirs(save_path, exist_ok=True)
            
            npz_filename = os.path.join(save_path, f'{y_str}_{max_target_file}.npz')
            
            if indi_gap.any():
                if os.path.exists(npz_filename):
                    try:
                        fp = np.load(npz_filename)
                        for band in bands:
                            dict_arr[band][indi_gap] = fp[band][indi_gap]
                        
                    except Exception as e:
                        print(e)
                        print("datasize is 0", npz_filename)
                        pass
            
            np.savez_compressed(npz_filename, **dict_arr)
            if verbose:
                print(f"npz file for X={str(x)}, Y={y_str}, Z={str(z)} date={max_target_file} generated!")
            if save_rgb:
                arr_rgb = np.stack([dict_arr['B4'], dict_arr['B3'], dict_arr['B2']], axis=-1)
                arr_rgb = np.clip(arr_rgb / 3000. * 255, 0, 255).astype(np.uint8)
                image_tile = Image.fromarray(arr_rgb)
                
                png_filename = os.path.join(save_path, f'{y_str}_{max_target_file}.png')
                image_tile.save(png_filename, format='png')

    diff_list = []

    tasks = [
        (x, y) for x in range(tile_ul.x, tile_lr.x + 1)
        for y in range(tile_ul.y, tile_lr.y + 1)
    ]
    
    for x, y in track(tasks, description="converting tiles..."):
        work(x, y, z, save_rgb)
        diff_list.append(os.path.join(str(z), str(x), f'{y}_{max_target_file}.npz'))

    diff_path = os.path.join(save_dir, 'diff', 'new')
    os.makedirs(diff_path, exist_ok=True)
    diff_filename = os.path.join(diff_path, f"{z}-{os.path.splitext(os.path.basename(fn_img))[0]}-{uuid.uuid1()}.txt")
    with open(diff_filename, 'w') as f:
        f.write('\n'.join(diff_list))

    print("time cost :", time() - t_start)

def main():
    args = get_args_parser().parse_args()
    process_s1(args)

if __name__ == '__main__':
    main()


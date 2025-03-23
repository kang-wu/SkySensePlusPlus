from rsi_process.script_s1_tiles import process_s1
from rsi_process.script_s2_tiles import process_s2
from rsi_process.script_wv_tiles import process_wv
import EasyDict as edict


def process_adapter(fn_img, save_dir, verbose, use_gcj02):
    satellite_info = fn_img.split('/')[-1].split('_')[0]
    if 'S2' in satellite_info:
        satellite = 'S2'
    elif 'S1' in satellite_info:
        satellite = 'S1'
    elif 'WV' in satellite_info:
        satellite = 'WV'
    args = edict(fn_img=fn_img, save_dir=save_dir, verbose=verbose, use_gcj02=use_gcj02)
    if satellite == 'S1':
        process_s1(args)
    elif satellite == 'S2':
        process_s2(args)
    elif satellite == 'WV':
        process_wv(args)

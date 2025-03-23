import lmdb
import os
import json
from rich import print
from rsi_download.download_async import download_core
from rsi_process.adapter import process_adapter
import asyncclick as click

@click.command()
@click.argument("lmdb_path", type=click.STRING)
async def read_lmdb_file(lmdb_path):
    """
    Read the LMDB file and print all key-value pairs
    
    Args:
        lmdb_path: LMDB file path
    """
    if not os.path.exists(lmdb_path):
        print(f"Error: LMDB path '{lmdb_path}' does not exist")
        return
        
    try:
        print(f"Reading Pretraining List from LMDB file from {lmdb_path}...")
        env = lmdb.open(lmdb_path, readonly=True)
        total_length = 0
        with env.begin() as txn:
            key = b'length'
            total_length = int(txn.get(key))
            print(f"Total length of the Pretraining Data: {total_length:,}")
            print("Example Data:")
            for i in range(10):
                print(txn.get(f"{i}".encode()).decode('utf-8'))
            for i in range(total_length):
                key = f"{i}".encode()
                data = json.loads(txn.get(key).decode('utf-8'))
                print("*"* 116 + "\n" + f"* Current Data [{i+1} / {total_length}]: {data} *" + "\n" + "*"* 116 )
                print(f"Downloading: {data}")
                await download_core(
                    x=data['x'],
                    y=data['y'],
                    z=data['z'],
                    date_min=data['date_min'],
                    date_max=data['date_max'],
                    username=os.getenv("USERNAME"),
                    password=os.getenv("PASSWORD"),
                    cloud_coverage=20.0,
                    tci=False
                )    
                print('-'* 40)
            print(f"Processing: {data}")
            process_list = os.listdir('out_raw/')
            total_len_process = len(process_list)
            for fn in process_list:
                print(f"Processing: {fn} [{i+1} / {total_len_process}]...")
                process_adapter(
                    fn_img=f'out_raw/{fn}',
                    save_dir='out_processed/',
                    verbose=True,
                    use_gcj02=False
                )
            print('-'* 40)
            print("Done!")
            
    except lmdb.Error as e:
        print(f"Error reading LMDB file: {str(e)}")
    finally:
        env.close()

if __name__ == "__main__":
    read_lmdb_file()
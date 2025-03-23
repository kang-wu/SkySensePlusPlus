import argparse
from rsi_process.adapter import process_adapter

def get_main_parser():
    parser = argparse.ArgumentParser(description='RSI Processing Pipeline')
    parser.add_argument('--fn_img', help='input zip file')
    parser.add_argument('--save_dir', default='output/', help='prefix on oss bucket')
    parser.add_argument('--verbose', action='store_true', default=True, help='whether to print info')
    parser.add_argument('--use_gcj02', action='store_true', default=False, help='whether to use GCJ02 coordinate system')
    return parser


def main():
    parser = get_main_parser()
    args = parser.parse_args()
    process_adapter(args.fn_img, args.save_dir, args.verbose, args.use_gcj02)

if __name__ == '__main__':
    main()

import click
from rsi_download.download_async import download_core
import asyncio

@click.command()
@click.argument("x", type=click.STRING)
@click.argument("y", type=click.STRING)
@click.argument("z", type=click.STRING)
@click.argument("date_min", type=click.STRING)
@click.argument("date_max", type=click.STRING)
@click.option(
    "--username",
    "-u",
    type=click.STRING,
    help="Username for Copernicus Data Space Ecosystem",
)
@click.option(
    "--password", "-p", prompt=True, hide_input=True, confirmation_prompt=False
)
@click.option(
    "--api_key", "-k", prompt=True, hide_input=True, confirmation_prompt=False
)
@click.option(
    "--max",
    "-m",
    "max_",
    default=100,
    type=click.INT,
    show_default=True,
    help="maximum number of results returned",
)
@click.option(
    "--cloud-coverage",
    "-c",
    "cloud_coverage",
    default=10.00,
    type=click.FLOAT,
    show_default=True,
    help="Get only results with a cloud coverage percentage less then the argument given.",
)

@click.option(
    "--platform-name",
    "-n",
    "platform_name",
    default="S2",
    type=click.Choice(["S2", "S1", "WV3"]),
    show_default=True,
    help="Get only results with a platform name.",
)

@click.option(
    "--debug",
    default=False,
    is_flag=True,
    type=click.BOOL,
    show_default=True,
    help="Debug the http requests and extra debug logging",
)
@click.option(
    "--tci",
    default=False,
    is_flag=True,
    type=click.BOOL,
    show_default=True,
    help="Download only True Color Image (TCI)",
)

def main(x, y, z, date_min, date_max, username, password, api_key, max_, cloud_coverage, debug, tci, platform_name):
    return asyncio.run(download_core(x, y, z, date_min, date_max, username, password, api_key, max_, cloud_coverage, debug, tci, platform_name))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序已终止")
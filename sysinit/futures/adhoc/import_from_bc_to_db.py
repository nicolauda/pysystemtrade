from sysdata.config.production_config import get_production_config
from syscore.fileutils import resolve_path_and_filename_for_package
from sysdata.csv.csv_futures_contract_prices import ConfigCsvFuturesPrices
from sysinit.futures.contract_prices_from_split_freq_csv_to_db import (
    init_db_with_split_freq_csv_prices_for_code,
)
SYMBOL = "AEX"
BARCHART_CONFIG = ConfigCsvFuturesPrices(

    input_date_index_name="Time",
    input_skiprows=0,
    input_skipfooter=0,
    input_date_format="ISO8601",
    input_column_mapping=dict(
        OPEN="Open", HIGH="High", LOW="Low", FINAL="Close", VOLUME="Volume"
    ),
)

# assuming bc-utils config pasted into private
datapath = resolve_path_and_filename_for_package(
    get_production_config().get_element_or_default("barchart_path", None)
)

# import prices for a single instrument
init_db_with_split_freq_csv_prices_for_code(instrument_code=SYMBOL, datapath=datapath, csv_config=BARCHART_CONFIG)

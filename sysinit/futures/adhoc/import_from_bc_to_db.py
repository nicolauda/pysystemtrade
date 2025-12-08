from sysdata.config.production_config import get_production_config
from syscore.fileutils import resolve_path_and_filename_for_package
from sysdata.csv.csv_futures_contract_prices import ConfigCsvFuturesPrices
from sysinit.futures.contract_prices_from_split_freq_csv_to_db import (
    init_db_with_split_freq_csv_prices_for_code,
)
from sysinit.futures.rollcalendars_from_db_prices_to_csv import (
    build_and_write_roll_calendar,
)
from sysinit.futures.multipleprices_from_db_prices_and_csv_calendars_to_db import (
    process_multiple_prices_single_instrument,
)
from sysinit.futures.adjustedprices_from_db_multiple_to_db import (
    process_adjusted_prices_single_instrument,
)

BARCHART_CONFIG = ConfigCsvFuturesPrices(
    input_date_index_name="Time",
    input_skiprows=0,
    input_skipfooter=0,
    input_date_format="ISO8601",
    input_column_mapping=dict(
        OPEN="Open", HIGH="High", LOW="Low", FINAL="Close", VOLUME="Volume"
    ),
)
instrument_code = input("Enter the futures symbol (e.g., 'BITCOIN'): ")
calendar_output_datapath = (
    "/home/algotrader/pst/pysystemtrade-private/private/futures/roll_calendars_csv"
)

# assuming bc-utils config pasted into private
datapath = resolve_path_and_filename_for_package(
    get_production_config().get_element_or_default("barchart_path", None)
)

# import prices for a single instrument
init_db_with_split_freq_csv_prices_for_code(
    instrument_code=instrument_code, datapath=datapath, csv_config=BARCHART_CONFIG
)

# build and write roll calendar for the instrument
build_and_write_roll_calendar(instrument_code, output_datapath=calendar_output_datapath)
_ = input(
    f"Roll calendar for {instrument_code} written to {calendar_output_datapath}. Press Enter to continue."
)
# process multiple prices for a single instrument
process_multiple_prices_single_instrument(instrument_code=instrument_code)

# process adjusted prices for a single instrument
process_adjusted_prices_single_instrument(instrument_code=instrument_code)

from pathlib import Path

from syscore.fileutils import resolve_path_and_filename_for_package
from sysdata.futures.instruments import futuresInstrumentData
from syscore.constants import arg_not_supplied
from sysobjects.instruments import (
    futuresInstrument,
    futuresInstrumentWithMetaData,
    instrumentMetaData,
    META_FIELD_LIST,
)
from syslogging.logger import *
import pandas as pd

INSTRUMENT_CONFIG_PATH = "data.futures.csvconfig"
CONFIG_FILE_NAME = "instrumentconfig.csv"


class csvFuturesInstrumentData(futuresInstrumentData):
    """
    Get data about instruments from a special configuration used for initialising the system

    """

    def __init__(
        self,
        datapath=arg_not_supplied,
        log=get_logger("csvFuturesInstrumentData"),
    ):
        super().__init__(log=log)

        # Prefer canonical repo location to avoid being shadowed by current working dir packages
        repo_config = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "futures"
            / "csvconfig"
            / CONFIG_FILE_NAME
        )

        if datapath is arg_not_supplied:
            resolved = repo_config
            if not resolved.exists():
                resolved = Path(
                    resolve_path_and_filename_for_package(
                        INSTRUMENT_CONFIG_PATH, CONFIG_FILE_NAME
                    )
                )
        else:
            resolved = Path(
                resolve_path_and_filename_for_package(datapath, CONFIG_FILE_NAME)
            )
            # If the requested path is missing but the canonical repo file exists, prefer it
            if (not resolved.exists()) and repo_config.exists():
                resolved = repo_config

        self._config_file = str(resolved)
        self._fallback_candidates = [
            str(path)
            for path in {repo_config, Path(self._config_file)}
            if path is not None
        ]

    def get_list_of_instruments(self) -> list:
        return list(self.get_all_instrument_data_as_df().index)

    def _get_instrument_data_without_checking(
        self, instrument_code: str
    ) -> futuresInstrumentWithMetaData:
        all_instrument_data = self.get_all_instrument_data_as_df()
        instrument_with_meta_data = get_instrument_with_meta_data_object(
            all_instrument_data, instrument_code
        )

        return instrument_with_meta_data

    def _delete_instrument_data_without_any_warning_be_careful(
        self, instrument_code: str
    ):
        raise NotImplementedError(
            "Can't overwrite part of .csv use write_all_instrument_data_from_df"
        )

    def _add_instrument_data_without_checking_for_existing_entry(
        self, instrument_object: futuresInstrumentWithMetaData
    ):
        raise NotImplementedError(
            "Can't overwrite part of .csv use write_all_instrument_data_from_df"
        )

    def write_all_instrument_data_from_df(self, instrument_data_as_df: pd.DataFrame):
        instrument_data_as_df.to_csv(
            self._config_file,
            index_label="Instrument",
            columns=[field for field in META_FIELD_LIST],
        )

    def get_all_instrument_data_as_df(self) -> pd.DataFrame:
        """
        Get configuration information as a dataframe

        :return: dict of config information
        """
        config_data = self._instrument_csv_as_df()
        return config_data

    def _instrument_csv_as_df(self) -> pd.DataFrame:
        config_data = getattr(self, "_instrument_df", None)
        if config_data is None:
            config_data = self._load_and_store_instrument_csv_as_df()

        return config_data

    def _load_and_store_instrument_csv_as_df(self) -> pd.DataFrame:
        candidate_paths = [self.config_file]
        # Include any precomputed fallbacks (canonical repo path etc.)
        for path in getattr(self, "_fallback_candidates", []):
            if path not in candidate_paths:
                candidate_paths.append(path)

        config_data = None
        errors = []
        for path in candidate_paths:
            try:
                config_data = pd.read_csv(path)
                if path != self.config_file:
                    self.log.warning(
                        "Instrument config %s missing; using fallback %s",
                        self.config_file,
                        path,
                    )
                    # cache the fallback so subsequent calls don't keep failing
                    self._config_file = path
                break
            except BaseException as exc:
                errors.append(f"{path} ({exc})")

        if config_data is None:
            raise Exception(
                "Can't read instrument config (tried %s)" % ", ".join(errors)
            )

        try:
            config_data.index = config_data.Instrument
            config_data.drop(labels="Instrument", axis=1, inplace=True)

        except BaseException:
            raise Exception("Badly configured file %s" % (self._config_file))

        self._config_data = config_data

        return config_data

    @property
    def config_file(self):
        return self._config_file

    def __repr__(self):
        return "Instruments data from %s" % self._config_file


def get_instrument_with_meta_data_object(
    all_instrument_data: pd.DataFrame, instrument_code: str
) -> futuresInstrumentWithMetaData:
    config_for_this_instrument = all_instrument_data.loc[instrument_code]
    config_items = all_instrument_data.columns

    meta_data_dict = get_meta_data_dict_for_instrument(
        config_for_this_instrument, config_items
    )

    instrument = futuresInstrument(instrument_code)
    meta_data = instrumentMetaData.from_dict(meta_data_dict)

    instrument_with_meta_data = futuresInstrumentWithMetaData(instrument, meta_data)

    return instrument_with_meta_data


def get_meta_data_dict_for_instrument(
    config_for_this_instrument: pd.DataFrame, config_items: list
):
    meta_data = dict(
        [
            (item_name, getattr(config_for_this_instrument, item_name))
            for item_name in config_items
        ]
    )

    return meta_data

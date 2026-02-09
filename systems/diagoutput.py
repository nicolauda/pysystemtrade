"""
Suite of functions to analyse a system, and produce configuration that can be saved to a yaml file
"""

import datetime
from typing import TYPE_CHECKING, Any

import yaml
import numpy as np
import pandas as pd

from syscore.constants import arg_not_supplied
from syscore.dateutils import ROOT_BDAYS_INYEAR
from syscore.interactive.progress_bar import progressBar
from systems.forecast_mapping import estimate_mapping_params
from sysquant.estimators.correlations import CorrelationList, correlationEstimate
from sysquant.estimators.cross_sectional_ic import CrossSectionalIC
from sysquant.estimators.forecast_persistence import ForecastPersistence
from systems.diagresults import (
    CrossSectionalICGridResult,
    ForecastPersistenceResult,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


class systemDiag(object):
    def __init__(self, system):
        self.system = system

    def instrument_list(self):
        return self.system.get_instrument_list()

    def trading_rules(self):
        return self.system.rules.trading_rules().keys()

    def target_forecast_value(self):
        return self.system.config.average_absolute_forecast

    def check_forecast_scaling(self):
        """
        Check forecast scaling

        Returns a list of tuples, ordered with largest error first

        :param system:
        :return: list of tuples
        """
        system = self.system
        instrument_list = system.get_instrument_list()
        rule_list = self.trading_rules()
        target_forecast_value = self.target_forecast_value()

        results_list = []
        for rule in rule_list:
            for instrument in instrument_list:
                forecast = system.forecastScaleCap.get_capped_forecast(instrument, rule)
                error = forecast_error(forecast, target_forecast_value)
                results_list.append((instrument, rule, error))

        sorted_by_max_error = sorted(results_list, key=lambda tup: tup[2], reverse=True)

        return sorted_by_max_error

    def check_combined_forecast_scaling(self, forecast_type="raw"):
        """
        Check combined forecast scaling

        Returns a list of tuples, ordered with largest error first

        :param system:
        :param forecast_type: raw or final. If raw is specified will be before any forecast scaling is applied
        :return: list of tuples
        """
        system = self.system
        instrument_list = system.get_instrument_list()
        target_forecast_value = self.target_forecast_value()

        try:
            attr_name_dict = dict(
                raw="_get_raw_combined_forecast", final="get_combined_forecast"
            )
            attr_name = attr_name_dict[forecast_type]
        except KeyError:
            raise Exception(
                "forecast_type must be one of %s" % str(attr_name_dict.keys())
            )

        try:
            forecast_func = getattr(system.combForecast, attr_name)
        except BaseException:
            raise Exception("%s not a method system.combForecast" % attr_name)

        results_list = []
        for instrument in instrument_list:
            forecast = forecast_func(instrument)

            error = forecast_error(forecast, target_forecast_value)
            results_list.append((instrument, error))

        sorted_by_max_error = sorted(results_list, key=lambda tup: tup[1], reverse=True)

        return sorted_by_max_error

    def forecast_mapping(self, target_position_at_avg_forecast=2.0):
        """
        Fit threshold values for forecasts

        :return: dict, suitable for dropping into a config object or yaml file
        """
        system = self.system
        instrument_list = self.instrument_list()
        avg_forecast = self.target_forecast_value()

        forecast_mapping = {}
        for instrument in instrument_list:
            position = system.portfolio.get_notional_position(instrument)
            forecast = system.combForecast.get_combined_forecast(instrument)
            scalar = position / forecast
            scalar_ewma = scalar.ewm(500).mean()
            position_at_avg_forecast = avg_forecast * scalar_ewma.values[-1]

            if np.isnan(
                position_at_avg_forecast
            ):  # In case no position was open for a given instrument
                position_at_avg_forecast = 0.0
                a_param = 0.0
            else:
                a_param = target_position_at_avg_forecast / position_at_avg_forecast

            print("%s avg position %.2f" % (instrument, position_at_avg_forecast))

            if a_param < 1.2:
                # no need to do anything
                print("Forecast scaling not required for %s" % instrument)
            elif a_param > 1.7:
                print(
                    "Warning! Position at avg forecast of %.2f is too small for mapping to work for %s "
                    % (position_at_avg_forecast, instrument)
                )
            else:
                (
                    a_param,
                    b_param,
                    threshold_value,
                    capped_value,
                ) = estimate_mapping_params(a_param)
                map_dict = dict(
                    a_param=float(a_param),
                    b_param=float(b_param),
                    threshold=float(threshold_value),
                )
                forecast_mapping[instrument] = map_dict

        return forecast_mapping

    def forecast_scalars(self):
        """
        Returns final estimated values for forecast scalars, so they can be written into a config as fixed values

        :return: dict of forecast scalars
        """

        system = self.system
        instrument_list = self.instrument_list()
        rule_list = self.trading_rules()

        use_estimates = system.config.use_forecast_scale_estimates
        if not use_estimates:
            print("Can't output forecast scalar estimates, as they weren't estimated")

        pooling = system.config.forecast_scalar_estimate["pool_instruments"]
        if not pooling:
            print(
                "WARNING: No way of putting different forecast scalars for different instruments into config"
            )
        scalar_results = dict()

        for rule in rule_list:
            if not pooling:
                scalar_results[rule] = dict()

            for instrument in instrument_list:
                scalar = float(
                    system.forecastScaleCap.get_forecast_scalar(instrument, rule).iloc[
                        -1
                    ]
                )
                if pooling:
                    # will be overwritten for each instrument
                    scalar_results[rule] = scalar
                else:
                    scalar_results[rule][instrument] = scalar

        return scalar_results

    def forecast_div_multiplier(self):
        """
        Returns final estimated values for FDM, so they can be written into a config as fixed values

        :return: dict
        """

        system = self.system
        instrument_list = self.instrument_list()
        fdm_results = dict()
        for instrument in instrument_list:
            fdm = system.combForecast.get_forecast_diversification_multiplier(
                instrument
            ).values[-1]
            fdm_results[instrument] = float(fdm)

        return fdm_results

    def forecast_weights(self):
        """
        Returns final estimated values for forecast weights, so they can be written into a config as fixed values

        :return: dict of dicts
        """
        # forecast weights
        system = self.system
        instrument_list = self.instrument_list()
        forecast_weights = dict()
        for instrument in instrument_list:
            weights = dict(
                system.combForecast.get_forecast_weights(instrument).iloc[-1]
            )
            weights = dict(
                (str(rule_name), float(weight)) for rule_name, weight in weights.items()
            )
            forecast_weights[instrument] = weights

        return forecast_weights

    def instrument_weights(self):
        """
        Returns final estimated values for instrument weights, so they can be written into a config as fixed values

        :return: dict
        """
        system = self.system
        instrument_weights = system.portfolio.get_instrument_weights().iloc[-1]
        instrument_weights = dict(
            (str(key), float(value)) for key, value in instrument_weights.items()
        )

        return instrument_weights

    def instrument_div_multiplier(self):
        """
        Returns final estimated values for instrument diversification multiplier, so it can be written into a config as fixed values

        :return: dict
        """
        system = self.system
        instrument_div_multiplier = float(
            system.portfolio.get_instrument_diversification_multiplier().values[-1]
        )

        return instrument_div_multiplier

    def get_forecast_correlation_matrix_before_date(
        self,
        relevant_date: datetime.datetime | object = arg_not_supplied,
        instrument_list: list[str] | object = arg_not_supplied,
        rules_list: list[str] | object = arg_not_supplied,
        strict: bool = False,
    ) -> correlationEstimate:
        """Compute forecast-rule correlations before a date.

        Args:
            relevant_date: Reference datetime. If `arg_not_supplied`, use the
                latest available matrix.
            instrument_list: Optional instruments used to estimate pooled
                forecast correlations. If `arg_not_supplied`, all system
                instruments are used.
            rules_list: Optional rule names to include in the matrix. If
                `arg_not_supplied`, all available rules are kept.
            strict: If `True`, raise when requested rules are missing. If
                `False`, missing rules are ignored.

        Returns:
            A `correlationEstimate` for the selected date and rules.
        """
        system = self.system
        if instrument_list is arg_not_supplied:
            instrument_list = system.get_instrument_list()

        corr_list: CorrelationList = (
            system.combForecast.get_forecast_correlation_matrices_from_instrument_code_list(
                instrument_list
            )
        )
        corr: correlationEstimate = corr_list.most_recent_correlation_before_date_for_rules(
            relevant_date=relevant_date,
            rules_list=rules_list,
            strict=strict,
        )

        return corr

    def get_forecast_correlation_matrix_df_before_date(
        self,
        relevant_date: datetime.datetime | object = arg_not_supplied,
        instrument_list: list[str] | object = arg_not_supplied,
        rules_list: list[str] | object = arg_not_supplied,
        strict: bool = False,
    ) -> pd.DataFrame:
        """Return forecast-rule correlations as a DataFrame.

        Args:
            relevant_date: Reference datetime. If `arg_not_supplied`, use the
                latest available matrix.
            instrument_list: Optional instruments used to estimate pooled
                forecast correlations. If `arg_not_supplied`, all system
                instruments are used.
            rules_list: Optional rule names to include in the matrix. If
                `arg_not_supplied`, all available rules are kept.
            strict: If `True`, raise when requested rules are missing. If
                `False`, missing rules are ignored.

        Returns:
            A square `pd.DataFrame` indexed and columned by rule name.
        """
        corr: correlationEstimate = self.get_forecast_correlation_matrix_before_date(
            relevant_date=relevant_date,
            instrument_list=instrument_list,
            rules_list=rules_list,
            strict=strict,
        )

        return corr.as_pd()

    def display_forecast_correlation_matrix_before_date(
        self,
        relevant_date: datetime.datetime | object = arg_not_supplied,
        instrument_list: list[str] | object = arg_not_supplied,
        rules_list: list[str] | object = arg_not_supplied,
        strict: bool = False,
        **display_kwargs: Any,
    ) -> tuple["Figure", "Axes"]:
        """Display forecast-rule correlations as a heatmap.

        Args:
            relevant_date: Reference datetime. If `arg_not_supplied`, use the
                latest available matrix.
            instrument_list: Optional instruments used to estimate pooled
                forecast correlations. If `arg_not_supplied`, all system
                instruments are used.
            rules_list: Optional rule names to include in the matrix. If
                `arg_not_supplied`, all available rules are kept.
            strict: If `True`, raise when requested rules are missing. If
                `False`, missing rules are ignored.
            **display_kwargs: Extra keyword arguments forwarded to
                `correlationEstimate.display(...)`.

        Returns:
            A tuple `(fig, ax)` with the created Matplotlib figure and axes.
        """
        corr: correlationEstimate = self.get_forecast_correlation_matrix_before_date(
            relevant_date=relevant_date,
            instrument_list=instrument_list,
            rules_list=rules_list,
            strict=strict,
        )

        return corr.display(**display_kwargs)

    def get_cross_sectional_ic(
        self,
        horizon_min: int,
        horizon_max: int,
        scale_by_sqrt: bool = True,
        show_progress: bool = True,
        spearman: bool = False,
    ) -> CrossSectionalICGridResult:
        """Compute cross-sectional IC grids across horizons and rules.

        Args:
            system: System instance providing forecasts and raw returns.
            horizon_min: Smallest horizon in days (inclusive).
            horizon_max: Largest horizon in days (inclusive).
            scale_by_sqrt: If True, divide each horizon sum by sqrt(h).
            show_progress: If True, display a progress bar.
            spearman: if True, IC is calculated with Spearman method

        Returns:
            Tuple of (rules_list, IC). `rules_list` is the
            ordered list of rule names used. The IC dicts are keyed by horizon
            (int) then rule name, with Series values indexed by date.

        Notes:
            Forecasts are grouped by rule, forward returns are computed per
            horizon, and `cross_sectional_ic` is applied for each rule/horizon
            pair. If a rule/horizon has no overlap in columns or dates, the
            stored Series can be empty.
        """
        system = self.system
        forecast_df_by_rule = system.combForecast.get_forecast_df_by_rule()
        forward_returns_by_horizon = system.rawdata.get_forward_returns_by_horizon(
            horizon_min=horizon_min,
            horizon_max=horizon_max,
            scale_by_sqrt=scale_by_sqrt,
        )

        rules_list = list(forecast_df_by_rule.keys())
        ic: dict[int, dict[str, pd.Series]] = {}

        progress = None
        total_iterations = len(forward_returns_by_horizon) * len(rules_list)
        if show_progress and total_iterations > 0:
            progress = progressBar(
                total_iterations, "Cross-sectional IC", show_each_time=True
            )

        for h, forward_ret_df in forward_returns_by_horizon.items():
            ic[h] = {}
            for rule in rules_list:
                information_coeff = CrossSectionalIC.cross_sectional_ic(
                    forecast_df=forecast_df_by_rule[rule],
                    returns_df=forward_ret_df,
                    spearman=spearman,
                )
                ic[h][rule] = information_coeff
                if progress is not None:
                    progress.iterate()
        return CrossSectionalICGridResult.from_diagoutput(
            rules_list=rules_list,
            ic_by_horizon=ic,
            horizon_min=horizon_min,
            horizon_max=horizon_max,
            scale_by_sqrt=scale_by_sqrt,
            show_progress=show_progress,
            spearman=spearman,
        )

    def get_persistence_function(
        self,
        horizon: int = 60,
        agg: str = "median",
        min_obs: int = 30,
        spearman: bool = False,
        half_life_level: float = 0.5,
        half_life_frac_of_rho1: float = 0.5,
        show_progress: bool = True,
    ) -> ForecastPersistenceResult:
        """
        Compute persistence curves and half-life summaries for multiple rules.

        Parameters
        ----------
        forecasts_by_rule : dict[str, pd.DataFrame]
            Mapping rule_name -> forecast DataFrame (T x N),
            index = time, columns = instruments.

        horizon : int
            Maximum lag for persistence computation.

        agg : {"median", "mean"}
            Aggregation across instruments.

        min_obs : int
            Minimum number of finite forecast pairs required per instrument.

        spearman : bool
            If True, use Spearman correlation; otherwise Pearson.

        half_life_level : float
            Absolute threshold for half-life extraction from the persistence curve.

        half_life_frac_of_rho1 : float
            Fraction of rho(1) used for relative half-life.

        show_progress : bool
            Show progress bar

        Returns
        -------
        persistence_by_rule : dict[str, pd.Series]
            rule -> persistence curve indexed by lag k.

        summary_by_rule : dict[str, dict]
            rule -> summary statistics with keys:
                - "half_life_abs"
                - "half_life_rel"
                - "rho1"
                - "n_lags"
                - "method"
                - "agg"
        """
        system = self.system
        forecast_df_by_rule = system.combForecast.get_forecast_df_by_rule()
        persistence_by_rule: dict[str, pd.Series] = {}
        summary_by_rule: dict[str, dict] = {}

        method = "spearman" if spearman else "pearson"

        progress = None
        total_iterations = len(forecast_df_by_rule)
        if show_progress and total_iterations > 0:
            progress = progressBar(
                total_iterations, "Persistence function", show_each_time=True
            )

        for rule, df_forecast_for_rule in forecast_df_by_rule.items():
            if df_forecast_for_rule is None or df_forecast_for_rule.empty:
                persistence = pd.Series(dtype=float)
                hl_abs = np.nan
                hl_rel = np.nan
            else:
                (
                    persistence,
                    hl_abs,
                    hl_rel,
                ) = ForecastPersistence.get_persistence_function_for_rule(
                    df_forecast_for_rule=df_forecast_for_rule,
                    horizon=horizon,
                    agg=agg,
                    min_obs=min_obs,
                    spearman=spearman,
                    half_life_level=half_life_level,
                    half_life_frac_of_rho1=half_life_frac_of_rho1,
                )
            persistence_by_rule[rule] = persistence

            # rho(1): prefer lag=1 if present, else first available value
            if persistence is not None and not persistence.empty:
                if 1 in persistence.index:
                    rho1 = float(persistence.loc[1])
                else:
                    rho1 = float(persistence.iloc[0])
            else:
                rho1 = np.nan

            summary_by_rule[rule] = {
                "half_life_abs": float(hl_abs) if np.isfinite(hl_abs) else np.nan,
                "half_life_rel": float(hl_rel) if np.isfinite(hl_rel) else np.nan,
                "rho1": rho1 if np.isfinite(rho1) else np.nan,
                "n_lags": int(len(persistence)) if persistence is not None else 0,
                "method": method,
                "agg": agg,
            }
            if progress is not None:
                progress.iterate()
        return ForecastPersistenceResult.from_diagoutput(
            persistence_by_rule=persistence_by_rule,
            summary_by_rule=summary_by_rule,
            horizon=horizon,
            agg=agg,
            min_obs=min_obs,
            spearman=spearman,
            half_life_level=half_life_level,
            half_life_frac_of_rho1=half_life_frac_of_rho1,
            show_progress=show_progress,
        )

    def output_config_with_estimated_parameters(
        self,
        attr_names=[
            "forecast_scalars",
            "forecast_weights",
            "forecast_div_multiplier",
            "forecast_mapping",
            "instrument_weights",
            "instrument_div_multiplier",
        ],
    ):
        output_dict = {}
        for config_item in attr_names:
            dict_function = getattr(self, config_item)
            try:
                dict_value = dict_function()
                output_dict[config_item] = dict_value
            except BaseException:
                print("Couldn't get %s will exclude from output" % config_item)

        return output_dict

    def yaml_config_with_estimated_parameters(
        self,
        yaml_filename,
        attr_names=[
            "forecast_scalars",
            "forecast_weights",
            "forecast_div_multiplier",
            "forecast_mapping",
            "instrument_weights",
            "instrument_div_multiplier",
        ],
    ):
        output_dict = self.output_config_with_estimated_parameters(
            attr_names=attr_names
        )
        with open(yaml_filename, "w") as f:
            yaml.dump(output_dict, f, default_flow_style=False)

    def calculation_details(self, instrument_code):
        """
        Explain how the position is calculated for a given instrument
        :return: pd.Series
        """

        system = self.system
        attributes_last_ts = [
            "combForecast.get_combined_forecast",
            "rawdata.daily_denominator_price",
            "rawdata.daily_returns_volatility",
            "positionSize.get_price_volatility",
            "positionSize.get_block_value",
            "positionSize.get_instrument_currency_vol",
            "positionSize.get_fx_rate",
            "positionSize.get_instrument_value_vol",
            "positionSize.get_average_position_at_subsystem_level",
            "positionSize.get_subsystem_position",
            "portfolio.get_notional_position",
        ]
        attributes_names = [
            "Fcast",
            "Price",
            "S(P_d)",
            "S(%daily)",
            "Blck val",
            "ICV",
            "FX",
            "IVV",
            "Vol scalar",
            "SS Pos",
            "Pos.",
        ]

        results = dict()
        for attribute, name in zip(attributes_last_ts, attributes_names):
            stage, method = attribute.split(".")
            stage_object = getattr(system, stage)
            stage_method = getattr(stage_object, method)
            result = stage_method(instrument_code).ffill().iloc[-1]
            results[name] = result

        attributes_dict = ["portfolio.get_instrument_weights"]
        attributes_names = ["Instr.Wt"]
        for attribute, name in zip(attributes_dict, attributes_names):
            stage, method = attribute.split(".")
            stage_object = getattr(system, stage)
            stage_method = getattr(stage_object, method)
            result_dict = stage_method()
            result = result_dict[instrument_code].ffill().iloc[-1]
            results[name] = result

        attributes_all = ["portfolio.get_instrument_diversification_multiplier"]
        attributes_names = ["IDM"]
        for attribute, name in zip(attributes_all, attributes_names):
            stage, method = attribute.split(".")
            stage_object = getattr(system, stage)
            stage_method = getattr(stage_object, method)
            result = stage_method().ffill().iloc[-1]
            results[name] = result

        attributes_scalar = ["data.get_value_of_block_price_move"]
        attributes_names = ["Blc size"]
        for attribute, name in zip(attributes_scalar, attributes_names):
            stage, method = attribute.split(".")
            stage_object = getattr(system, stage)
            stage_method = getattr(stage_object, method)
            result = stage_method(instrument_code)
            results[name] = result

        results["Daily VolTgt"] = system.positionSize.get_vol_target_dict()[
            "daily_cash_vol_target"
        ]

        buffers = system.portfolio.get_buffers_for_position(instrument_code).iloc[-1]
        results["Bfr+"], results["Bfr-"] = buffers.values

        return results

    def explain_calculator_for_code(self, instrument_code):
        results = self.calculation_details(instrument_code)
        explainers = [
            "Position  = Subsystem position * Instrument weight * IDM = %.2f * %.4f * %.2f = %.1f"
            % (results["SS Pos"], results["Instr.Wt"], results["IDM"], results["Pos."]),
            "Subsystem position = Combined forecast * Vol scalar / 10 = %.2f * %.2f / 10.0 = %.2f"
            % (results["Fcast"], results["Vol scalar"], results["SS Pos"]),
            "Vol scalar = Daily cash vol target / Instrument value vol = %.1f / %.1f = %.2f"
            % (results["Daily VolTgt"], results["IVV"], results["Vol scalar"]),
            "Instrument Value Vol = Instrument currency vol * FX rate = %.2f * %.6f = %.2f"
            % (results["ICV"], results["FX"], results["IVV"]),
            "Instrument currency vol = Block value * Daily %% Price vol = %.2f * %.4f = %.2f"
            % (results["Blck val"], results["S(%daily)"], results["ICV"]),
            "Daily %% Price vol = 100* Return difference vol / Price = %.6f / %.6f = %.4f (%.2f%% per year)"
            % (
                results["S(P_d)"],
                results["Price"],
                results["S(%daily)"],
                results["S(%daily)"] * ROOT_BDAYS_INYEAR,
            ),
            "Block value = Price * Block size * 0.01 = %.6f * %.1f * 0.01 = %.2f"
            % (results["Price"], results["Blc size"], results["Blck val"]),
            "OR Instrument currency vol = Return difference vol * Block size = %.6f * %.1f = %.2f"
            % (results["S(P_d)"], results["Blc size"], results["ICV"]),
        ]
        return explainers


def forecast_error(forecast, target_forecast_value):
    abs_size = forecast.abs().mean()
    std_size = forecast.std()
    abs_error = abs(abs_size - target_forecast_value)
    std_error = abs(std_size - target_forecast_value)
    max_error = max([abs_error, std_error])

    return max_error

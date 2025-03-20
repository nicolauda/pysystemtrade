# demo_system.py

from sysdata.config.configdata import Config
from systems.provided.futures_chapter15.estimatedsystem import futures_system
from matplotlib.pyplot import show
from sysdata.sim.csv_futures_sim_data import csvFuturesSimData
from sysbrokers.IB.ib_connection import connectionIB
from sysdata.data_blob import dataBlob
from sysobjects.contracts import futuresContract
from sysbrokers.IB.ib_futures_contract_price_data import ibFuturesContractPriceData
from sysbrokers.IB.ib_futures_contract_price_data import ibFuturesInstrumentData

from ib_insync import util
util.startLoop() #only required when running inside a notebook

my_config=Config("private.001_demo.config.yaml")
my_data = csvFuturesSimData()
# data confuguration in the link below
# https://github.com/robcarver17/pysystemtrade/discussions/1442

system = futures_system(config=my_config, data=my_data)
# System stages are listed below as reference
# RawData(),
# Rules(trading_rules)
# ForecastScaleCap(),
# ForecastCombine(),
# PositionSizing(),
# Portfolios(),
# Account(),

system.portfolio.get_notional_position("MXP")
system.portfolio.get_notional_position("V2X").curve().plot()
system.portfolio.get_notional_position("KR3")

system.accounts.portfolio().stats() ## see some statistics
system.accounts.portfolio().curve().plot()
system.portfolio.get_notional_position("NASDAQ_micro").plot()

system.accounts.portfolio().plot()

system.accounts.pandl_for_instrument_forecast("KR3", "ewmac64_256").sharpe() ## Sharpe for a specific trading rule variation
system.accounts.methods()
system.accounts.pandl_for_subsystem("MXP").percent.curve().plot()
system.accounts.pandl_for_subsystem("MXP").sharpe() ## Sharpe for a specific trading rule variation
system.accounts.pandl_for_subsystem("V2X").percent.curve().plot()
system.accounts.pandl_for_subsystem("V2X").sharpe() ## Sharpe for a specific trading rule variation
system.accounts.pandl_for_subsystem("KR3").percent.curve().plot()
system.accounts.pandl_for_subsystem("KR3").sharpe() ## Sharpe for a specific trading rule variation
system.accounts.pandl_for_trading_rule("carry").stats()
system.accounts.pandl_for_trading_rule("carry").percent.curve().plot()
system.accounts.pandl_for_trading_rule("carry").percent.curve().plot()




conn = connectionIB(101)
datablob = dataBlob()
ibfuturesdata = ibFuturesContractPriceData(conn, datablob)
datablob.broker_futures_instrument = ibFuturesInstrumentData(conn, data=datablob)

instrument_list=ibfuturesdata.get_list_of_instrument_codes_with_merged_price_data() # returns list of instruments defined in [futures config file](/sysbrokers/IB/ibConfigFutures.csv)
ibfuturesdata.contract_dates_with_merged_price_data_for_instrument_code("SP500_micro") # returns list of contract dates
ibfuturesdata.get_merged_prices_for_instrument(instrument_code="SP500_micro") # returns OHLC price and volume data
for instrument in instrument_list:
    print(instrument)
    print("")

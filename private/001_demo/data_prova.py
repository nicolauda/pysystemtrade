# data_prova.py
from syscore.dateutils import Frequency
from sysobjects.contracts import futuresContract
from sysbrokers.IB.ib_futures_contract_price_data import ibFuturesContractPriceData
from sysdata.data_blob import dataBlob


from sysbrokers.IB.ib_connection import connectionIB
from sysbrokers.IB.ib_instruments_data import ibFuturesInstrumentData
from sysbrokers.IB.ib_futures_contracts_data import ibFuturesContractData
from sysproduction.data.broker import dataBroker

from ib_insync import util
util.startLoop() #only required when running inside a notebook
conn = connectionIB(999) # the first compulsory value is the client_id; the keyword args are the default values and can be omitted
# data = dataBlob()
databroker = dataBroker()
# data.add_class_object(ibFuturesInstrumentData)
# data.add_class_object(ibFuturesContractData)

ibfuturesdata = ibFuturesContractPriceData(conn, data=databroker.data)
a=ibfuturesdata.get_list_of_instrument_codes_with_merged_price_data() # returns list of instruments defined in [futures config file](/sysbrokers/IB/ibConfigFutures.csv)
b=ibfuturesdata.contract_dates_with_merged_price_data_for_instrument_code("SP500_micro") # returns list of contract dates
c=ibfuturesdata.get_prices_at_frequency_for_contract_object(futuresContract("SP500_micro", "202503"), frequency=Frequency.Day) # returns daily price and volume data
c.head()
# expiry_date = "202503"
# contract_name = "SP500_micro"
# dates = databroker.get_list_of_contract_dates_for_instrument_code(contract_name, allow_expired=True)

# contract = futuresContract(contract_name, expiry_date)
# a = databroker.get_list_of_instrument_codes_with_merged_price_data()
# a = databroker.get_prices_at_frequency_for_contract_object(
#     contract_object=contract,
#     frequency=Frequency.Day
#     )
# b = databroker.get_prices_at_frequency_for_potentially_expired_contract_object(
#     contract_object=contract,
#     frequency=Frequency.Day
#     )
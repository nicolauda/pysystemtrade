from sysdata.data_blob import dataBlob
from sysproduction.data.contracts import dataContracts
from sysproduction.data.prices import diagPrices


def list_sampled_contracts():
    """
    Lists all the sampled contracts for each instrument.
    """
    with dataBlob(log_name="list_sampled_contracts") as data:
        diag_prices = diagPrices(data)
        data_contracts = dataContracts(data)

        instrument_list = diag_prices.get_list_of_instruments_in_multiple_prices()

        print("--- Sampled Contracts ---")

        for instrument_code in instrument_list:
            try:
                sampled_contracts = data_contracts.get_all_sampled_contracts(
                    instrument_code
                )
                if sampled_contracts:
                    print(f"\n--- {instrument_code} ---")
                    for contract in sampled_contracts:
                        print(contract)
                else:
                    print(f"\n--- {instrument_code} ---")
                    print("No sampled contracts found.")
            except Exception as e:
                print(f"\nCould not retrieve contracts for {instrument_code}: {e}")


if __name__ == "__main__":
    list_sampled_contracts()

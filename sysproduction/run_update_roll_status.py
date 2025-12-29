from syscontrol.run_process import processToRun
from sysproduction.update_roll_status import updateRollStatus
from sysdata.data_blob import dataBlob


def run_update_roll_status():
    process_name = "run_update_roll_status"
    data = dataBlob(log_name=process_name)
    list_of_timer_names_and_functions = get_list_of_timer_functions_for_roll_update(
        data
    )
    roll_process = processToRun(process_name, data, list_of_timer_names_and_functions)
    roll_process.run_process()


def get_list_of_timer_functions_for_roll_update(data: dataBlob):
    roll_data = dataBlob(log_name="update_roll_status")
    roll_update_object = updateRollStatus(roll_data)

    list_of_timer_names_and_functions = [
        ("update_roll_status", roll_update_object),
    ]

    return list_of_timer_names_and_functions


if __name__ == "__main__":
    run_update_roll_status()

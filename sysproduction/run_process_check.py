from syscontrol.run_process import processToRun
from sysdata.data_blob import dataBlob
from sysproduction.data.control_process import dataControlProcess


def run_process_check():
    process_name = "run_process_check"
    data = dataBlob(log_name=process_name)
    list_of_timer_names_and_functions = (
        get_list_of_timer_functions_for_process_check()
    )
    process_check = processToRun(process_name, data, list_of_timer_names_and_functions)
    process_check.run_process()


def get_list_of_timer_functions_for_process_check():
    data_process_control = dataBlob(log_name="process_check")
    process_control = dataControlProcess(data_process_control)

    list_of_timer_names_and_functions = [
        ("check_if_pid_running_and_if_not_finish_all_processes", process_control),
    ]

    return list_of_timer_names_and_functions


if __name__ == "__main__":
    run_process_check()

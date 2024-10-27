import os
import signal
import subprocess
import time

terminate_flag = False


def signal_handler(signum, frame):
    global terminate_flag
    terminate_flag = True
    print("Termination signal received. Exiting...")


SOURCE_DIR = "./optimal_ab_postprocess3/"


def execute(disccount):

    print(f"start: {disccount=}")

    filename_before = f"optimal_ab_table_all_{disccount}.txt"
    filename_after = f"optimal_ab_table_all_{disccount + 1}.txt"
    filename_log = f"optimal_ab_table_all_{disccount}_outputlog.txt"
    if disccount < 35:
        command = f"./reopening optimal-ab-postprocess4 {SOURCE_DIR}/{filename_before} {SOURCE_DIR}/{filename_after} {SOURCE_DIR}/{filename_log}"
    else:
        command = f"./reopening optimal-ab-postprocess4 {SOURCE_DIR}/{filename_before} {SOURCE_DIR}/{filename_log}"

    process = subprocess.Popen(
        command.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
    )

    while process.poll() is None and not terminate_flag:
        time.sleep(0.1)

    if terminate_flag:
        process.kill()
        print(f"Killed subprocess: {process.pid=}, {command=}")

    assert os.path.exists(f"{SOURCE_DIR}/{filename_log}")


def main():

    signal.signal(signal.SIGINT, signal_handler)

    for disccount in range(4, 36):
        execute(disccount)


if __name__ == "__main__":
    main()

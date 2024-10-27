import os
import signal
import subprocess
import time

terminate_flag = False


def signal_handler(signum, frame):
    global terminate_flag
    terminate_flag = True
    print("Termination signal received. Exiting...")


SOURCE_DIR = "./postprocess2/"
DEST_DIR = "./postprocess3/"


def execute(disccount):

    print(f"start: {disccount=}")

    os.makedirs(DEST_DIR, exist_ok=True)

    filename = f"optimal_reopening_ab_table_all_{disccount}.txt"
    command = f"./reopening postprocess3 {SOURCE_DIR}/{filename} {DEST_DIR}/{filename}"
    process = subprocess.Popen(
        command.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
    )

    while process.poll() is None and not terminate_flag:
        time.sleep(0.1)

    if terminate_flag:
        process.kill()
        print(f"Killed subprocess: {process.pid=}, {command=}")


def main():

    signal.signal(signal.SIGINT, signal_handler)

    for disccount in range(4, 36):
        execute(disccount)


if __name__ == "__main__":
    main()

import bz2
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time

global_count = 0
lock = threading.Lock()
job_queue = queue.Queue()
len_commands = 0
terminate_flag = False
OUTPUT_DIR = "./optimal_ab_endgames/"
OUTPUT_FILENAME_PREFIX = "optimal_ab_transposition_table"


def signal_handler(signum, frame):
    global terminate_flag
    terminate_flag = True
    print("Termination signal received. Exiting...")


TMP_DIR = "./optimal_ab_tmp_postprocess1/"
DEST_DIR = "./optimal_ab_postprocess1/"


def execute(n1n2):

    global global_count, lock

    with lock:
        global_count += 1
        print(f"start: {global_count} / {len_commands}, {n1n2=}")

    if n1n2 == "opening":
        tmp_dir = f"./"
        dest_dir = f"{DEST_DIR}/opening"
        unique_prefix = "opening"
        with lock:
            os.makedirs(dest_dir, exist_ok=True)
    else:
        n1, n2 = n1n2
        assert 0 <= n1 < 100
        assert 0 <= n2 < 100
        tmp_dir = f"{TMP_DIR}/{n1:02}/{n2:02}"
        bz2_files_dir = f"{OUTPUT_DIR}/{n1:02}/{n2:02}"
        dest_dir = f"{DEST_DIR}/{n1:02}/{n2:02}"
        unique_prefix = f"{n1:02}{n2:02}"
        if not os.path.exists(bz2_files_dir):
            return
        with lock:
            os.makedirs(tmp_dir, exist_ok=True)
            os.makedirs(dest_dir, exist_ok=True)
        for filename in os.listdir(bz2_files_dir):
            if not re.fullmatch(
                OUTPUT_FILENAME_PREFIX + r"[-OX]{64}\.csv.bz2", filename
            ):
                continue
            with bz2.open(os.path.join(bz2_files_dir, filename), "rb") as f_in:
                data = f_in.read()
            with open(os.path.join(tmp_dir, filename[:-4]), "wb") as f_out:
                f_out.write(data)

    command = f"./reopening optimal-ab-postprocess1 {tmp_dir} {dest_dir} {unique_prefix}"
    process = subprocess.Popen(
        command.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
    )

    while process.poll() is None and not terminate_flag:
        time.sleep(0.01)

    if terminate_flag:
        process.kill()
        print(f"Killed subprocess: {process.pid=}, {command=}")
    if tmp_dir != "./":
        shutil.rmtree(tmp_dir)


def worker_function():
    while not terminate_flag:
        job = job_queue.get()
        if job is None:
            break
        execute(job)
        job_queue.task_done()


def main():

    commands = [(n1, n2) for n1 in range(100) for n2 in range(100)]

    global len_commands, job_queue
    len_commands = len(commands) + 1
    job_queue.put("opening")
    for command in commands:
        job_queue.put(command)
    for _ in range(os.cpu_count()):
        job_queue.put(None)

    signal.signal(signal.SIGINT, signal_handler)

    threads = []
    for _ in range(os.cpu_count()):
        thread = threading.Thread(target=worker_function)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)


if __name__ == "__main__":
    main()

import bz2
import hashlib
import os
import queue
import random
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
OUTPUT_DIR = "./endgames/"
OUTPUT_FILENAME_PREFIX = "optimal_reopening_ab_transposition_table"


def signal_handler(signum, frame):
    global terminate_flag
    terminate_flag = True
    print("Termination signal received. Exiting...")


def execute(command: str):
    global global_count, lock
    with lock:
        global_count += 1
        print(f"start: {global_count} / {len_commands}, {command=}")

    process = subprocess.Popen(
        command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    while process.poll() is None and not terminate_flag:
        time.sleep(0.01)

    if terminate_flag:
        process.kill()
        print(f"Killed subprocess: {process.pid=}, {command=}")

    stderr_output = process.stderr.read()
    if len(stderr_output) > 0:
        print(f"{stderr_output=}")

    output_filename = f"{OUTPUT_FILENAME_PREFIX}{command.split(' ')[-2]}.csv"

    with open(output_filename, "rb") as f_in:
        data = f_in.read()
    with bz2.open(output_filename + ".bz2", "wb") as f_out:
        f_out.write(data)

    os.remove(output_filename)
    output_filename += ".bz2"
    sha256_hash_int = int(hashlib.sha256(output_filename.encode()).hexdigest(), 16)
    output_directory = (
        f"{OUTPUT_DIR}/{(sha256_hash_int%100):02}/{((sha256_hash_int//100)%100):02}"
    )
    destination_path = os.path.join(output_directory, output_filename)
    with lock:
        os.makedirs(output_directory, exist_ok=True)
    shutil.move(output_filename, destination_path)


def worker_function():
    while not terminate_flag:
        job = job_queue.get()
        if job is None:
            break
        execute(job)
        job_queue.task_done()


def compute_all_18_end():

    lines_dict = {}
    for filename in os.listdir("."):
        r = re.fullmatch(OUTPUT_FILENAME_PREFIX + r"([-OX]{64})\.csv", filename)
        if r is None:
            continue
        obf64 = r.group(1)
        if obf64.count("O") + obf64.count("X") != 10:
            continue

        with open(filename, "r") as f:
            firstline = f.readline().strip()
            assert firstline == "obf,disccount,nodekindcode,lowerbound,upperbound"
            for line in f:
                line = line.strip()
                columns = line.split(",")
                if columns[1] != "18":
                    continue
                filename_candidate = f"{OUTPUT_FILENAME_PREFIX}{columns[0][:64]}.csv"
                sha256_hash_int = int(
                    hashlib.sha256(filename_candidate.encode()).hexdigest(), 16
                )
                output_directory = f"{OUTPUT_DIR}/{(sha256_hash_int%100):02}/{((sha256_hash_int//100)%100):02}"
                destination_path = os.path.join(output_directory, filename_candidate)
                if os.path.exists(destination_path):
                    continue
                if columns[0] not in lines_dict:
                    lines_dict[columns[0]] = (
                        int(columns[2]),
                        int(columns[3]),
                        int(columns[4]),
                    )  # nodekindcode, lowerbound, upperbound
                else:
                    nodekindcode, lowerbound, upperbound = lines_dict[columns[0]]
                    nodekindcode |= int(columns[2])
                    lowerbound = max(lowerbound, int(columns[3]))
                    upperbound = min(upperbound, int(columns[4]))
                    assert lowerbound <= upperbound
                    lines_dict[columns[0]] = (nodekindcode, lowerbound, upperbound)

    # 難易度が最大のものでも実行時間はたかだか数分なので、シャッフルして実行する
    commands = []
    for k, v in lines_dict.items():
        commands.append(f"./reopening reopening-ab {v[0]} {v[1]} {v[2]} {k}")
    commands.sort()  # for reproducibility
    random.seed(12345)
    random.shuffle(commands)

    global len_commands, job_queue
    len_commands = len(commands)
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


if __name__ == "__main__":
    compute_all_18_end()

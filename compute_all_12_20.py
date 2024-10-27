import os
import queue
import signal
import subprocess
import threading
import time

global_count = 0
lock = threading.Lock()
job_queue = queue.Queue()
len_commands = 0
terminate_flag = False


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


def worker_function():
    while not terminate_flag:
        job = job_queue.get()
        if job is None:
            break
        execute(job)
        job_queue.task_done()


def compute_all_12_20():
    commands = []
    with open(
        "optimal_reopening_ab_transposition_table---------------------------OX------XO---------------------------.csv",
        "r",
    ) as f:
        firstline = f.readline().strip()
        assert firstline == "obf,disccount,nodekindcode,lowerbound,upperbound"
        for line in f:
            line = line.strip()
            columns = line.split(",")
            if columns[1] != "11":
                continue
            filename_candidate = (
                f"optimal_reopening_ab_transposition_table{columns[0][:64]}.csv"
            )
            if os.path.exists(filename_candidate):
                continue
            command = f"./reopening reopening-ab {columns[2]} {columns[3]} {columns[4]} {columns[0]}"
            commands.append(command)

    # 難易度に大きな差があり、実行時間が数分～数時間かかるため、ソートして難しいものから実行する
    def keyfunc(x):
        nodekindcode = int(x.split(" ")[2])
        nodekindcode = (nodekindcode & (nodekindcode - 1)) ^ nodekindcode
        nodekindcode = 5 if nodekindcode == 2 else nodekindcode
        absminbounds = min(abs(int(x.split(" ")[3])), abs(int(x.split(" ")[4])))
        return nodekindcode, absminbounds

    commands.sort(key=keyfunc)

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
    compute_all_12_20()

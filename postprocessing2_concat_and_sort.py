import os
import shutil
import signal
import subprocess
import time

terminate_flag = False


def signal_handler(signum, frame):
    global terminate_flag
    terminate_flag = True
    print("Termination signal received. Exiting...")


SOURCE_ROOT_DIR = "./postprocess1/"
TMP_DIR = "./reopening_tmp/"
DEST_DIR = "./postprocess2/"


def execute(disccount):
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(f"{TMP_DIR}", exist_ok=True)

    print(f"start: {disccount=}")

    # SOURCE_ROOT_DIRの中のファイルのうち、f"optimal_reopening_ab_table_{disccount}"ではじまり".txt"で終わるファイルをディレクトリを再帰的に走査して列挙
    filenames = []
    for root, _, files in os.walk(SOURCE_ROOT_DIR):
        for file in files:
            if file.startswith(f"optimal_reopening_ab_table_{disccount}") and file.endswith(".txt"):
                filenames.append(os.path.join(root, file))
    
    print(f"{len(filenames)=}")

    # ファイルを全て結合したものをf"{TMP_DIR}/optimal_reopening_ab_table_all_{disccount}.txt"に書き込む
    output_tmp_filename = f"{TMP_DIR}/optimal_reopening_ab_table_all_{disccount}.txt.tmp"
    with open(output_tmp_filename, "w") as f_out:
        for filename in filenames:
            with open(filename, "r") as f_in:
                f_out.write(f_in.read())

    # output_filenameをlinuxのsortコマンドでソートする。一時ファイル置き場にはf"{TMP_DIR}/tmp/以下を指定する。並列実行を許す。
    os.makedirs(f"{TMP_DIR}/tmp/", exist_ok=True)
    os.makedirs(DEST_DIR, exist_ok=True)
    output_filename = f"{DEST_DIR}/optimal_reopening_ab_table_all_{disccount}.txt"
    command = f"sort -o {output_filename} --batch-size=32 --buffer-size=32G --parallel={os.cpu_count()} -u -T {TMP_DIR}/tmp/ {output_tmp_filename}"

    os.environ['LC_ALL'] = 'C'
    process = subprocess.Popen(
        command.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
    )

    while process.poll() is None and not terminate_flag:
        time.sleep(0.1)

    if terminate_flag:
        process.kill()
        print(f"Killed subprocess: {process.pid=}, {command=}")
        
        if os.path.exists(TMP_DIR):
            shutil.rmtree(TMP_DIR)
        if os.path.exists(DEST_DIR):
            shutil.rmtree(DEST_DIR)
        exit(1)

    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)


def main():

    signal.signal(signal.SIGINT, signal_handler)

    for disccount in range(4, 36):
        execute(disccount)


if __name__ == "__main__":
    main()

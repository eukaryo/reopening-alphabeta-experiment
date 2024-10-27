import subprocess


def solve():
    for i in range(4, 37):
        subprocess.run(["./enumerate", str(i), "f"])


if __name__ == '__main__':

    solve()

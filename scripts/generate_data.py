import subprocess
import sys


if __name__ == "__main__":
    subprocess.run([sys.executable, "generate_data.py", *sys.argv[1:]], check=True)
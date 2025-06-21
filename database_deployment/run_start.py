import subprocess
import logging

script_path = "start.sh"
logfile_path = "output.log"

# Set up logging: INFO level, output to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(logfile_path, mode='a'),
        logging.StreamHandler()
    ]
)

# Start the process
process = subprocess.Popen(
    ["sudo", "bash", script_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    universal_newlines=True
)

# Read output line-by-line and log it
for line in process.stdout:
    logging.info(line.rstrip())  # remove trailing newline because logger adds one

process.stdout.close()
return_code = process.wait()

logging.info(f"Process finished with return code {return_code}")

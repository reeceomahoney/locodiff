import argparse
import subprocess

# Parser
parser = argparse.ArgumentParser(description="Copy logs from remote server")
parser.add_argument(
    "-t", type=str, help="The time string in the format YYYY-MM-DD/HH-MM-SS"
)
parser.add_argument("-f", type=str, help="file")
args = parser.parse_args()

# Define the bash command
if args.f:
    cmd = f"scp wanming:/home/reece/ws/dev/locodiff/{args.f} {args.f}"
elif args.t:
    cmd = f"scp -r wanming:/home/reece/ws/dev/locodiff/{args.t}/ {args.t}/"

# Call the bash command
subprocess.run(cmd, shell=True)

import os
from dotenv import load_dotenv
import subprocess

# Load .env
load_dotenv()

# Start Label Studio
subprocess.run(["label-studio", "start"])

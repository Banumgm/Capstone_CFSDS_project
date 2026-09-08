import os
import subprocess
import sys
from pathlib import Path

scripts_dir = Path("scripts")

script_files = sorted([f for f in scripts_dir.glob("*.py") if f.name.endswith(".py")])

print(f"Scripts to run: {len(script_files)}\n")

for script in script_files:
    print(f"=" * 50)
    print(f"Run: {script.name}")
    print(f"=" * 50)
    
    result = subprocess.run([sys.executable, str(script)])
    
    if result.returncode != 0:
        print(f"\n Error {script.name}. Stopped.")
        sys.exit(result.returncode)

print("\n Success!")
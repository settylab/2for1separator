#!/usr/bin/env python

from sep241 import path
import sep241deconvolve as deconvolve
import inspect

sbatch_info = f"""#!/usr/bin/env python
#SBATCH --job-name=2for1separator
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00

import sys
sys.path.append(\"{path}\")
"""

script = inspect.getsource(deconvolve)
script = script[script.index('\n') + 1:]
script = sbatch_info + script

def main():
    print(script)

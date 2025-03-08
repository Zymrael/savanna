"""
Runs last failed tests under TRITON INTERPRETER mode.
"""

import os
from pathlib import Path

parent_dir = Path(os.getcwd()).parent 
cache_dir = parent_dir / '.pytest_cache/v/cache/'

# Get first commandline argument
test_file = os.sys.argv[1]
if os.path.exists(cache_dir):
    if os.path.exists(cache_dir / 'lastfailed'):
        print("Re-running failed tests under TRITON INTERPRETER mode...", flush=True)
        os.system(f'TRITON_INTERPRET=1 pytest -sv --lf {test_file}')
    else:
        print("No failed tests to run.")
else:
    print("No cache found. Run the tests first.")

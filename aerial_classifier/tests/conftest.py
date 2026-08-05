"""
Common test setup.

The project modules live at the package root and are imported by short name
(`import features`), the same way the project itself imports them. The root is
therefore added to sys.path — there is deliberately no separate package with
__init__.py here, so that `python main.py ...` works from the project folder
without installation.

The whole test suite requires neither the dataset, nor trained checkpoints, nor
torch: everything needed is generated in the tests themselves. This way they run
on any machine, not only where the 83 MB of BMP files live.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

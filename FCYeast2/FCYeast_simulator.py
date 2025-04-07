# This file is a "forwarding" module.
# It allows the FCYeast_simulator module to remain usable in this folder,
# even though the actual library (FCYeast_simulator.py) is in ../FCYeast/.


import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from FCYeast.FCYeast_simulator import *

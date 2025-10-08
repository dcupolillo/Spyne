"""
Spyne - A Python package for analyzing calcium imaging data from dendrites and spines.
"""
__version__ = "0.1.0"

from .core.imaging.imagingdataset import ImagingDataset
from .core.electrophysiology.ephydataset import EphyDataset
from .core.spines.segmenter import DatasetSegmenter

# Import plot submodule
from . import plot

# Import dataframe utility
from .core.spines.dataframe import to_dataframe

import sys
from pathlib import Path

def _info():
    """display information about the spyne package."""
    import platform
    import numpy
    print("==== Spyne Information ====\n")
    print("Python", sys.version)
    print("System:", platform.system(), platform.release())
    print("numpy version:", numpy.__version__)
    print("spyne version:", __version__)
    print("spyne path:" , Path(__file__).parent.resolve())
    print("\n")
    
def showInfo():
    _info()

def help():
    """launch the spyne project page in a browser."""
    import webbrowser
    webbrowser.open("http://github.com/dcupolillo/spyne/tree/stable", new=2)
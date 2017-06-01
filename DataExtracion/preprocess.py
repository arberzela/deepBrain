import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib as mpl

from wyrm import processing as proc
from wyrm.types import Data
from wyrm import plot
from wyrm.io import load_bcicomp3_ds1
plot.beautify()

b, a = proc.signal.butter(5, [13 / 500], btype='low')


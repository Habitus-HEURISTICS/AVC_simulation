import copy
import pandas as pd
import numpy as np
from numpy.random import default_rng
rng = default_rng(10) # stuff for random sampling

_path = "/Users/Allegra/Documents/Postdoc/habitus/modules/github/AVC_simulation_origin/"

# Read in DSSAT information ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

def get_harvest(pop):
	dssat = pd.read_csv(_path + "Data/DSSAT_example.csv")
	hdat = dssat['HDAT'] # These are test results where HDAT is the predicted DOY of harvest. Not time series.

	# When we figure out how to actually interface with DSSAT, we won't need this:
	average_hdat = np.mean(hdat)
	faked_hdat = np.array(rng.normal(average_hdat, 7, size = pop.size))
	pop.make_column('hdat', np.round(faked_hdat))
	return pop
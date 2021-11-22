import copy
import pandas as pd
import numpy as np
from numpy.random import default_rng
rng = default_rng(10) # stuff for random sampling
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_path = "/Users/Allegra/Documents/Postdoc/habitus/modules/github/AVC_simulation_origin/"
sys.path.insert(1, "/Users/Allegra/Documents/Postdoc/habitus/modules/github/AVC_simulation_origin/Framework/")
sys.path.insert(1, "/Users/Allegra/Documents/Postdoc/habitus/modules/github/AVC_simulation_origin/Models/Functions/")
from decision_classes import Decision, Influence, Experiment
from decision_utilities import ps, row_sums, col_sums, row_margins, col_margins, row_renorm, index2onehot, onehot2index, row_sample, ged, ternary

from global_pop import pop, season_param_rule, update_credit_rule

from crop_selection import *
from harvest_timing import *
from plotting_functions import * # Keep this commented out unless you're plotting because plotnine takes forever to load

import plotnine
from plotnine import *


# Params and columns that need updating
# season_param_rule.funrun() # Don't update season, we want hot/dry (initialized as hot/dry)
update_p_success_rule.funrun()
update_credit_rule.funrun()
evaluate_success_rule.funrun()
print("\n\nSeason updated. \n Probability of crop success updated. \n Actual success updated. \n Credit updated. \n")

# What you need for crop selection
crop_selection = hd if pop.season.val == 'hot/dry' else (cd if pop.season.val == 'cold/dry' else r)
crop_selection.update_mask()
crop_selection.apply_influences()
crops = crop_selection.decision(form = 'onehot')
pop.selected_crop.assign(crops)
pop.selected_crop_index.assign(onehot2index(pop.selected_crop.val))

print("Crops selected. \n")
print(pop.selected_crop_index.val, "\n\n")

plot_three_seasons(pd.DataFrame({'Time': 0, 'Season': pop.season.val, 'Region': pop.region.val, 'Crop': pop.selected_crop_index.val, 'Hectares': 1}), 0)
plot_map_discrete(pop, 'selected_crop_index', {0:'g', 1:'r', 2:'orange', 3:'b', 4:'purple'})


harvest_timing.apply_influences()
harvest_times = harvest_timing.decision()
pop.harvest_times.assign(harvest_times)
# print(harvest_timing.probs[0])
print("Harvest times selected. \n")
print(pop.harvest_times.val, "\n\n")

df1 = pd.DataFrame({'harvest_times': pop.harvest_times.val, 'hdat': DOY_to_bin(pop.hdat.val)})
bb, rr, hh, ll, ss = np.unique(np.where(birds.astype(bool))[1]), np.unique(np.where(rains.astype(bool))[1]), np.unique(np.where(harvesters.astype(bool))[1]), np.unique(np.where(labor.astype(bool))[1]), np.unique(np.where(season_bins.astype(bool))[1])

print((ggplot() #+ geom_histogram(df1, aes('hdat'), fill = 'green', bins = 52, alpha = 0.5)
	+ geom_histogram(df1, aes('harvest_times'), fill = 'orange', bins = 52, alpha = 0.5) 
		+ geom_point(mapping = aes(bb, y = [150]*len(bb)), color = 'red', size = 2.5)
		+ geom_point(mapping = aes(hh, y = [156]*len(hh)), color = 'dodgerblue', size = 2.5)
		+ geom_point(mapping = aes(ll, y = [159]*len(ll)), color = 'green', size = 2.5)
		+ geom_point(mapping = aes(rr, y = [153]*len(rr)), color = 'red', size = 2.5)
		+ geom_point(mapping = aes(ss, y = [142]*len(ss)), color = 'purple', size = 2.5)
		+ geom_line(mapping = aes(bb, y = [150]*len(bb)), color = 'red', size = 2.5)
		+ geom_line(mapping = aes(hh, y = [156]*len(hh)), color = 'dodgerblue', size = 2.5)
		+ geom_line(mapping = aes(ll, y = [159]*len(ll)), color = 'green', size = 2.5)
		+ geom_line(mapping = aes(rr, y = [153]*len(rr)), color = 'red', size = 2.5)
		+ geom_line(mapping = aes(ss, y = [142]*len(ss)), color = 'purple', size = 2.5)
		+ xlim(0,52) + labs(x = "Harvest week")
		+ theme_minimal()))












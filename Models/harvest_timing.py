import copy
import pandas as pd
import numpy as np
from numpy.random import default_rng
rng = default_rng() # stuff for random sampling
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_path = "/Users/Allegra/Documents/Postdoc/habitus/modules/github/AVC_simulation_origin/"
sys.path.insert(1, "/Users/Allegra/Documents/Postdoc/habitus/modules/github/AVC_simulation_origin/Framework/")
sys.path.insert(1, "/Users/Allegra/Documents/Postdoc/habitus/modules/github/AVC_simulation_origin/Models/Functions/")
from decision_classes import Decision, Influence, Experiment
from decision_utilities import ps, row_sums, col_sums, row_margins, col_margins, row_renorm, index2onehot, onehot2index, row_sample, ged, ternary

from global_pop import pop

from season_maintenance import *
from timing_functions import *


# Making up data -- all of this ought to be read in from some file that has information about the season ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# Making up data about availability and crop maturity -- everything is the same for every farmer
# Note: crop maturity should get passed to us from pipeline.py by a column in pop
#       and harvesters, labor, birds and rains should probably all be Params governed by Rules
season_bins = get_harvest_bins(pop.season.val, pop.size)

harvesters, labor, birds, rains = season_bins.copy(), season_bins.copy(), np.zeros_like(season_bins),  np.zeros_like(season_bins)
harvesters[:, 21:26] = [4,5,3,1,1]
harvesters[:, 26:] = 0
labor[:, 21:24] = 0 # Labor available the last week of the season plus two weeks
labor[:, 25:28] = [5,4,3] # Labor available the last week of the season plus two weeks

birds[24:27] = 1 # Make up when birds will arrive
rains[26:28] = 1 # Make up when rains will arrive

crop_maturity = np.zeros((pop.size, len(season_bins[0,:]))) 
crop_maturity[:, 20:28] = [1,2,6,8,8,6,2,1] # Making up data about crop maturity
initial_probs = row_renorm(crop_maturity) # Initial probability is crop maturity, but we might want to change this


# Making up data about cost and area
harvester_rate = 0.5 # Half a day per hectare for a harvester
harvester_cost = 0.18 # Percent of yield
labor_rate = 9 # 9 days for one laborer per hectare
labor_cost = 0.1 # Percent of yield
minimum_payment_h, minimum_payment_l = 0.75, 0.25 # Made this up!




# Step one: Ensure that you can't harvest during the times you don't have harvesters or labor ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

def get_init_probs():
    return initial_probs

def can_harvest():
    return np.logical_or(harvesters.astype(bool), labor.astype(bool))

def cant_harvest():
    return np.invert(can_harvest())

def set_small(**kwargs):
    probs = kwargs.get('probs')
    mask = kwargs.get('mask')
    probs[mask] = 0.01
    return probs

harvest_timing = Decision(
    name = 'when_to_harvest', 
    get_init_probs = get_init_probs,
    get_mask = cant_harvest,
    get_clamped_probs = set_small
    )



# Step two: Influence harvest timing based on duration, birds and rains ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 

# MISSING: Any rules about cost; any rules about marketplaces; humidity of paddy (labor is better for wet rice); whether you want to crop next season


def calculate_duration():
    rl = ((labor_rate * pop.land_area.val)/7)
    required_labor = np.tile(rl, (gfreq,1)).T
    l = (np.divide(required_labor, labor, where = labor != 0, out=np.zeros_like(labor))) # Duration given available labor (pretending that you can have all the labor that's available and that you keep the same labor from week to week)
    rh = (harvester_rate * pop.land_area.val)/7
    required_harvesters = np.tile(rh, (gfreq,1)).T
    h = (np.divide(required_harvesters, harvesters, where = harvesters != 0, out=np.zeros_like(harvesters))) # Duration given number of harvesters (not quite the same as you don't use two harvesters; proxy for availability)
    return [l,h]

def harvest_duration(**kwargs):
    d = calculate_duration()
    l, h = d[0], d[1]
    # Now we'd like to know whether, if you start harvesting at the index, what week you'll finish in; if it's past your cutoff date, assign it a high penalty
    end_l, end_h = l + np.tile(np.array(range(len(labor))), (gfreq,1)).T, h + np.tile(np.array(range(len(harvesters))), (gfreq,1)).T # Not masking here because you don't need to
    l[(end_l > len(labor)) & (labor.astype(bool))] = 10
    h[(end_h > len(harvesters)) & (harvesters.astype(bool))] = 10

    lscore = np.divide(1.0, l, where = l != 0, out=np.zeros_like(l)) # Shorter duration is better, apply a score of 1 / duration
    hscore = np.divide(1.0, h, where = h != 0, out=np.zeros_like(h))

    return row_renorm(lscore + hscore)

harvest_timing.make_influence(
    get_destinations = harvest_duration,
    rate = .5
)


# Hubert's code follows:

def avoid_birds_and_rains(**kwargs):
    critical_period = np.logical_or(birds, rains) # When birds come or it rains (High chance to lose my grains)
    first_week_bird_or_rain = np.where(critical_period==True)[0][0] # Choose the week birds or rains apear for the first time
    weeks_before_critical_period = np.zeros_like(season_bins)
    weeks_before_critical_period[:, :first_week_bird_or_rain] = 1 # I want to harvest in any week before birds or rains come
    return row_renorm(weeks_before_critical_period)

harvest_timing.make_influence(
    get_destinations = avoid_birds_and_rains,
    rate = .5
)






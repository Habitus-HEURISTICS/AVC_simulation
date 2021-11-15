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

from global_pop import pop, li_p_crop_hotdry, li_p_crop_colddry, li_p_crop_rainy

from season_maintenance import *
from success import *
from credit import *

rice_index = 0

def hdip(**kwargs): return np.tile(li_p_crop_hotdry, (pop.size,1))
def cdip(**kwargs): return np.tile(li_p_crop_colddry, (pop.size,1))
def rip(**kwargs): return np.tile(li_p_crop_rainy, (pop.size,1))


# Success rules ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
#  These aren't Decisions or Headings because they're hard-coded adjustments to the probability and actuality of success,
#    which is not something the farmer has control over in this specific model.
update_p_success_rule = pop.make_rule(name = 'update_p_success', fun = update_p_success, columns=[pop.p_crop_success], cohorts = [cohort for cohort in pop.dynamic_cohorts.values() if 'region' in cohort.name])
evaluate_success_rule = pop.make_rule(name = 'evaluate_success', fun = evaluate_success, columns=[pop.p_crop_success, pop.crop_success, pop.selected_crop])


# Success influences, with credit mask ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
def rice_credit(**kwargs):
	c = np.zeros_like(pop.selected_crop.val)
	c[:, rice_index] = pop.credit.val
	return c.astype(bool)

# Set rice probability to be very small if you don't have credit
def no_credit_rice_probs(**kwargs): 
	probs = kwargs.get('probs')
	mask = kwargs.get('mask')
	probs[mask] = .05
	return probs


hd = Decision(
    name = 'hot_dry', 
    get_init_probs = hdip,
    get_mask = rice_credit,
    get_clamped_probs = no_credit_rice_probs
    )

hd.make_influence(
    get_destinations = success_adjustment,
    rate = .1
)

cd = Decision(
    name = 'hot_dry', 
    get_init_probs = cdip,
    get_mask = rice_credit,
    get_clamped_probs = no_credit_rice_probs
    )

cd.make_influence(
    get_destinations = success_adjustment,
    rate = .1
)

r = Decision(
    name = 'hot_dry', 
    get_init_probs = rip,
    get_mask = rice_credit,
    get_clamped_probs = no_credit_rice_probs
    )

r.make_influence(
    get_destinations = success_adjustment,
    rate = .1
)



# Social rules (written by Sheng-Tai and Zhuoyu) ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ---------- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ---------- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ---------- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----


def follow_the_leadership (decision,**kwargs):
    new_probs = decision.probs.copy()
    for cohort in pop.dynamic_cohorts.values():
    	if 'coop' in cohort.name:
		    i = np.argmax(pop.leadership.val[cohort.members])
		    new_probs[cohort.members] = decision.probs[i]
    return new_probs


def follow_the_education (decision,**kwargs):
    new_probs = decision.probs.copy()
    for cohort in pop.dynamic_cohorts.values():
    	if 'coop' in cohort.name:
		    i = np.argmax(pop.education.val[cohort.members])
		    new_probs[cohort.members] = decision.probs[i]
    return new_probs

hd.make_influence(get_destinations = follow_the_leadership, rate = .1)
cd.make_influence(get_destinations = follow_the_leadership, rate = .1)
r.make_influence(get_destinations = follow_the_leadership, rate = .1)

hd.make_influence(get_destinations = follow_the_education, rate = .1)
cd.make_influence(get_destinations = follow_the_education, rate = .1)
r.make_influence(get_destinations = follow_the_education, rate = .1)























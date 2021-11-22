import copy
import pandas as pd
import numpy as np
from numpy.random import default_rng
rng = default_rng() # stuff for random sampling
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.neighbors import NearestNeighbors

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


def follow_the_neighbor (decision,**kwargs):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pop.xy.val)
    distances, indices = nbrs.kneighbors(pop.xy.val)
    i = indices[:, 1]
    # return the decision probabilities for the ith cohort member
    return decision.probs[i, :]


def follow_the_gender_leadership (cohort,decision,**kwargs):
    pop = decision.pop
    # Get all leaders
    leaders = np.where(pop.leader.val[cohort.members] == 1)
    # Get leaders based on gender
    male_leader = np.where((pop.leader.val[cohort.members] == 1) & (pop.gender.val[cohort.members] == "male"))
    female_leader = np.where((pop.leader.val[cohort.members] == 1) & (pop.gender.val[cohort.members] == "female"))
    
    # Get the nearest male leader
    m_nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pop.xy.val[male_leader], male_leader)
    _, indices = m_nbrs.kneighbors(pop.xy.val)
    indices[male_leader, 1] = indices[male_leader, 0]
    indices = indices[:, 1]
    
    # Get the nearest female leader
    f_nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pop.xy.val[female_leader], female_leader)
    _, f_indices = f_nbrs.kneighbors(pop.xy.val)
    f_indices[female_leader, 1] = f_indices[female_leader, 0]
    f_indices = f_indices[:, 1]
    
    # Combine the nearest leader to make men follow male leaders, and women follow female leaders
    indices[np.where(pop.gender.val[cohort.members] == "female")] = f_indices[np.where(pop.gender.val[cohort.members] == "female")]
    for k in range(len(leaders[0])):
        indices[np.where(indices == k)] = leaders[0][k]
    
    # return the decision probabilities for the ith cohort member
    return decision.probs[indices] 
    

hd.make_influence(get_destinations = follow_the_leadership, rate = .1)
cd.make_influence(get_destinations = follow_the_leadership, rate = .1)
r.make_influence(get_destinations = follow_the_leadership, rate = .1)

hd.make_influence(get_destinations = follow_the_education, rate = .1)
cd.make_influence(get_destinations = follow_the_education, rate = .1)
r.make_influence(get_destinations = follow_the_education, rate = .1)

hd.make_influence(get_destinations = follow_the_neighbor, rate = .1)
cd.make_influence(get_destinations = follow_the_neighbor, rate = .1)
r.make_influence(get_destinations = follow_the_neighbor, rate = .1)





















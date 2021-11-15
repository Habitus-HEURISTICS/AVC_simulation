#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday, Oct 20

@author: aabcohen
"""

# Functions for use in success Rules

import numpy as np
from numpy.random import default_rng
rng = default_rng() # stuff for random sampling

from decision_classes import Decision, Influence, Experiment
from decision_utilities import ps, row_sums, col_sums, row_margins, col_margins, row_renorm, index2onehot, onehot2index, row_sample, ged, ternary

from global_pop import pop, number_to_region

def success(row):
	return (rng.random(len(row)) < row).astype(int)

# IMPROVEMENT NOTES:
#    When we write rules that determine success probabilistically or empirically (e.g. SAED data, DSSAT runs, market price data).
#      we can invoke them here. For example, if we have feedback from DSSAT about the yield, we can combine with market price data
#      for all five regions in an "empirical" evaluation of success.
#    For now, I manually set the probability of success based on data from Dr. Fall about planting in each region and my
#      guesses about how successful each crop is based on interviews.
def update_p_success(pop, columns, cohorts, **kwargs):
	p_crop_success = columns.get('p_crop_success')

	if pop.season.val == 'hot/dry':
		seasonal_success = np.tile(np.array([0.85, 0, 0, 0.05, 0]), (pop.size,1))
	elif pop.season.val == 'rainy':
		seasonal_success = np.tile(np.array([0.7, 0, 0, 0.1, 0]), (pop.size,1))
	else: # Cold/dry
		seasonal_success = np.tile(np.array([0, 0.5, 0.5, 0.5, 0]), (pop.size,1))
	
	p_crop_success.assign(seasonal_success)
	
	# Make region-level adjustments where needed by hand, based on interviews
	for cohort in cohorts.values():

		rgn_name = number_to_region[int(cohort.name[-1])]

		if rgn_name == 'dagana' and pop.season.val == 'cold/dry': # We know Dagana has tomato processing plants, so increase the likelihood of tomatoes paying off
			p_crop_success.assign( np.tile(np.array([0, 0.75, 0.5, 0.25, 0]), (len(cohort.members),1)) , selected = cohort.members)

		elif rgn_name == 'podor' and pop.season.val == 'cold/dry': # We know Podor plants onions more often, so maybe there's something that makes it more successful
			p_crop_success.assign( np.tile(np.array([0, 0.25, 0.75, 0.5, 0]), (len(cohort.members),1)) , selected = cohort.members)

		elif rgn_name == 'matam' and pop.season.val == 'cold/dry': # Matam doesn't like tomatoes at all but does plant onions, so maybe the climate is better
			p_crop_success.assign( np.tile(np.array([0, 0.1, 0.75, 0.75, 0]), (len(cohort.members),1)) , selected = cohort.members)

		elif rgn_name == 'matam' and pop.season.val == 'rainy': # We might think that Matam has more rain in the rainy season, so increase success of rice
			p_crop_success.assign( np.tile(np.array([0.85, 0.0, 0.0, 0.1, 0]), (len(cohort.members),1)) , selected = cohort.members)





def evaluate_success(columns, cohorts, **kwargs):
	selected_crop = columns.get('selected_crop')
	crop_success = columns.get('crop_success')
	p_crop_success = columns.get('p_crop_success') # The probabilities of success for any crop

	planted_probs = p_crop_success.val * selected_crop.val # Get the probability of success for the crop you actually planted

	hypothetical_crop_success = np.apply_along_axis(success, 1, planted_probs) # Evaluate whether you succeeded given the probability of the crop you actually planted
	actual_crop_success = hypothetical_crop_success * selected_crop.val
	crop_success.assign(actual_crop_success)




def success_adjustment (**kwargs):
	pl = pop.selected_crop.val.copy() # The one you planted last time
	s = pop.crop_success.val.copy()

	ipl = np.invert(pl.astype(bool)).astype(int)
	good_rows = (s == pl).all(axis = 1)
	not_good_rows = np.invert(good_rows)
	s[np.invert(good_rows)] = ipl[np.invert(good_rows)]

	return row_renorm(s)





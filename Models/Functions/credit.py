#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday, Oct 20

@author: aabcohen
"""

import numpy as np
from numpy.random import default_rng
rng = default_rng() # stuff for random sampling

from decision_classes import Decision, Influence, Experiment
from decision_utilities import ps, row_sums, col_sums, row_margins, col_margins, row_renorm, index2onehot, onehot2index, row_sample, ged, ternary


# def update_best_customers(pop, cohorts, **kwargs):
# 	# 15 to 20% of the coops should be "best customers" and there should be some variation based on repayment rates. We don't
# 	#  have that implemented and there are only five coops so for now we'll pick one randomly, weighted by land area (assuming banks prefer bigger coops.)
# 	land_area = columns.get('land_area')
# 	coops = list(cohorts)
# 	land_areas = [np.sum(land_area.val[coop.members]) for coop in coops]
# 	best_customer_index = np.random.choice(range(len(land_area)), p = renormalize(np.array([land_areas]))) # This should be whatever based on default, not probabilistic
# 	best_customer = coops[best_customer_index]
# 	pop.best_customers.assign([best_customer])


# This is a bad way of doing it...
def update_credit(pop, columns, cohorts, params, **kwargs):
	credit = pop.columns.get('credit')
	credit_val = credit.val
	bank_credit = np.random.choice([0, 1], p = [0.2, 0.8]) # Let's just say for now that the bank gets it late or on time
	credit.assign(np.tile(bank_credit, (pop.size))) # Everyone gets what the bank gets
	# If the bank got it late, everyone gets it late

	# If the bank got it on time, some people (the non-best-customers) MAY still get it late
	#  Pick out rows with credit = 1 (that means bank got it on time) and replace with np.random.choice([0, 1])
	credit_val[credit_val == 1] = [np.random.choice([0, 1], p = [0.35, 0.65]) for c in list(credit_val[credit_val == 1])]
	credit.assign(credit_val)








import sys
import pandas as pd
import numpy as np
from numpy.random import default_rng
rng = default_rng()

import plotly
import shapefile # If you need to install this, it's in the pyshp package
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import Point # Point class
from shapely.geometry import shape # shape() is a function to convert geo objects through the interface

_path = "/Users/Allegra/Documents/Postdoc/habitus/modules/github/AVC_simulation_origin/"
shape_path = "/Users/Allegra/Documents/Postdoc/habitus/modules/github/SRVzones/shapes/"
sys.path.insert(1, "/Users/Allegra/Documents/Postdoc/habitus/modules/github/AVC_simulation_origin/Framework/")
sys.path.insert(1, "/Users/Allegra/Documents/Postdoc/habitus/modules/github/AVC_simulation_origin/Models/Functions/")

from pop_classes import Pop
from decision_utilities import ps, row_sums, col_sums, row_margins, col_margins, index2onehot, onehot2index, row_sample, ged, ternary
from season_maintenance import *
from credit import update_credit
from plotting_functions import *

# Initialize pop  ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
n = 400                     # number of farms
pop = Pop(size=n)           # override pop


# Get location information set up  ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# Zhuoyu's code follows:
sf = shapefile.Reader(shape_path + 'SN_LHZ_2021.shp')
sf_LowerDelta = shapefile.Reader(shape_path + 'SRV_LowerDelta-polygon.shp')
sf_MiddleValley_Matam = shapefile.Reader(shape_path + 'SRV_MiddleValley_Matam-polygon.shp')
sf_MiddleValley_Podor = shapefile.Reader(shape_path + 'SRV_MiddleValley_Podor-polygon.shp')
sf_UpperDelta = shapefile.Reader(shape_path + 'SRV_UpperDelta_Extended-polygon.shp')
sf1, sf2, sf3, sf4 = sf_LowerDelta, sf_MiddleValley_Matam, sf_MiddleValley_Podor, sf_UpperDelta

shape1, shape2, shape3, shape4 = sf1.shapes(), sf2.shapes(), sf3.shapes(), sf4.shapes() 
zone1, zone2, zone3, zone4 = shape1[0], shape2[0], shape3[0], shape4[0]

bounds1 = zone1.bbox
xmin1, ymin1, xmax1, ymax1 = bounds1
xext1 = xmax1 - xmin1
yext1 = ymax1 - ymin1

bounds2 = zone2.bbox
xmin2, ymin2, xmax2, ymax2 = bounds2
xext2 = xmax2 - xmin2
yext2 = ymax2 - ymin2

bounds3 = zone3.bbox
xmin3, ymin3, xmax3, ymax3 = bounds3
xext3 = xmax3 - xmin3
yext3 = ymax3 - ymin3

bounds4 = zone4.bbox
xmin4, ymin4, xmax4, ymax4 = bounds4
xext4 = xmax4 - xmin4
yext4 = ymax4 - ymin4

po = []

def make_points(po, xmin, ymin, xext, yext, zone):
	# generate a random x and y
	x = xmin + rng.random() * xext
	y = ymin + rng.random() * yext
	point_to_check = (x, y)
	if Point(point_to_check).within(shape(zone)):
	    po.append(point_to_check)
	return po

while len(po) < n/4: po = make_points(po, xmin1, ymin1, xext1, yext1, zone1)
while len(po) < n/2: po = make_points(po, xmin2, ymin2, xext2, yext2, zone2)
while len(po) < (n/4 + n/2): po = make_points(po, xmin3, ymin3, xext3, yext3, zone3)
while len(po) < n: po = make_points(po, xmin4, ymin4, xext4, yext4, zone4)


# Set up params ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
pop.make_param("season", 'hot/dry')
season_param_rule = pop.make_rule(name = 'season_param_rule', fun = update_season, params = [pop.season])

pop.make_param("zones", [zone1, zone2, zone3, zone4])
pop.make_param("sf", sf)

# Set up pop ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# we'll have separate x and y columns
pop.make_column('x',[i[0] for i in po[0:n]] )
pop.make_column('y',[i[1] for i in po[0:n]] )

xmax, ymax = max(pop.x.val), max(pop.y.val)
halfway =  np.floor(xmax/2)
# we'll also have a column of x,y pairs to facilitate measuring distances
pop.make_column('xy',po)

# Each farm belongs to a region
number_to_region = {0: 'dagana', 1: 'podor', 2: 'matam', 3: 'other'}
pop.make_column('region',np.zeros(pop.size))
pop.region.assign(0,selected = pop.members[0:int(n/4)])
pop.region.assign(1,selected = pop.members[int(n/4):int(n/2)])
pop.region.assign(2,selected = pop.members[int(n/2):int(n/2 + n/4)])
pop.region.assign(3,selected = pop.members[int(n/2 + n/4):n])

pop.make_cohort('region_0', lambda: pop.region.eq(0))
pop.make_cohort('region_1', lambda: pop.region.eq(1))
pop.make_cohort('region_2', lambda: pop.region.eq(2))
pop.make_cohort('region_3', lambda: pop.region.eq(3))

# each farm belongs to a coop that is the same as their region
pop.make_column('coop',np.zeros(pop.size))
pop.coop.assign(0,selected = pop.members[0:int(n/4)])
pop.coop.assign(1,selected = pop.members[int(n/4):int(n/2)])
pop.coop.assign(2,selected = pop.members[int(n/2):int(n/2 + n/4)])
pop.coop.assign(3,selected = pop.members[int(n/2 + n/4):n])

# make a cohort for each coop
pop.make_cohort('coop_0', lambda: pop.coop.eq(0))
pop.make_cohort('coop_1', lambda: pop.coop.eq(1))
pop.make_cohort('coop_2', lambda: pop.coop.eq(2))
pop.make_cohort('coop_3', lambda: pop.coop.eq(3))


# let's define markets for products A, B and C by their locations
pop.make_param('A_mkt',np.array([10,10])) # near the northwest corner
pop.make_param('B_mkt',np.array([.9*xmax,.9*ymax])) # near the southeast corner
pop.make_param('C_mkt',np.array([halfway,halfway])) # near the center 


# Each farm is some distance to each market
pop.make_column('dists_to_mkts', 
                np.vstack((
                    ged(pop.A_mkt.val,pop.xy.val),
                    ged(pop.B_mkt.val,pop.xy.val),
                    ged(pop.C_mkt.val,pop.xy.val)
                    )).T)


# Each farm has some attributes
pop.make_column('wealth',rng.exponential(.5,pop.size)) # Wealth between 0 and 10
pop.make_column("land_area", rng.normal(1, 0.5, pop.size))
pop.make_column("predicted_yields", rng.normal(5.75, 0.5, pop.size) * pop.land_area.val)

# This is code Sheng-Tai and Zhuoyu wrote
def build_dist(dist, arr):
	arr[0 : dist[0]] = 0
	arr[dist[0] : dist[0] + dist[1]] = 1
	arr[dist[0] + dist[1] : dist[0] + dist[1] + dist[2]] = 2
	arr[dist[0] + dist[1] + dist[2] :  dist[0] + dist[1] + dist[2] + dist[3]] = 3
	arr[dist[0] + dist[1] + dist[2] + dist[3] :  dist[0] + dist[1] + dist[2] + dist[3] + dist[4]] = 4
	arr[dist[0] + dist[1] + dist[2] + dist[3] + dist[4] :  dist[0] + dist[1] + dist[2] + dist[3] + dist[4]+ dist[5]] = 5
	np.random.shuffle(arr)
	return arr

edu_dist = np.random.multinomial(pop.size, [43/100.] + [40/100.] + [10/100.] + [4/100.] + [2/100.] + [1/100.])
edu = np.zeros(pop.size)
lead_dist = np.random.multinomial(pop.size, [60/100.] + [20/100.] + [12/100.] + [5/100.] + [2/100.] + [1/100.])
lead = np.zeros(pop.size)
pop.make_column('education', build_dist(edu_dist, edu))
pop.make_column('leadership', build_dist(lead_dist, lead))



pop.make_column('credit', np.array(np.random.choice([0,1], p = [0.5,0.5], size = pop.size)))
update_credit_rule = pop.make_rule(name = 'update_credit', fun = update_credit, columns=[pop.credit], cohorts = [cohort for cohort in pop.dynamic_cohorts.values() if 'coop' in cohort.name])


# Some planting probabilities
i_p_crop_hotdry = {'rice': 0.97, 'toms': 0.01, 'onions': 0.01, 'other': 0.01, 'none': 0.0} # Probability OF PLANTING crop, guessed from Dr. Fall data
i_p_crop_colddry = {'rice': 0.01, 'toms': 0.15, 'onions': 0.42, 'other': 0.42, 'none': 0.0}
i_p_crop_rainy = {'rice': 0.61, 'toms': 0.02, 'onions': 0.02, 'other': 0.35, 'none': 0.0} 
li_p_crop_hotdry, li_p_crop_colddry, li_p_crop_rainy = np.array(list(i_p_crop_hotdry.values())), np.array(list(i_p_crop_colddry.values())), np.array(list(i_p_crop_rainy.values()))
num_crops = len(i_p_crop_hotdry.keys())
priors = li_p_crop_hotdry if pop.season.val == 'hot/dry' else (li_p_crop_colddry if pop.season.val == 'cold/dry' else li_p_crop_rainy)



pop.make_column('selected_crop', rng.multinomial(1, priors, pop.size))
pop.make_column('selected_crop_index', onehot2index(pop.selected_crop.val))
pop.make_column('p_crop_success', np.zeros((pop.size,num_crops)))        # This should come from some historical data
pop.make_column('crop_success', np.zeros((pop.size, num_crops)))         # The actual success of the selected crop, should come from DSSAT
pop.make_column('harvest_times', np.zeros(pop.size))         # The actual success of the selected crop, should come from DSSAT

 

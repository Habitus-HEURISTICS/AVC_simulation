# Plotting functions.

import plotnine
from plotnine import *
import pandas as pd
import plotly
import plotly.express as px
from plotly.colors import n_colors, unconvert_from_RGB_255
import shapefile # If you need to install this, it's in the pyshp package
import matplotlib.pyplot as plt
import shapely
import numpy as np

def plot_three_seasons(pld, time_point):
	print(pld)
	pld.Crop.replace({0: 'rice', 1: 'tomatoes', 2: 'onions', 3: 'other', 4: 'nothing'}, inplace = True)
	pld.Region.replace({0: "dagana", 1: "podor", 2: "matam", 3: "other"}, inplace = True)
	pld_grp = pld[pld.Time >= (time_point - 2)].groupby(['Season', 'Region', 'Crop']).agg({'Hectares': 'sum'})

	grp_pcts = pld_grp.groupby(level=[0,1]).apply(lambda x: 100 * x / float(x.sum())).reset_index().rename(columns = {'Hectares' : 'pct_ha'})

	p2 =  (ggplot(grp_pcts) + 
		geom_bar(aes(x = 'Region', y = 'pct_ha', fill = 'factor(Crop)'), position = 'stack', stat = 'identity') + 
		facet_grid('.~Season') + theme_minimal() + scale_x_discrete(limits=["dagana", "podor", "matam", "other"] ) + labs(fill = 'Selected crop'))

	print(p2)


# Modified from Zhuoyu's code
def plot_map_discrete(pop, variable, *color_map):
	zones = pop.zones.val
	zone1, zone2, zone3, zone4 = zones[0], zones[1], zones[2], zones[3]
	var = eval('pop.' + variable + ".val")

	plt.figure()

	if color_map:
		color_map = color_map[0]
	else:
		# colorbook = [unconvert_from_RGB_255(clr) for clr in n_colors((2, 2, 2), (255, 255, 255), len(np.unique(var)))]
		colorbook = [unconvert_from_RGB_255(eval(clr[3:])) for clr in px.colors.qualitative.Vivid]
		color_map = dict(zip(np.unique(var), colorbook[0:len(np.unique(var))]))

	colors = [color_map[i] for i in var]
	plt.scatter(pop.x.val,pop.y.val, c = colors, marker="o", s = 2, picker=True)


	for shape in pop.sf.val.shapeRecords():
		x1 =  [i[0] for i in zone1.points[:]]
		y1 =  [i[1] for i in zone1.points[:]] 

		x2 =  [i[0] for i in zone2.points[:]]
		y2 =  [i[1] for i in zone2.points[:]] 

		x3 =  [i[0] for i in zone3.points[:]]
		y3 =  [i[1] for i in zone3.points[:]] 

		x4 =  [i[0] for i in zone4.points[:]]
		y4 =  [i[1] for i in zone4.points[:]]     
	    
		plt.fill(x1, y1, c='b', alpha=0.01)
		plt.fill(x2, y2, c='r', alpha=0.01)
		plt.fill(x3, y3, c='orange', alpha=0.01)
		plt.fill(x4, y4, c='g', alpha=0.01)


	plt.show()
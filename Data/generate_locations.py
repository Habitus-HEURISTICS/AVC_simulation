import sys
import pandas as pd
import numpy as np
from numpy.random import default_rng
rng = default_rng()
import csv

import plotly
import shapefile # If you need to install this, it's in the pyshp package
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import Point # Point class
from shapely.geometry import shape # shape() is a function to convert geo objects through the interface

_path = "/Users/Allegra/Documents/Postdoc/habitus/modules/github/AVC_simulation_origin/"
shape_path = "/Users/Allegra/Documents/Postdoc/habitus/modules/github/SRVzones/shapes/"

# This code assigns n farms to locations within the SRV zones.
# I put it in its own file because its load time is large and we don't need to do it every time. Remember that the size of the population
#    here needs to match the size in global_pop.py

n = 400
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


with open(_path + "/Data/farm_locations.csv", "w") as f:
    writer = csv.writer(f)
    for row in po:
        writer.writerow(row)







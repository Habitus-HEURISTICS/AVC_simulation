#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday, Oct 20

@author: aabcohen
"""

# Functions for use in season Rules


def update_season(pop, params, **kwargs):
	progression = ['cold/dry', 'hot/dry', 'rainy']
	pop.season.assign(progression[(progression.index(pop.season.val) + 1) % len(progression)])
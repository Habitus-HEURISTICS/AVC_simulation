#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 20:04:51 2021

@author: prcohen
"""

import copy

import numpy as np
import numpy.ma as ma
from numpy.random import default_rng
rng = default_rng(1108) # stuff for random sampling; fix random seed
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

from decision_utilities import ps, row_sums, col_sums, row_margins, col_margins, row_renorm, index2onehot, onehot2index, row_sample, ged, ternary


class Decision ():
    '''
    A decision space for an individual is represented as a multinomial distribution
    over c options.  For example, [.1,.3,.25,.35] is a distribution over c = 4 
    options. Equivalently, a decision space for an individual is a point in c-space,
    which we'll call a decision point.  Deliberating a decision involves iteratively 
    adjusting decision probabilities, or, equivalently, moving a decision point. 
    This is done by the Influence class.  
    
    The decision spaces of a cohort of r individuals is represented as an array 
    called probs which has shape r,c.  The initial value of probs is init_probs. 
    
    probs can be masked and masked values can be clamped.  For efficiency and 
    understandability, masking is limited to entire columns.  (In theory probs 
    could have a 2D mask, but this would require reimplementing more of the 
    functionality of numpy masked arrays or using the ma package, which is an 
    order of magnitude slower than what's here.)  
    
    `mask` should be a boolean vector of length c, where True means masked and 
    False means unmasked (the convention of the ma package).
    
    If masked vals is None, then the masked values are untouched, but sometimes
    we want to clamp the masked values; for example, we might want them to be 
    very small probabilities. In this case, masked_vals should be a vector of
    length c or broadcastable to a vector of length c.
    
    Currently you cannot clamp unmasked values. That is, clamping means, "set the
    values and don't allow them to change."
    
    Influences needs to know the mask and clamped values to ensure that decision
    points remain in the probability simplex.  Influences inherit masks and 
    clamped values from Decisions. 
    
    If copy_probs = True, then self.probs is a copy of init_probs, otherwise 
    changes are made to init_probs.
    
    *args and **kwargs are passed on to Influences.  For example, cohorts or
    populations might be passed.
    
    
    '''
    
    def __init__(self, name, get_init_probs, get_mask = None, get_clamped_probs = None, copy_probs=False, **kwargs): 
        
        self.name = name
           
        # **kwargs must hold named parameters required by get_init_probs
        self.get_init_probs = get_init_probs
        self.get_mask = get_mask
        self.get_clamped_probs = get_clamped_probs
        
        self.funargs = kwargs
            
        # this can be called repeatedly inside of experiments
        self.setup_probs()
        
        self.probs = copy.copy(self.probs) if copy_probs else self.probs
        self.probs0 = copy.copy(self.probs)  # store "prior" distribution
        
        self.r = self.probs.shape[0]
        self.c = self.probs.shape[1]
        self.rc = self.probs.shape
        
        self.influences = []
         
    
    def setup_probs (self):
        ''' Gets initial probabilities, masks and clamps them as required. '''
        
        self.probs = self.get_init_probs(**self.funargs)
           
        self.update_mask()


    def update_mask (self):
        if self.get_mask is not None:
            self.mask = self.get_mask(**self.funargs)
            # check the form
            if type(self.mask[0][0]) != np.bool_ or self.mask.shape != self.probs.shape:
                raise ValueError(f"Mask need to be a boolean array of the same shape as probs")
                
            # clamping happens only if a mask is specified       
            if self.get_clamped_probs is not None: self.clamp_probs()
        else:
            self.mask = None
            self.clamped_probs = None

        
    
    def clamp_probs (self,error_check=True): 
        ''' Masked values can be clamped to anything in the range 0 ≤ x < 1,
        in which case self.probs must be renormalized.  This is done here to 
        ensure that Influences are supplied with simplexes.  Turn off error checking 
        to speed up experiments. '''
        
        self.clamped_probs = self.get_clamped_probs(**self.__dict__)
        
        if error_check:         
            # clamp_probs should return an array of the same shape as probs      
            if self.clamped_probs.shape != self.probs.shape:
                raise ValueError("Clamped probs shape is not probs shape")
            
            # check for cases where the sum of clamped probs >= 1
            x = self.clamped_probs * self.mask
            if any(np.sum(x,axis=1) >= 1.0):
                raise ValueError("The probability mass of clamped values is >= 1 in at least one row.")
        
                  
        # set the values of the masked parts 
        self.probs[self.mask] = self.clamped_probs[self.mask]
        
        # renormalize ONLY the unmasked values:
        x = self.clamped_probs*self.mask
        sumx = 1 - np.sum(x,axis=1)[:,np.newaxis]
        sumy = np.sum(self.probs*~self.mask,axis=1)[:,np.newaxis]
        y = self.probs*~self.mask*sumx/sumy
        self.probs = x+y     
        
    
    def make_influence (self, get_destinations, rate):
        infl = Influence(self,get_destinations, rate)
        self.influences.append(infl)
        return infl
      
    
    def apply_influences (self,  shuffle = True):  
        if shuffle: rng.shuffle(self.influences)
        for infl in self.influences:
            infl.move_p1()
            
            
            
    def decision (self, form = 'index', unmasked_only = False):
        '''Decisions are taken by sampling from probs, which returns a one-hot
        encoding of the selected option.  This transforms the one-hot encoding 
        into the index of the selected column if form == 'index'. 
        
        Depending on the values in masked columns, the decision might be for
        a masked option.  For some decision problems this makes sense, for others
        it defeats the purpose of masking.  The parameter 'unmasked_only' restricts
        decisions to unmasked columns.
        '''
        
        s = row_sample(self.probs[:,~self.mask]) if unmasked_only else row_sample(self.probs)
        return onehot2index(s) if form=='index' else s


class Influence () :
    ''' p1 is a 2D array of shape r,c where each row is a multinomial over
    c classes. Rows p1 in are points in a probability simplex.  The `fun` argument 
    is bound to a function that returns 'destination' points, called p0. Points 
    in p0 are multinomials with c classes. `fun` returns either one point (i.e.,
    a vector of length c) or r points (i.e., an array of the same shape as p0).
    Influences move p1 toward p0 at a specified rate.  The `move_p1` method 
    ensures that all points in p1 remain in the probability simplex.
    
    Influences get most of their parameters from the Decision objects that they
    influence.  This ensures that Influences inherit masking and clamping from
    Decisions, which is necessary if masking or clamping change during a 
    decision problem.  However, it requires a one-to-one correspondence between 
    Decision objects and Influence objects; you can't 'reuse' an influence in 
    another decision, you have to create a new influence object for each decision.
    In practice, this is easy because creating an influence object requires few 
    parameters, and can be done by calling the `make_influence` method of a 
    Decision object.
    
    '''
    
    def __init__(self, decision, get_destinations, rate = .1, *args, **kwargs) :
        
        self.decision = decision
        self.p1 = self.decision.probs
        self.mask = self.decision.mask
        self.umask = ~self.mask if self.mask is not None else None
        
        self.fun = get_destinations
        self.p0 = self.funrun()

        self.rate=rate
        
        self.bad_count = 0
        self.good_count = 0
        
    def funrun(self):
        ''' Runs self.fun. Any parameters to pass to fun can be specified
        as *args or **kwargs when creating an Influence.'''
        
        f = self.fun (**self.__dict__)
        
        # if p0 is a vector, make it a 2D array with only one row
        return np.array([f]) if f.ndim == 1 else f
             
    def move_p1 (self):
        ''' Moves points in self.p1 toward self.p0. The result remains in the 
        simplex.  However, if some columns are masked and we move the remaining 
        lower-dimensional points, then their coordinates plus those of the masked 
        columns can sum to > 1.0. To ensure that moves with masked columns remain 
        in the original simplex, we must first rescale the unmasked columns, 
        then move their points, then invert the rescaling. '''
        
        if self.mask is None:        
            # no rescaling needed
            self.p1 = (1 - self.rate) * self.p1  + self.rate * self.p0
        
        else:
            
            p0_u = self.p0 * self.umask
            p0_u_rescaled = p0_u / np.sum(p0_u,axis = 1)[:,np.newaxis]
            
            p1_u = self.p1 * self.umask
            p1_sums = np.sum(p1_u,axis=1)[:,np.newaxis]
            p1_u_rescaled = p1_u/p1_sums
            
            moved = ((1 - self.rate) * p1_u_rescaled  + self.rate * p0_u_rescaled) * p1_sums
            
            self.p1[self.umask] = moved[self.umask]
        
        self.decision.probs = self.p1



class Experiment ():
    def __init__ (self, decision, k, m, fun, x ):
        self.decision = decision
        # k is the number of replicates
        self.k = k
        # m is the number of times each influence acts within a replicate
        self.m = m
        # x is a list of attributes of the decision to report on
        self.x = x
        # fun is the "inner loop" function that activates influences and calculates things to report
        self.fun = fun
        # report is a list of lists, each inner list contains values of attributes in x
        self.report = []
        
    
    def run (self):
        for i in range(self.k):
            self.decision.setup_probs()
            # TODO: package the following in Influence
            for infl in self.decision.influences:
                infl.p1 = self.decision.probs
                infl.mask = self.decision.mask
                infl.umask = ~infl.mask if infl.mask is not None else None
            for j in range(self.m):
                self.fun(self)
                
                #Copying because otherwise in-place operations overwrite                   
                self.report.append(
                    [i,j] +
                    copy.copy(
                        [self.decision.__getattribute__(var) for var in self.x])
                    )
            



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:21:51 2021

@author: prcohen
"""


from types import SimpleNamespace
from functools import partial

import numpy as np
import numpy.ma as ma
from numpy.random import default_rng
rng = default_rng() # stuff for random sampling
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

# import plotnine
# from plotnine import *

import ternary

class Pop(SimpleNamespace):
    ''' Generally a population will be created from a flat file in which columns are
    variables such as age, wealth, location etc. and rows are individuals. If no 
    file is specified, one can still create an empty population, however, virtually all
    the classes and methods in this framework depend on the population size, so one
    must specify that by overriding the default value. 
    
    '''
    def __init__(self,filepath=None,size=100):
        self.name = 'pop' 
        self.params = {}
        self.columns = {}
        self.static_cohorts = {}
        self.dynamic_cohorts = {}
        self.decisions = {}
        self.rules = {}
        
        if filepath is not None:
            data = pd.read_csv(filepath)    # make a pandas dataframe   
            self.size = len(data)           # set the population size           
            for colname in data.columns:
                # create a Column object for each dataframe column
                self.make_column(colname,np.array(data[colname]))
        else:    
            self.size = size            # create an empty pop of default or given size
            
        self.members = np.arange(self.size)
        self.selected = np.ones(size).astype(bool)
             
    
    def get (self,x):
        return self.__dict__.get(x)

    def make_column (self,name,values):
        column = Column(name,values)
        self.__dict__.update({name:column})
        self.columns.update({name:column})

    def make_cohort (self, name, sexpr, dynamic = True):
        """ makes a cohort and registers it with self.static_cohorts or
        self.dynamic_cohorts depending on the 'dynamic' parameter. sexpr is a
        callable that Cohort uses to decide which individuals are members."""
        cohort = Cohort(name,sexpr,dynamic)
        self.__dict__.update({name:cohort})
        if dynamic:
            self.dynamic_cohorts.update({name:cohort})
        else:
            self.static_cohorts.update({name:cohort})

    def make_param (self, name, initial_value, *args):
        ''' This is for making params that are global to the population '''
        param = Param(name,initial_value,*args)
        self.__dict__.update({name:param})
        self.params.update({name:param})
        return param

    def make_rule (self, name, **kwargs):
        r = Rule(pop = self, **kwargs)
        self.rules.update({name: r})
        return r

    def make_counter (self,name, initial_value = 0, increment = 1):
        counter = Counter(name, initial_value, increment)
        self.__dict__.update({name:counter})
        self.counters.append(counter)
        return counter
        
    
    def reset (self):
        """ This deletes all the Params, counters, columns, and static and
        dynamic cohorts, and resets the lists that contain these entities to {}.
        It preserves pop.size.  This would be used only during development and
        testing when one wants a 'clean' population.
        """
        size = self.size
        self.__dict__.clear()
        self.size = size
        self.counters, self.params, self.columns,self.static_cohorts, self.dynamic_cohorts = {},{},{},{},{}

    
    def map_me (self, xmax, ymax, *column_name, color_names = None, selected = None, **kwargs):
        ''' If column_name is not specified then this plots the x,y locations associated
        with the group members.  If column_name is specified, then it can have two forms:
        a string, which is used to find a column of that name, or a list [name,index], which
        is used when the column holds a 2D array. In this case, index identifies which column
        of the 2D array to use. 
        '''
        map_me = np.empty((xmax+1, ymax+1))*np.nan
        if selected == None:
            x, y = self.__getattribute__('x').val, self.__getattribute__('y').val
        else:
            x, y = self.__getattribute__('x').val[selected.members], self.__getattribute__('y').val[selected.members]

        if column_name:
            if len(column_name) == 1 and type(column_name[0]) == str and selected == None:
                z = self.__getattribute__(column_name[0]).val
            elif len(column_name) == 1 and type(column_name[0]) == str and selected != None:
                z = self.__getattribute__(column_name[0]).val[selected.members]
            elif type(column_name) != str and selected == None:
                z = self.__getattribute__(column_name[0][0]).val[:,column_name[0][1]]
            else:
                z = self.__getattribute__(column_name[0][0]).val[selected.members][:,column_name[0][1]]
        else:
            z = 1
            color_values, color_names = [1], ['location']

        if type(z) == np.ndarray:
            if z.ndim == 1:
                color_values = list(np.unique(z))
            else:
                color_values = list(range(len(z[0,:])))
                z = np.argmax(z, axis=1) # Undo one-hot encoding

        if color_names == None:
            color_names = color_values

        map_me[x, y] = z

        plt.cm.get_cmap().set_bad(color='white')
        if 'vmin' in kwargs.keys() and 'vmax' in kwargs.keys():
            im = plt.imshow(map_me, vmin = kwargs['vmin'], vmax = kwargs['vmax'])
        else:
            im = plt.imshow(map_me)

        
        colors = [ im.cmap(im.norm(value)) for value in color_values] 
        # Stolen from https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib -----
        patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=color_names[i]) ) for i in range(len(color_values)) ]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. ) 

        plt.show()
    
    def describe (self,level=0):
        return (
            f"".join(
                [f"size: {self.size}\n",
                 f"Counters:\n",
                 *[f"{c.describe(level+1)}\n" for c in self.counters],
                 f"Params:\n",
                 *[f"{p.describe(level+1)}\n" for p in self.params],
                 f"Columns:\n",
                 *[f"{c.describe(level+1)}\n" for c in self.columns],
                 f"Static cohorts:\n",
                 *[f"{s.describe(level+1,print_selector=False)}\n" for s in self.static_cohorts],
                 f"Dynamic cohorts:\n",
                 *[f"{d.describe(level+1,print_selector=False)}\n" for d in self.dynamic_cohorts]
                ]
            ))


class Column ():
    """ A Column represents one attribute of all the agents.  """

    def __init__(self,name,values):

        self.name = name
        self.val = values
        self.n = len(values) # should always equal pop.size

        """ Elementary comparison operators for columns. """
        self.eq = partial (self.op, fn = (lambda x,y : x == y) )
        self.ge = partial (self.op, fn = (lambda x,y : x >= y) )
        self.le = partial (self.op, fn = (lambda x,y : x <= y) )
        self.ne = partial (self.op, fn = (lambda x,y : x != y) )
        self.gt = partial (self.op, fn = (lambda x,y : x > y) )
        self.lt = partial (self.op, fn = (lambda x,y : x < y) )
        self.legt = partial (self.op, fn = (lambda x,y : (x <= y[1]) & (x > y[0])) )


    def op (self,y,fn):
        if callable (y):
            return fn(self.val,y.__call__())
        else:
            return fn(self.val,y)

    def assign (self,val,selected=None):
        ''' Assigns value(s) to this column. Let n be the length of the column. 
        Several cases:  
            
            - selected is None and val is a single value : assign val
            to the entire column
            
            - selected is None and val is an np.array:  if len(val) == n, then 
            val is assigned to the column, else an error is thrown
            
            - selected is a boolean array of length n or an index array and val 
            is single value:  selected selects the elements of the column to update 
            with val
            
            - selected is a boolean array of length n and val is an np.array
            of length n:  selected selects the elements of the column to update
            with val
            
            - selected is an index array of length ≤ n and val is an array of
            the same length:  selected replaces the indexed values of the column
            with the values of val      
            
        '''
        val_type = type(val)
        
        if selected is None and val_type in [int,float,bool]:
            self.val[:] = val
        
        elif selected is None and val_type == np.ndarray:
            if len(val) == self.n:
                self.val[:] = val
            else:
                raise ValueError("val must be of the same length as the column")
                
        elif selected is not None and val_type in [int,float,bool]:
             self.val[selected] = val
       
        elif selected is not None and type(selected[0]) in  [np.bool_, bool]:
            if len(selected) == self.n:
                self.val[selected] = val[selected]
            else:
                raise ValueError("val must be of the same length as the column")
        
        elif selected is not None and type(selected[0]) in [np.int_, int]:
            if len(selected) == len(val):
                self.val[selected] = val
            else:
                raise ValueError("val must be of the same length as selected")
                
        else:
            raise ValueError (f"Unable to assign values to {self.name}")


    def get_val (self):
        return self.val

    def itself (self): return self
    
    def hist (self,bins=30):
        plt = (ggplot(pd.DataFrame(self.val,columns = [self.name]), aes(x = self.name) )
               + geom_histogram(bins=bins)
               )
        print(plt)
    
class Param ():
    """
    Whereas Columns must be arrays of length group.size that represent
    agent attributes, Params can be anything: a prior or vector of priors,
    the name of the season, etc. Params require a name,  an initial value 
    and an optional updater, which must be a callable. Params hold values that
    do not depend on the individual or cohort; if you want to define something 
    that *does* depend on the individual or cohort, use Column.
    """
    def __init__(self, name, val):
        self.name = name
        self.val  = val

    def assign (self,val):
        self.val = val

    def get_val (self):
        return self.val
    
    def get (self,x):
        return self.__dict__.get(x)

    def itself (self): return self



class Rule ():
    def __init__(self, fun, pop, cohorts=None, params=None, columns=None):
        
        self.fun = fun
        self.pop = pop
        self.record = {'params' : params, 'columns' : columns}
        self.history = []
        self.cohorts = {i.name:i for i in cohorts or []}
        self.params =  {i.name:i for i in params or []}
        self.columns = {i.name:i for i in columns or []}
        
        if fun is not None: 
            self.fun = fun
            self.record.update({'fun': self.fun.__name__})
        
        if cohorts is not None:
            self.record.update({'cohorts':[c.name for c in cohorts]})
            self.members = np.hstack([c.members for c in cohorts])
            self.selected =  np.logical_or.reduce([c.selected for c in cohorts])
        else: 
            if pop is not None:
                self.members =  pop.members
                self.selected = pop.selected
            else:
                raise ValueError("You must specify either cohorts or pop")
                
        
    def runfun (self):
        self.history.append(self.record)
        # run the function with all the named arguments that were created above
        # use self.__dict__.keys() to see what they are
        self.fun (**self.__dict__)
        
    def funrun (self):
        # for Allegra
        self.runfun()
        


class Cohort(SimpleNamespace):
    
    def __init__(self, name, selector, dynamic=True):
        self.name = name
        self.dynamic = dynamic
        self.selector = selector
        self.selected =  self.select()   # select initial membership

    def select (self):
        if (not self.dynamic) and self.__dict__.get('selected') is not None:
            return self.selected

        else:
            if callable(self.selector):                 # e.g., a lambda expression
                self.selected = self.selector()
            else:
                raise TypeError ("Param {self.name} updater must be callable or a WN expression")

            self.members  = np.where(self.selected)[0]  # indices of rows that satisfy the selector
            self.size = len(self.members)               # number of members
            return self.selected

    def get_members (self):
        return self.members
    
    def get (self,x):
        return self.__dict__.get(x)





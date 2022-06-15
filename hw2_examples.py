#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import warnings
with warnings.catch_warnings():
        warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# In[17]:


from programming_assignment_2 import phi_calc,log_barrier,line_search,newton,newton_method,interior,linear_problem,lp_ineq_constraints,quadratic_problem_ineq_constraints,quadratic_problem,plot_iterations,plot_feasible_set_3d,plot_linear_feasible_region


# # Quadratic problem

# In[15]:


def test_quadratic_problem():
    start_point = np.array([[0.1], [0.2], [0.7]])
    init_step_len = 1.0
    slope_ratio = 1e-4
    back_track_factor = 0.2
    A = np.array([[1, 1, 1]])
    b = [[1]]
    X,obj_values,outer_obj_values,x_s = interior(quadratic_problem,quadratic_problem_ineq_constraints,A,b,start_point,init_step_len,slope_ratio,back_track_factor)
    
    plot_iterations("Quadratic Problem",outer_obj_values,obj_values,"Outer objective values","Objective values")
    plot_feasible_set_3d(x_s)
test_quadratic_problem()


# # Linear problem
# 

# In[45]:


def test_linear_problem():
    start_point = np.array([[0.5], [0.75]])
    init_step_len = 1.0
    slope_ratio = 1e-4
    back_track_factor = 0.2
    A = np.zeros((0, 0))
    b = np.zeros((0))
    X,obj_values,outer_obj_values,x_s = interior(linear_problem,lp_ineq_constraints,A,b,start_point,init_step_len,slope_ratio,back_track_factor)
    
    plot_iterations("Objective function values of lp function",outer_obj_values,obj_values,"Outer objective values","Objective values")
    plot_linear_feasible_region("linear minimization problem - feasible region",x_s)
test_linear_problem()


# In[ ]:





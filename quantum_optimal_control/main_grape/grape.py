import tensorflow as tf
import numpy as np
import scipy.linalg as la
from quantum_optimal_control.core.tensorflow_state import TensorflowState
from quantum_optimal_control.core.system_parameters import SystemParameters
from quantum_optimal_control.core.convergence import Convergence
from quantum_optimal_control.core.run_session import run_session

import random as rd
import time
from IPython import display

import os


def Grape(H0, Hops, Hnames, U, total_time, steps, 
          states_concerned_list, 
          convergence=None, 
          U0=None, reg_coeffs=None, 
          dressed_info=None, maxA=None, 
          use_gpu=True, sparse_H=True, 
          sparse_U=False, sparse_K=False, 
          draw=None, initial_guess=None, 
          show_plots=True, unitary_error=1e-4, 
          method='Adam', state_transfer=False, 
          no_scaling=False, freq_unit='GHz', 
          save=True, data_path=None, 
          Taylor_terms=None, use_inter_vecs=True):
    
    # start time
    grape_start_time = time.time()
    
    # set timing unit used for plotting
    freq_time_unit_dict = {"GHz": "ns", "MHz": "us","KHz":"ms","Hz":"s"}
    time_unit = freq_time_unit_dict[freq_unit]
    
    # make sparse_{H,U,K} False if use_gpu is True, as GPU Sparse Matmul is not supported yet.
    if use_gpu:
        sparse_H = False
        sparse_U = False
        sparse_K = False
    
    if U0 is None:
        U0 = np.identity(len(H0))

    if convergence is None:
        convergence = {'rate': 0.01, 'update_step': 100, 
                       'max_iterations': 5000,
                       'conv_target': 1e-8, 
                       'learning_rate_decay': 2500}
        
    if maxA is None:
        if initial_guess is None:
            maxAmp = 4*np.ones(len(Hops))
        else:
            maxAmp = 1.5*np.max(np.abs(initial_guess))*np.ones(len(Hops))
    else:
        maxAmp = maxA
    
    # pass in system parameters
    sys_para = SystemParameters(H0, Hops, Hnames, U, U0, total_time, steps, 
                                states_concerned_list, dressed_info, maxAmp, 
                                draw, initial_guess, show_plots, unitary_error, 
                                state_transfer, no_scaling, reg_coeffs, save, Taylor_terms, 
                                use_gpu, use_inter_vecs, sparse_H, sparse_U, sparse_K)
    
    if use_gpu:
        dev = '/gpu:0'
    else:
        dev = '/cpu:0'
        
    with tf.device(dev):
        tfs = TensorflowState(sys_para) # create tensorflow graph
        graph = tfs.build_graph()

    conv = Convergence(sys_para, time_unit, convergence)
    
    # run the optimization
    try:
        SS = run_session(tfs, graph, conv, sys_para, method, 
                         show_plots=sys_para.show_plots, 
                         use_gpu=use_gpu)
        return SS.uks,SS.Uf
    except KeyboardInterrupt:
        display.clear_output()
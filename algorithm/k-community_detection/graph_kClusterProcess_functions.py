#!/usr/bin/env python

import matplotlib.pyplot as plt
import re, os, sys

import networkx as nx
from numpy import linalg as la
from networkx.generators.atlas import *
import numpy as np
import networkx as nx
import random, copy
import math
from scipy.sparse import csr_matrix
import argparse
import logging
import datetime as dt


#
# The Quantum Graph Community Detection Algorithm has been described
# in the following publications. Please cite in your publication.
#
# H. Ushijima-Mwesigwa, C. F. A. Negre, S. M. Mniszewski,
# 2017, Graph Partitioning using Quantum Annealing on the
# D-Wave System, Proceedings of the 2nd International
# Workshop on Post Moore’s Era Supercomputing (PMES), 22-29.
#
# C. F. A. Negre, H. Ushijima-Mwesigwa, S. M. Mniszewski 2020, Detecting
# Multiple Communities using Quantum Annealing on the D-Wave System,
# PLOS ONE 15(2): e0227538. https://doi.org/10.1371/journal.pone.0227538
#
# S. M. Mniszewski, P. A. Dub, S. Tretiak, P. M. Anisimov, Y. Zhang,
# C. F. A. Negre, 2021, Reduction of the Molecular Hamiltonian Matrix using 
# Quantum Community Detection, Sci Rep 11, 4099 (2021).
# https://doi.org/10.1038/s41598-021-83561-x#
#


def build_mod(Adj, thresh, num_edges):
  #Builds the modularity matrix from the Adjacency matrix.
  #Given an adj matrix, it constructs the modularity matrix and its graph.

  Dim = Adj.shape[1]
  print ("\n Dim = ", Dim)

  print ("\n Computing modularity matrix ...")

  Deg = np.zeros([Dim])

  M = 0.0

  # Calc Adj degrees
  Deg = Adj.sum(1)
  M = Deg.sum()
  mtotal = M/2.0

  Mod = np.zeros([Dim,Dim])

  # Calc modularity matrix
  Mod = Mod + Adj

  Mod = Mod - (Deg * Deg.T)/M

  np.set_printoptions(precision=3)

  return mtotal, Mod


def get_block_number(big_indx, num_blocks, num_nodes):
	  
  #indx = math.ceil(big_indx/num_nodes) # node indx starts from 0
  indx = math.floor(big_indx/num_nodes) # node indx starts from 0
  #print("big_indx=", big_indx," Indx=", indx, " num_blocks=", num_blocks)
  if indx > num_blocks-1:
    raise ValueError("block indx cannot be larger than num_blocks-1")
  return int(indx)


def get_indx_within_block(big_indx, num_nodes):
	  
  return big_indx%num_nodes


def get_entry_beta_B(i_indx, j_indx, beta, graph, modularity, num_nodes, num_blocks):
	  
  i_block_indx = get_block_number(i_indx, num_blocks, num_nodes)
  j_block_indx = get_block_number(j_indx, num_blocks, num_nodes)
  i_indx_within_block = get_indx_within_block(i_indx, num_nodes)
  j_indx_within_block = get_indx_within_block(j_indx, num_nodes)
  
  if i_block_indx == j_block_indx:
    return beta*modularity[i_indx_within_block, j_indx_within_block]
  else:
    return 0

	    
def get_entry_B_Gamma(i_indx, j_indx, modularity, beta,gamma, GAMMA, num_nodes, num_parts, num_blocks):

  i_indx_within_block = get_indx_within_block(i_indx, num_nodes)
  j_indx_within_block = get_indx_within_block(j_indx, num_nodes)
  if i_indx_within_block == j_indx_within_block:
    return gamma[i_indx_within_block]
	    
  else:
    return 0
	    
	    
def get_entry_add_diag(i_indx,gamma, GAMMA, num_nodes, num_parts, num_blocks):
  gamma_entry = GAMMA[i_indx]

  return -2*gamma_entry
	  

def get_i_j_entry(i_indx, j_indx, modularity, beta, gamma, GAMMA, graph, num_nodes, num_parts, num_blocks):

  #print("i_indx=", i_indx," j_indx=", j_indx)
  if i_indx == j_indx:
    bB = get_entry_beta_B(i_indx, j_indx, beta, graph, modularity, num_nodes, num_blocks)
    BG = get_entry_B_Gamma(i_indx, j_indx, modularity, beta, gamma, GAMMA, num_nodes, num_parts, num_blocks)
    diag = get_entry_add_diag(i_indx,gamma, GAMMA, num_nodes, num_parts, num_blocks)
    return bB + BG + diag 
	    
  else:
    bB = get_entry_beta_B(i_indx, j_indx, beta, graph, modularity, num_nodes, num_blocks)
    BG = get_entry_B_Gamma(i_indx, j_indx, modularity, beta, gamma, GAMMA, num_nodes, num_parts, num_blocks)

    return bB + BG


def threshold_mmatrix(graph, mmatrix, threshold):

  msize = mmatrix.shape[0]
  for i in range(0, msize):
    mmatrix[i,i] = mmatrix[i,i] + graph.degree(i)

  for i in range(0, msize):
    for j in range(0, msize):
      if i!=j and abs(mmatrix[i,j]) < threshold:
        mmatrix[i,j] = 0.0

  return mmatrix


def makeQubo(graph, modularity, beta, gamma, GAMMA, num_nodes, num_parts, num_blocks, threshold):

  # Create QUBO matrix
  qsize = num_blocks*num_nodes
  Q = np.zeros((qsize,qsize))

  # Note: weights are set to the negative due to maximization

  # Set node weights
  for i in range(qsize):
    entry = get_i_j_entry(i, i, modularity, beta, gamma, GAMMA, graph, num_nodes, num_parts, num_blocks)
    Q[i,i] = -entry

  # Set off-diagonal weights
  for i in range(qsize):
    for j in range(i, qsize):
      if i != j:
        entry = get_i_j_entry(i, j, modularity, beta, gamma, GAMMA, graph, num_nodes, num_parts, num_blocks)
        if abs(entry) > threshold:
          Q[i,j] = -entry
          Q[j,i] = -entry

  return Q


def violating_contraints(graph, x_indx, num_blocks, num_nodes, num_parts, result):

  #each node in exactly one part
  
  for node in range(num_nodes):
    value = 0
    for j in range(num_blocks):
      value += x_indx[(node, j)]
    if value >1:
      print ("constraint violated: node %d in %d parts. Degree: %d" %(node, value, graph.degree(node)))
    value = 0
 
 #balancing contraints
  sum_v_i = 0
  for node in range(num_nodes):
    sum_x_ik = 0
    for j in range(num_blocks):
      sum_x_ik +=  x_indx[(node, j)]
    node_i = (1 - sum_x_ik)

    sum_v_i += node_i

  print ("\nlast part size",sum_v_i , - num_nodes/float(num_parts))
  num_clusters_found = 0
  for j in range(num_blocks):
    value = 0
    for node in range(num_nodes):
      value += x_indx[(node, j)]
    print ("part %d has %d nodes" %(j, value))
    if value > 0:
      num_clusters_found += 1
  result['num_clusters_found'] = num_clusters_found
    

#######################################################
########  penalty weight function #####################
#######                            ###################
def set_penalty_constant(num_nodes, num_blocks, beta0, gamma0):
 beta = beta0 
 gamma = [gamma0 for i in range(num_nodes)] 
 GAMMA = [gamma[i] for j in range(num_blocks) for i in range(num_nodes) ]

 return beta, gamma, GAMMA
 #########
 

def calcModularityMetric(mtotal, modularity, part_number):
  Dim = modularity.shape[1]
  print ("\n Dim = ", Dim)
  msum = 0.0
  for ii in range(0, Dim):
    for jj in range(0, Dim): 
      if part_number[ii] == part_number[jj]:
        msum = msum + modularity[ii,jj]

  mmetric = msum / (2.0 * mtotal)

  return mmetric


def calc_cut(graph, part_number):

  cut = 0
  for edge in graph.edges():
    u, v = edge
    if part_number[u] != part_number[v]:
      cut += 1

  return cut


def process_solution(ss, graph, num_blocks, num_nodes, num_parts, result):

  qsol = {}
  for i in range(num_blocks*num_nodes):
    qsol[i] = int(ss[0,i])

  qtotal = 0
  for i in range(num_blocks*num_nodes):
    qtotal += qsol[i]
  print('\nnum non-zeros = ', qtotal)

  x_indx = {}
  qubo_soln = qsol
  for i in range(num_blocks*num_nodes):
    i_block_indx = get_block_number(i, num_blocks, num_nodes)
    i_indx_within_block = get_indx_within_block(i, num_nodes)
    x_indx[(i_indx_within_block, i_block_indx)] = qubo_soln[i]

  violating_contraints(graph, x_indx, num_blocks, num_nodes, num_parts, result)

  part_number = {}
  for key in x_indx:
    node, part = key
    if x_indx[key] == 1:
      part_number[node] = part

  return part_number


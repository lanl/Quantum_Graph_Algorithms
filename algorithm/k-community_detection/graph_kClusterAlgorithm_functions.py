#!/usr/bin/env python

import matplotlib.pyplot as plt
import re, os, sys

from dwave_qbsolv import QBSolv
from dwave.system.samplers import DWaveSampler, DWaveCliqueSampler
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
import dimod
import hybrid
import minorminer

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
  Q = np.zeros([qsize,qsize])

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


def write_qubo_file(graph, modularity, beta, gamma, GAMMA, num_nodes, num_parts, num_blocks, threshold):

  ###qubo format
  # p qubo target maxDiagonals nDiagonals nElements
  #target = 0 implies unconstrained problem
  
  nElements = 0 #to be counted
  maxDiagonals = num_nodes*num_blocks # number of diagonal in topology
  nDiagonals = num_nodes*num_blocks #number of diagonals the problem

  qubo_file = open("body.qubo", "w")

  # Write node header
  qubo_string_diag = "".join(["\nc nodes first \n"])
  qubo_file.write(qubo_string_diag)

  # Write nodes
  for i in range(num_blocks*num_nodes):
    entry = get_i_j_entry(i, i, modularity, beta, gamma, GAMMA, graph, num_nodes, num_parts, num_blocks)
    qubo_string_diag = "".join([str(i)+" "+str(i)+" "+str(entry)+"\n"])
    qubo_file.write(qubo_string_diag)

  # Write coupler header
  qubo_string_couplers  = "".join(["\nc couplers \n"])
  qubo_file.write(qubo_string_couplers)

  # Write couplers
  for i in range(num_blocks*num_nodes):
    for j in range(i, num_blocks*num_nodes):
      if i != j:
        entry = get_i_j_entry(i, j, modularity, beta, gamma, GAMMA, graph, num_nodes, num_parts, num_blocks)
        if abs(entry) > threshold:
          qubo_string_couplers = "".join([str(i)+" "+str(j)+" "+str(2*entry)+"\n"]) #x2 because of what qbsolv minimizes
          qubo_file.write(qubo_string_couplers)
          nElements += 1

  qubo_file.close()

  # Write header to separate file now that we know the nElements
  # p qubo target maxDiagonals nDiagonals nElements       
  qubo_file = open("graph.qubo", "w")

  qubo_string_initialize = "".join(["p qubo 0 " + str(maxDiagonals)+" "+str(nDiagonals)+" "+str(nElements)+"\n"])

  qubo_file.write(qubo_string_initialize)
  qubo_file.close()

  # Put qubo file together - header and body
  os.system("cat body.qubo >> graph.qubo")
  os.system("rm body.qubo")
  
        
def get_qubo_solution():

  myFile = open("dwave_output.out", 'r')
  line_count = 0
  
  for lines in myFile:
    line_count += 1
    if line_count == 2:
      bit_string = lines
      break

  return bit_string.strip() 
        
       
def violating_contraints(graph, x_indx, num_blocks, num_nodes, num_parts):

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
  for j in range(num_blocks):
    value = 0
    for node in range(num_nodes):
      value += x_indx[(node, j)]
    print ("part %d has %d nodes" %(j, value))
    

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


def run_qbsolv():

  rval = random.randint(1,1000)
  estring = "qbsolv -r " + str(rval) + " -i graph.qubo -m -o dwave_output.out"
  print('\n', estring)
  os.system(estring)


def process_solution_qbsolv(graph, num_blocks, num_nodes, num_parts):

  bit_string =  get_qubo_solution()
  print (bit_string)
  print ("num non-zeros: ", sum([int(i) for i in bit_string]))

  x_indx = {}
  qubo_soln = [int(i) for i in bit_string]
  for i in range(num_blocks*num_nodes):
    i_block_indx = get_block_number(i, num_blocks, num_nodes)
    i_indx_within_block = get_indx_within_block(i, num_nodes)
    x_indx[(i_indx_within_block, i_block_indx)] = qubo_soln[i]

  violating_contraints(graph, x_indx, num_blocks, num_nodes, num_parts)

  part_number = {}
  for key in x_indx:
    node, part = key
    if x_indx[key] == 1:
      part_number[node] = part

  return part_number


def process_solution(ss, graph, num_blocks, num_nodes, num_parts):

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

  violating_contraints(graph, x_indx, num_blocks, num_nodes, num_parts)

  part_number = {}
  for key in x_indx:
    node, part = key
    if x_indx[key] == 1:
      part_number[node] = part

  return part_number


def getEmbedding(qsize):

  #dsystem = DWaveCliqueSampler()
  #embedding = dsystem.largest_clique()
  #print('embedding found, len = ', len(embedding))
  #print('embedding = ', embedding)
  #exit(0)

  ksize = qsize 
  qsystem = DWaveSampler()
  ksub = nx.complete_graph(ksize).edges()
  embedding = minorminer.find_embedding(ksub, qsystem.edgelist)
  print('\nembedding done')

  return embedding


def runDwave(Q, num_nodes, k, embedding, qsize, run_label, result):

  # Using D-Wave/qbsolv
  # Needed when greater than number of nodes/variables that can fit on the D-Wave
  sampler  = FixedEmbeddingComposite(DWaveSampler(), embedding)
  #sampler  = DWaveCliqueSampler()

  t0 = dt.datetime.now()
  solution = QBSolv().sample_qubo(Q, solver=sampler,
                           label=run_label)
  wtime = dt.datetime.now() - t0
  result['wall_clock_time'] = wtime

  # Collect first energy and num_occ, num diff solutions, and total solutions
  first = True
  ndiff = 0
  total_solns = 0
  for sample, energy, num_occurrences in solution.data():
    #print(sample, "Energy: ", energy, "Occurrences: ", num_occurrences)
    if first == True:
      result['energy'] = energy
      result['num_occ'] = num_occurrences
      first = False
    ndiff += 1
    total_solns += num_occurrences
  result['num_diff_solns'] = ndiff
  result['total_solns'] = total_solns

  print('\n qbsolv response:')
  print(solution)
  ss = solution.samples()
  #print("\n qbsolv samples=" + str(list(solution.samples())))
  #print('\nss = ', ss)
  print(flush=True)

  return ss 

def runDwaveHybrid(Q, num_nodes, k, sub_qsize, run_label, result):

  bqm = dimod.BQM.from_qubo(Q)

  rparams = {}
  rparams['label'] = run_label

  # define the workflow
  iteration = hybrid.Race(
    hybrid.InterruptableTabuSampler(),
    hybrid.EnergyImpactDecomposer(size=sub_qsize, rolling=True, rolling_history=0.15)
    | hybrid.QPUSubproblemAutoEmbeddingSampler(num_reads=100, sampling_params=rparams)
    | hybrid.SplatComposer()
  ) | hybrid.MergeSamples(aggregate=True)
  workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=3)

  # Run the workflow
  init_state = hybrid.State.from_problem(bqm)
  t0 = dt.datetime.now()
  solution = workflow.run(init_state).result()
  wtime = dt.datetime.now() - t0

  result['wall_clock_time'] = wtime

  # Collect from lowest energy result
  result['energy'] = solution.samples.first.energy
  result['num_occ'] = solution.samples.first.num_occurrences

  # Collect number of different solutions w different energies
  result['num_diff_solns'] = len(solution.samples)

  total_solns = 0
  for energy, num_occ in solution.samples.data(['energy', 'num_occurrences']):
    total_solns += num_occ
  result['total_solns'] = total_solns

  # Show list of results in energy order
  print(solution.samples)

  # Collect the first solution
  ss = np.zeros([1,num_nodes])
  for i in range(num_nodes):
    ss[0,i] = solution.samples.first.sample[i]

  return ss

def cluster(Q, k, embedding, qsize, run_label, result):

  # Start with Q
  qsize = Q.shape[1]
  print('\n Q size = ', qsize)

  # Cluster into k parts using DWave
  ss = runDwave(Q, qsize, k, embedding, qsize, run_label, result)

  return ss

def clusterHybrid(Q, k, sub_qsize, run_label, result):
  
# Start with Q
  qsize = Q.shape[1]
  print('\n Q size = ', qsize)

  # Cluster into k parts using Hybrid/DWave ocean
  ss = runDwaveHybrid(Q, qsize, k, sub_qsize, run_label, result)

  return ss


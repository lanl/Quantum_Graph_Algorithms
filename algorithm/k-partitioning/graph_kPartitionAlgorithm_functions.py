#!/usr/bin/env python

import matplotlib.pyplot as plt
import re, os, sys

from dwave.system import LeapHybridSampler
from dwave.system.samplers import DWaveSampler, DWaveCliqueSampler
from dwave.system.composites import EmbeddingComposite
import dimod
import hybrid
import minorminer
from functools import partial
from dwave.embedding.chain_strength import uniform_torque_compensation

import numpy as np
import networkx as nx
from numpy import linalg as la
from networkx.generators.atlas import *
import random, copy
import math
import argparse
from scipy.sparse import csr_matrix
import datetime as dt

import graphFileUtility_functions as GFU
from qpu_sampler_time import QPUTimeSubproblemAutoEmbeddingSampler

#
# The Quantum Graph Partitioning Algorithm has been described
# in the following publications. Please cite in your publication.
#
# H. Ushijima-Mwesigwa, C. F. A. Negre, S. M. Mniszewski,
# 2017, Graph Partitioning using Quantum Annealing on the
# D-Wave System, Proceedings of the 2nd International
# Workshop on Post Moore’s Era Supercomputing (PMES), 22-29.
#


def get_block_number(big_indx, num_blocks, num_nodes):
  
  #indx = math.ceil(big_indx/num_nodes) # node indx starts from 0
  indx = math.floor(big_indx/num_nodes) # node indx starts from 0
  if indx > num_blocks-1:
    raise ValueError("block indx cannot be larger than num_blocks-1")
  return int(indx)


def get_indx_within_block(big_indx, num_nodes):
  
  return big_indx%num_nodes


def get_entry_beta_L(i_indx, j_indx, beta, graph, laplacian, num_nodes, num_blocks):
  
  i_block_indx = get_block_number(i_indx, num_blocks, num_nodes)
  j_block_indx = get_block_number(j_indx, num_blocks, num_nodes)
  i_indx_within_block = get_indx_within_block(i_indx, num_nodes)
  j_indx_within_block = get_indx_within_block(j_indx, num_nodes)
  
  if i_block_indx == j_block_indx:
    return beta*laplacian[i_indx_within_block, j_indx_within_block]
  else:
    return 0


def get_entry_alpha_I(i_indx, j_indx, alpha, num_nodes, num_blocks):
  i_block_indx = get_block_number(i_indx, num_blocks, num_nodes)
  j_block_indx = get_block_number(j_indx, num_blocks, num_nodes)
  i_indx_within_block = get_indx_within_block(i_indx, num_nodes)
  j_indx_within_block = get_indx_within_block(j_indx, num_nodes)
  
  if i_block_indx == j_block_indx:
    return alpha[i_block_indx]
  else:
    return 0
    
    
def get_entry_B_Gamma(i_indx, j_indx, laplacian, alpha, beta,gamma, GAMMA, num_nodes, num_parts, num_blocks):
  i_indx_within_block = get_indx_within_block(i_indx, num_nodes)
  j_indx_within_block = get_indx_within_block(j_indx, num_nodes)
  if i_indx_within_block == j_indx_within_block:
    return gamma[i_indx_within_block]
    
  else:
    return 0
    
    
def get_entry_add_diag(i_indx,alpha,gamma, GAMMA, num_nodes, num_parts, num_blocks):
  i_block_indx = get_block_number(i_indx, num_blocks, num_nodes)
  gamma_entry = GAMMA[i_indx]
  alpha_one_entry = alpha[i_block_indx]
  
  return -2*gamma_entry - 2*num_nodes/float(num_parts)*alpha_one_entry
  

def get_i_j_entry(i_indx, j_indx, laplacian, alpha, beta,gamma, GAMMA, graph, num_nodes, num_parts, num_blocks):

  if i_indx == j_indx:
    bL = get_entry_beta_L(i_indx, j_indx, beta, graph, laplacian, num_nodes, num_blocks)
    aI = get_entry_alpha_I(i_indx, j_indx, alpha, num_nodes, num_blocks)
    BG = get_entry_B_Gamma(i_indx, j_indx, laplacian, alpha, beta,gamma, GAMMA, num_nodes, num_parts, num_blocks)
    diag = get_entry_add_diag(i_indx, alpha, gamma, GAMMA, num_nodes, num_parts, num_blocks)
    return bL + aI + BG + diag 
    
  else:
    bL = get_entry_beta_L(i_indx, j_indx, beta, graph, laplacian, num_nodes, num_blocks)
    aI = get_entry_alpha_I(i_indx, j_indx, alpha, num_nodes, num_blocks)
    BG = get_entry_B_Gamma(i_indx, j_indx, laplacian, alpha, beta,gamma, GAMMA, num_nodes, num_parts, num_blocks)

    return bL + aI + BG


def makeQubo(laplacian, alpha, beta, gamma, GAMMA, graph, num_nodes, num_parts, num_blocks):

  # Create QUBO matrix
  qsize = num_blocks*num_nodes
  Q = np.zeros([qsize,qsize])

  for i in range(qsize):
    for j in range(i, qsize):
      if i == j:
        entry = get_i_j_entry(i, j, laplacian, alpha, beta, gamma, GAMMA, graph, num_nodes, num_parts, num_blocks)
        Q[i,i] = entry
      else:
        entry = get_i_j_entry(i, j, laplacian, alpha, beta,gamma, GAMMA, graph, num_nodes, num_parts, num_blocks)
        if entry > 1e-5 or entry < -1e-5:
          #entry != 0
          Q[i,j] = entry
          Q[j,i] = entry

  return Q


def write_qubo_file(laplacian, alpha, beta, gamma, GAMMA, graph, num_nodes, num_parts, num_blocks):

  ###qubo format
  # p qubo target maxDiagonals nDiagonals nElements
  #target = 0 implies uncontrained problem
  
  nElements = 0 #to be counted
  maxDiagonals = num_nodes*num_blocks # number of diagonal in topology
  nDiagonals = num_nodes*num_blocks #number of diagonals the problem

  qubo_string_diag = "\nc nodes first \n"
  qubo_string_couplers  = "\nc couplers \n"
  for i in range(num_blocks*num_nodes):
    for j in range(i, num_blocks*num_nodes):
      if i == j:
        entry = get_i_j_entry(i, j, laplacian, alpha, beta, gamma, GAMMA, graph, num_nodes, num_parts, num_blocks)
        qubo_string_diag += str(i)+" "+str(j)+" "+str(entry)+"\n"
      else:
        entry = get_i_j_entry(i, j, laplacian, alpha, beta,gamma, GAMMA, graph, num_nodes, num_parts, num_blocks)
        if entry > 1e-5 or entry < -1e-5:
          #entry != 0
          qubo_string_couplers += str(i)+" "+str(j)+" "+str(2*entry)+"\n" #x2 because of what qbsolv minimizes
          nElements += 1
  # p qubo target maxDiagonals nDiagonals nElements       
  qubo_string_initialize = "p qubo 0 " + str(maxDiagonals)+" "+str(nDiagonals)+" "+str(nElements)+"\n"
  
  qubo_string = "".join([qubo_string_initialize, qubo_string_diag, qubo_string_couplers])
  
  qubo_file = open("graph.qubo", "w")
  qubo_file.write(qubo_string)
  qubo_file.close()
        
  
def get_qubo_solution():
  myFile = open("dwave_output.out", 'r')
  line_count = 0
  
  for lines in myFile:
    line_count += 1
    if line_count == 2:
      bit_string = lines
      break

  return bit_string.strip() 
        
       
def violating_contraints(graph, x_indx, num_blocks, num_nodes, num_parts, result):
  #each node in exactly one part
  
  for node in range(num_nodes):
    value = 0
    for j in range(num_blocks):
      value += x_indx[(node, j)]
    if value != 1:
      print("constraint violated: node %d in %d parts" %(node, value))
    value = 0
 
 #balancing contraints
  sum_v_i = 0
  for node in range(num_nodes):
    sum_x_ik = 0
    for j in range(num_blocks):
      sum_x_ik +=  x_indx[(node, j)]
    node_i = (1 - sum_x_ik)
    sum_v_i += node_i

  print("\nlast part size",sum_v_i , - num_nodes/float(num_parts))
  num_parts_found = 0
  for j in range(num_blocks):
    value = 0
    for node in range(num_nodes):
      value += x_indx[(node, j)]
    print("part %d has %d nodes" %(j, value))
    if value > 0:
      num_parts_found += 1
  result['num_parts_found'] = num_parts_found
    

def compute_cut(graph, part_number, num_nodes, num_parts, num_blocks):

  nodes_in_part = [[] for i in range(num_parts)]

  for node in graph.nodes():
    if node not in part_number:
      #part_number[node] = num_parts-1
      print("node %d not assigned a part" %node)
      #print "assigning part %d to node %d" %(num_parts-1, node)

  for node in part_number:
    nodes_in_part[part_number[node]].append(node)

  print('\nnode2part ', part_number)
  print('\n')
  ip = 0
  for part in nodes_in_part:
    print("part ", ip, " size: ", len(part), part)
    ip = ip + 1

  cut = 0
  for edge in graph.edges():
    u, v = edge
    if part_number[u] != part_number[v]:
      cut += 1

  return cut


def compute_cut_qbsolv(graph, bit_string, num_nodes, num_parts, num_blocks, result):
  
  nodes_in_part = [[] for i in range(num_parts)]
  x_indx = {}
  qubo_soln = [int(i) for i in bit_string]
  
  for i in range(num_blocks*num_nodes):
      i_block_indx = get_block_number(i, num_blocks, num_nodes)
      i_indx_within_block = get_indx_within_block(i, num_nodes)
      x_indx[(i_indx_within_block, i_block_indx)] = qubo_soln[i]

  violating_contraints(graph, x_indx, num_blocks, num_nodes, num_parts, result)

  part_number = {}
  for key in x_indx:
    node, part = key
    if x_indx[key] == 1:
      #nodes_in_part[part].append(node)
      #if node in part_number:
      #  print "node %d in multiple part: (%d, %d)" %(node, part_number[node], part)
      #  print "placing in part %d" %part
      part_number[node] = part

  GFU.write_partfile(graph, part_number, num_nodes, num_parts)

  for node in graph.nodes():
    if node not in part_number:
      #part_number[node] = num_parts-1
      print("node %d not assigned a part" %node)
      #print "assigning part %d to node %d" %(num_parts-1, node)

  for node in part_number:
    nodes_in_part[part_number[node]].append(node)
 
  print('\nnode2part ', part_number)
  print('\n')
  ip = 0
  for part in nodes_in_part:
    print("part ", ip, " size: ", len(part), part)
    ip = ip + 1

  cut = 0  
  for edge in graph.edges():
    u, v = edge
    if part_number[u] != part_number[v]:
      cut += 1

  return cut, part_number


def write_metis_graph_file(graph):
    
  gfile = open("graph.txt", "w")
  n = str(nx.number_of_nodes(graph))
  m = str(nx.number_of_edges(graph))
  out = " ".join([n, m, "\n"])
  gfile.write(out)
  #for u in nx.nodes_iter(graph):
  for u in nx.nodes(graph):
      neighbors = [1 + i for i in nx.neighbors(graph,u)]
      v = str(neighbors).strip('[]')
      v = v.replace(',', '')
      out = " ".join([v, "\n"])
      gfile.write(out)
    
  gfile.close()


def get_metis_edgecut():

  myFile = open("metis_output.out", 'r')
  
  metis_cost = 0
  for lines in myFile:
    lines_split = lines.split()
    if(len(lines_split) > 1):
      if lines_split[1] == "Edgecut:":
        metis_cost = lines_split[2].replace(',', '')
        metis_cost.strip()
        break
    
  return metis_cost


def get_kaffpa_edgecut():

  myFile = open("kaffpa_output.out", 'r')
  
  kaffpa_cut = 0
  for lines in myFile:
    lines_split = lines.split()
    if(len(lines_split) > 1):
      if lines_split[0] == "cut":
        kaffpa_cost = lines_split[1]
        kaffpa_cost.strip()
        break
        
  return kaffpa_cost


#######################################################
########  penalty weight function #####################
#######                            ###################
def set_penalty_constant(num_nodes, num_blocks, beta0, alpha0, gamma0):
 beta = beta0
 alpha = [alpha0 for i in range(num_blocks)] #balancing constraints
 gamma = [gamma0 for i in range(num_nodes)] #each node in one part
 GAMMA = [gamma[i] for j in range(num_blocks) for i in range(num_nodes) ]

 return beta, alpha, gamma, GAMMA
 #########
 

def compare_with_metis_and_kahip(graph, part_number, num_nodes, num_parts, num_blocks, result):

  qubo_cut = compute_cut(graph, part_number, num_nodes, num_parts, num_blocks)

  write_metis_graph_file(graph)
  #os.system("./gpmetis -ufactor=1 graph.txt "+str(num_parts)+"  > metis_output.out") 
  metis_cut = 0
  #metis_cut = get_metis_edgecut()
  #os.system("./kaffpa graph.txt " + "--k=" +str(num_parts)+"  > kaffpa_output.out")

  kaffpa_cut = 0
  #kaffpa_cut = get_kaffpa_edgecut()

  fedges = float(nx.number_of_edges(graph))
  print('\nEdge cut results:')
  print("dwave cut: ", qubo_cut, " fraction: ", float(qubo_cut)/fedges)
  if metis_cut > 0:
    print("metis cut: ",metis_cut, " fraction: ", float(metis_cut)/fedges)
  if kaffpa_cut > 0:
    print("kahip:", kaffpa_cut, " fraction: ", float(kaffpa_cut)/fedges)
  if metis_cut > 0 and qubo_cut > 0:
    print("difference: ", abs(int(float(metis_cut) - float(qubo_cut))), " fraction: ", abs(float(metis_cut)/fedges - float(qubo_cut)/fedges))

  return qubo_cut


def compare_with_metis_and_kahip_qbsolv(graph, bit_string, num_nodes, num_parts, num_blocks):

  qubo_cut, part_number = compute_cut_qbsolv(graph, bit_string, num_nodes, num_parts, num_blocks, result)

  write_metis_graph_file(graph)
  #os.system("./gpmetis -ufactor=1 graph.txt "+str(num_parts)+"  > metis_output.out") 
  metis_cut = 0
  #metis_cut = get_metis_edgecut()

  #os.system("./kaffpa graph.txt " + "--k=" +str(num_parts)+"  > kaffpa_output.out")
  kaffpa_cut = 0
  #kaffpa_cut = get_kaffpa_edgecut()

  fedges = float(nx.number_of_edges(graph))
  print('\nEdge cut results:')
  print("qbsolv cut: ", qubo_cut, " fraction: ", float(qubo_cut)/fedges)
  if metis_cut > 0:
    print("metis cut: ",metis_cut, " fraction: ", float(metis_cut)/fedges)
  if kaffpa_cut > 0:
    print("kahip:", kaffpa_cut, " fraction: ", float(kaffpa_cut)/fedges)
  if metis_cut > 0 and qubo_cut > 0: 
    print("difference: ", abs(int(float(metis_cut) - float(qubo_cut))), " fraction: ", abs(float(metis_cut)/fedges - float(qubo_cut)/fedges)) 
 
  return part_number


def process_solution_qbsolv(graph, num_blocks, num_nodes, num_parts):

  bit_string =  get_qubo_solution()
  print('\nResults from D-Wave:',bit_string)
  print("num non zeros: ", sum([int(i) for i in bit_string]))

  return bit_string

def process_solution(ss, graph, num_blocks, num_nodes, num_parts, result):

  qsol = {}
  for i in range(num_blocks*num_nodes):
    qsol[i] = int(ss[0,i])
    #qsol[i] = int(ss[ii,i])

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

def getEmbedding(qsize, run_profile):

  ksize = qsize
  qsystem = DWaveSampler(profile=run_profile)
  ksub = nx.complete_graph(ksize).edges()
  embedding = minorminer.find_embedding(ksub, qsystem.edgelist)
  print('\nembedding done')

  return embedding


#def runDwave(Q, num_nodes, k, embedding, sub_qsize, run_label, run_profile, result):
def runDwave(Q, num_nodes, k, run_label, run_profile, result):

  # Using D-Wave
  #sampler = DWaveCliqueSampler(profile=run_profile, annealing_time=15)
  sampler = DWaveCliqueSampler(profile=run_profile)
  chain_strength = partial(uniform_torque_compensation, prefactor=2)
  num_reads=1000

  #sampler  = EmbeddingComposite(DWaveSampler(profile=run_profile), embedding)
  #chain_strength = partial(uniform_torque_compensation, prefactor=2)
  #num_reads=1000

  # Run directly on the Dwave
  t0 = dt.datetime.now()
  solution = sampler.sample_qubo(Q, num_reads=num_reads, chain_strength=chain_strength, label=run_label)
  wtime = dt.datetime.now() - t0
  result['wall_clock_time'] = wtime

  # Collect first energy and num_occ, num diff solutions, and total solutions
  first = True
  ndiff = 0
  total_solns = 0
  for sample, energy, num_occurrences, other in solution.data():
    #print(sample, "Energy: ", energy, "Occurrences: ", num_occurrences)
    if first == True:
      result['energy'] = energy
      result['num_occ'] = num_occurrences
      first = False
    ndiff += 1
    total_solns += num_occurrences
  result['num_diff_solns'] = ndiff
  result['total_solns'] = total_solns

  print(solution)
  ss = solution.samples()
  #print('\nss = ', ss)
  print(flush=True)

  return ss

def runDwaveHybrid(Q, num_nodes, k, sub_qsize, run_label, run_profile, result):

  bqm = dimod.BQM.from_qubo(Q)

  rparams = {}
  rparams['label'] = run_label

  # QPU sampler with timing
  QPUSubSamTime = QPUTimeSubproblemAutoEmbeddingSampler(num_reads=100, sampling_params=rparams)

  # define the workflow
  iteration = hybrid.Race(
    hybrid.InterruptableTabuSampler(),
    #hybrid.EnergyImpactDecomposer(size=sub_qsize, rolling=True, rolling_history=0.15)
    #hybrid.EnergyImpactDecomposer(size=sub_qsize, rolling=True, rolling_history=1.00)
    hybrid.EnergyImpactDecomposer(size=sub_qsize, rolling=True, rolling_history=0.30)
    #| hybrid.QPUSubproblemAutoEmbeddingSampler(num_reads=100, sampling_params=rparams)
    | QPUSubSamTime
    | hybrid.SplatComposer()
  ) | hybrid.MergeSamples(aggregate=True)
  workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=3)

  # run the workflow
  init_state = hybrid.State.from_problem(bqm)
  t0 = dt.datetime.now()
  solution = workflow.run(init_state).result()
  wtime = dt.datetime.now() - t0

  result['wall_clock_time'] = wtime

  # Collect number of QPU accesses and QPU time used
  result['num_qpu_accesses'] = QPUSubSamTime.num_accesses
  result['total_qpu_time'] = QPUSubSamTime.total_qpu_time

  # Collect from lowest energy result
  result['energy'] = solution.samples.first.energy
  result['num_occ'] = solution.samples.first.num_occurrences

  # Collect number of different solutions w different energies
  result['num_diff_solns'] = len(solution.samples)

  total_solns = 0
  for energy, num_occ in solution.samples.data(['energy', 'num_occurrences']):
    total_solns += num_occ
  result['total_solns'] = total_solns

  print(solution.samples)
  ss = np.zeros([1,num_nodes])
  for i in range(num_nodes):
    ss[0,i] = solution.samples.first.sample[i]

  return ss

def runDwaveLeapHybrid(Q, num_nodes, k, run_label, run_profile, tlimit, result):

  bqm = dimod.BQM.from_qubo(Q)

  rparams = {}
  rparams['label'] = run_label

  sampler = LeapHybridSampler(profile=run_profile)
  # run the workflow
  t0 = dt.datetime.now()
  if tlimit > 0:
    solution = sampler.sample(bqm, time_limit=tlimit)
  else:
    solution = sampler.sample(bqm)
  wtime = dt.datetime.now() - t0

  result['wall_clock_time'] = wtime

  # Collect QPU time and total time used
  result['run_time'] = solution.info['run_time'] 
  result['qpu_time'] = solution.info['qpu_access_time']

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

  print(solution)
  xx = solution.samples()
  ss = np.zeros([1,num_nodes])
  for i in range(num_nodes):
    ss[0,i] = xx[0,i]

  print('\nss = ', ss)

  return ss 

#def partition(Q, k, embedding, sub_qsize, run_label, run_profile, result):
def partition(Q, k, run_label, run_profile, result):

  # Start with Q
  qsize = Q.shape[1]
  print('\n Q size = ', qsize)

  # Partition into k parts using DWave ocean
  #ss = runDwave(Q, qsize, k, embedding, sub_qsize, run_label, run_profile,result)
  ss = runDwave(Q, qsize, k, run_label, run_profile,result)

  return ss

def partitionHybrid(Q, k, sub_qsize, run_label, run_profile, result):

  # Start with Q
  qsize = Q.shape[1]
  print('\n Q size = ', qsize)

  # Partition into k parts using Hybrid/DWave ocean
  ss = runDwaveHybrid(Q, qsize, k, sub_qsize, run_label, run_profile, result)

  return ss

def partitionLeapHybrid(Q, num_parts, run_label, run_profile, tlimit, result):
 
  # Start with Q
  qsize = Q.shape[1]
  print('\n Q size = ', qsize)

  # Partition into k parts using LeapHybrid/DWave ocean
  ss = runDwaveLeapHybrid(Q, qsize, num_parts, run_label, run_profile, tlimit, result)

  return ss 

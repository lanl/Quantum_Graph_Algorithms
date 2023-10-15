#!/usr/bin/env python

import matplotlib.pyplot as plt
import re, os, sys

import neal
from dwave.system.samplers import DWaveSampler, DWaveCliqueSampler
from dwave.system.composites import FixedEmbeddingComposite, EmbeddingComposite
import dimod
import hybrid
import minorminer
from functools import partial
from dwave.embedding.chain_strength import uniform_torque_compensation
import greedy

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


def makeQubo(laplacian, alpha, beta, gamma, graph, num_nodes, num_parts):

  # Create QUBO matrix
  n = num_nodes
  Q = np.zeros([n,n])
  
  # Create Adjacency matrix A
  A = nx.adjacency_matrix(graph)

  # Calculate degree for each node
  g = np.zeros([n])
  g = A.sum(1)

  for i in range(n):
    for j in range(i, n):
      if i == j:
        entry = beta * g[i] - alpha * (n - 1)
        Q[i,i] = entry
      else:
        if A[i,j] > 0:
          entry = alpha - beta
        else:
          entry = alpha
        Q[i,j] = entry
        Q[j,i] = entry

  return Q


def compute_cut(graph, part_number, num_nodes, num_parts):

  cut = 0
  for edge in graph.edges():
    u, v = edge
    if part_number[u] != part_number[v]:
      cut += 1

  return cut


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


def compare_with_metis_and_kahip(graph, part_number, num_nodes, num_parts, result):

  qubo_cut = compute_cut(graph, part_number, num_nodes, num_parts)

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


def process_solution(ss, graph, num_nodes, num_parts, result):

  # Determine sizes of 2 parts
  cc = np.zeros([num_nodes])
  c0 = 0
  c1 = 0
  for i in range(num_nodes):
    if ss[0,i] == 0:
      c0 = c0 + 1
    else:
      c1 = c1 + 1
      cc[i] = 1

  print('\ndwave 2 parts of size:', c0, c1)

  cldet = [[] for i in range(2)]
  cdet = np.zeros([num_nodes])
  for i in range(num_nodes):
    cdet[i] = i

  cldet[0] = cdet[cc == 0]
  cldet[1] = cdet[cc == 1]

  for i in range(2):
    print('cldet ', i, cldet[i])
  print(flush=True)

  return cc, cldet

def getBQMEmbedding(qsize):
  
  
  bqm = dimod.BQM.from_qubo(Q)

  # Find minor embedding
  sampler = DWaveSampler()
  print(f"\nMinor embedding problem into {sampler.solver.name}")
  embedding = minorminer.find_embedding(
            dimod.to_networkx_graph(bqm), sampler.to_networkx_graph())

  return embedding

def getEmbedding(qsize):

  subqubo_size = qsize 
  qsystem = DWaveSampler()
  kqsize = nx.complete_graph(qsize).edges()
  embedding = minorminer.find_embedding(kqsize, qsystem.edgelist)
  print('\nembedding done')

  return embedding


def runDwave(Q, num_nodes, k, embedding, sub_qsize, run_label, run_profile, result):

  # Using D-Wave/qbsolv
  # Needed when greater than number of nodes/variables that can fit on the D-Wave
  sampler  = FixedEmbeddingComposite(DWaveSampler(profile=run_profile), embedding)
  #sampler  = DWaveCliqueSampler(annealing_time=10, profile=run_profile)
  #sampler  = DWaveCliqueSampler(profile=run_profile)

  rval = random.randint(1,10000)
  t0 = dt.datetime.now()
  #solution_qbs = QBSolv().sample_qubo(Q, solver=sampler, seed=rval,
  #                         label=run_label)
  while solution_qbs.done() == False:
    xx = 0
  wtime = dt.datetime.now() - t0
  result['wall_clock_time'] = wtime

  solution = solution_qbs

  # Run greedy
  #solver_greedy = greedy.SteepestDescentSolver()
  #solution = solver_greedy.sample_qubo(Q, initial_states=solution_qbs)

  # Collect first energy and num_occ, num diff solutions, and total solutions
  first = True
  ndiff = 0
  total_solns = 0
  for sample, energy, num_occurrences in solution.data():
  #for sample, energy, num_occurrences, other in solution.data():
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

def runDwaveHybrid(Q, num_nodes, k, sub_qsize, run_label, run_profile, result):

  bqm = dimod.BQM.from_qubo(Q)

  rparams = {}
  rparams['label'] = run_label

  # QPU sampler with timing
  QPUSubSamTime = QPUTimeSubproblemAutoEmbeddingSampler(num_reads=10000, sampling_params=rparams)

  # define the workflow
  iteration = hybrid.Race(
    hybrid.InterruptableTabuSampler(),
    #hybrid.InterruptableSimulatedAnnealingProblemSampler(),
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


def runDwaveHybrid2(Q, num_nodes, k, sub_qsize, run_label, run_profile, result):

  bqm = dimod.BQM.from_qubo(Q)

  rparams = {}
  rparams['label'] = run_label


  # define the workflow
  t0 = dt.datetime.now()
  solution = hybrid.KerberosSampler(profile=run_profile).sample(bqm, max_iter=10, convergence=3)
  wtime = dt.datetime.now() - t0

  result['wall_clock_time'] = wtime

  print('\nsolution = ', solution)

  # Collect number of QPU accesses and QPU time used
  #result['num_qpu_accesses'] = QPUSubSamTime.num_accesses
  #result['total_qpu_time'] = QPUSubSamTime.total_qpu_time

  # Collect from lowest energy result
  result['energy'] = solution.record.energy[0]
  result['num_occ'] = solution.record.num_occurrences[0]

  # Collect number of different solutions w different energies
  #result['num_diff_solns'] = len(solution.samples)

  total_solns = 1
  #for energy, num_occ in solution.samples.data(['energy', 'num_occurrences']):
  #  total_solns += num_occ
  result['total_solns'] = total_solns

  #print(solution.record.sample)
  ss = np.zeros([1,num_nodes])
  for i in range(num_nodes):
    ss[0,i] = solution.record.sample[0,i]

  return ss

def runDwaveDirect(Q, run_label, run_profile, result):

  # Use D-Wave directly
  atime=20
  #sampler = DWaveCliqueSampler(profile=run_profile, annealing_time=atime)
  sampler = DWaveCliqueSampler(profile=run_profile)
  #sampler = EmbeddingComposite(DWaveSampler(profile=run_profile))
  chain_strength = partial(uniform_torque_compensation, prefactor=2)
  #chain_strength = partial(uniform_torque_compensation, prefactor=1)
  #chain_strength = partial(uniform_torque_compensation, prefactor=3)
  num_reads=1000

  t0 = dt.datetime.now()
  solution_dwave = sampler.sample_qubo(Q, num_reads=num_reads, chain_strength=chain_strength, label=run_label)
  while solution_dwave.done() == False:
    xx = 0
  wtime = dt.datetime.now() - t0
  result['wall_clock_time'] = wtime

  print('\n wall clock time = ', wtime)
  print('\nPercentage of samples with high rates of breaks (> 0.10) is ',
        np.count_nonzero(solution_dwave.record.chain_break_fraction > 0.10)/num_reads*100)

  solution = solution_dwave

  # Run greedy
  #solver_greedy = greedy.SteepestDescentSolver()
  #solution = solver_greedy.sample_qubo(Q, initial_states=solution_dwave)

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

  print('\n dwave direct response:')
  print(solution)
  ss = solution.samples()
  #print("\n dwave samples=" + str(list(solution.samples())))
  #print('\nss = ', ss)
  print(flush=True)

  return ss  


def runSA(Q, num_nodes, result):

  sampler  = neal.SimulatedAnnealingSampler()
  num_reads=1

  t0 = dt.datetime.now()
  response = sampler.sample_qubo(Q, num_reads=num_reads)
  wtime = dt.datetime.now() - t0
  result['wall_clock_time'] = wtime
  print('\n wall clock time = ', wtime)

  # Collect first energy and num_occ, num diff solutions, and total solutions
  first = True
  ndiff = 0
  total_solns = 0
  for sample, energy, num_occurrences in response.data():
    #print(sample, "Energy: ", energy, "Occurrences: ", num_occurrences)
    if first == True:
      result['energy'] = energy
      result['num_occ'] = num_occurrences
      first = False
    ndiff += 1
    total_solns += num_occurrences
  result['num_diff_solns'] = ndiff
  result['total_solns'] = total_solns

  print('\n SA response:')
  print(response)
  ss = response.samples()
  #print("\n qbsolv samples=" + str(list(response.samples())))
  print(flush=True)

  return ss


def partition(Q, k, embedding, sub_qsize, run_label, run_profile, result):

  # Start with Q
  qsize = Q.shape[1]
  print('\n Q size = ', qsize)

  # Partition into 2 parts using DWave ocean/qbsolv
  ss = runDwave(Q, qsize, k, embedding, sub_qsize, run_label, run_profile, result)

  return ss

def partitionHybrid(Q, k, sub_qsize, run_label, run_profile, result):

  # Start with Q
  qsize = Q.shape[1]
  print('\n Q size = ', qsize)

  # Partition into 2 parts using Hybrid/DWave ocean
  ss = runDwaveHybrid(Q, qsize, k, sub_qsize, run_label, run_profile, result)

  return ss

def partitionDirect(Q, run_label, run_profile, result):

  # Start with Q
  qsize = Q.shape[1]
  print('\n Q size = ', qsize)

  # Partition into 2 parts directly on the D-Wave
  ss = runDwaveDirect(Q, run_label, run_profile, result)

  return ss


def partitionSA(Q, num_nodes, result):

  # Start with Q
  qsize = Q.shape[1]
  print('\n Q size = ', qsize)

  # Partition into 2 parts using SA
  ss = runSA(Q, num_nodes, result)

  return ss


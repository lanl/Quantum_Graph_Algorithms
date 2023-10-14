import numpy as np
import numpy.linalg as la
import networkx as nx

import neal
#from dwave_qbsolv import QBSolv
from dwave.system.samplers import DWaveSampler, DWaveCliqueSampler
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
from dimod.reference.samplers import ExactSolver
import dimod
import hybrid
import minorminer
from functools import partial
from dwave.embedding.chain_strength import uniform_torque_compensation

from scipy.io import mmread
import matplotlib.pyplot as plt
import argparse
import logging
import os
import sys
import random
import datetime as dt

import graphFileUtility_functions as GFU
from qpu_sampler_time import QPUTimeSubproblemAutoEmbeddingSampler

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

def buildMod(Adj, thresh):
        """Builds the modularity matrix from the Adjacency matrix.

        Given an adj matrix, it constructs the modularity matrix and its graph.

        """

        Dim = Adj.shape[1]
        #print ("\n Dim = ", Dim)

        #print ("\n Computing modularity matrix ...")

        Deg = np.zeros((Dim))

        M = 0.0

        #print ("\n Adj degrees: ")
        for ii in range(Dim):
          for jj in range(Dim):
            if ii != jj:
              Deg[ii] = Deg[ii] + Adj[ii, jj]

          M = M + Deg[ii]

          #print ii,Deg[ii]
          mtotal = M/2.0

        #print ("\n Number of edges:", M)

        Mod = np.zeros([Dim,Dim])

        # Calc modularity matrix
        for ii in range(Dim):
          for jj in range(Dim):
            Mod[ii,jj] = Adj[ii,jj] - (Deg[ii]*Deg[jj])/M
            if(ii == jj):
              Mod[ii,jj] = - (Deg[ii]*Deg[jj])/M

        # Calc modularity degrees
        DegM = np.zeros([Dim])
        #print ("\n Mod degrees: ")
        for ii in range(Dim):
          for jj in range(Dim):
           if ii != jj:
              DegM[ii] = DegM[ii] + Mod[ii,jj]
              M = M + DegM[ii]
              #print ii,DegM[ii]

        # Threshold mod
        nonzeros = 0
        no_edges = 0
        for ii in range(Dim):
          for jj in range(Dim):
            if ii != jj:
              if abs(Mod[ii,jj]) > thresh:
                nonzeros =  nonzeros + 1
              else:
                Mod[ii,jj] = 0.0

        #print ('\n Mod matrix: ', '\n', Mod)

        return mtotal, Mod


def makeQubo(C):
  # Create QUBO formulation
  num_nodes = C.shape[1]

  # Create QUBO from modularity matrix
  Q = {}
  for i in range(num_nodes):
    Q[i,i] = -C[i,i]
    for j in range(i+1,num_nodes):
      if i != j:
        Q[i,j] = -C[i,j]
        Q[j,i] = -C[j,i]

  #print('\nQ = \n', Q)

  return Q


def qubo2file(qmatrix, qsize):

  nElements = 0
  maxDiagonals = qsize
  nDiagonals = qsize

  qubo_file = open("body.qubo", "w")

  # Write node header
  qubo_string_diag = "".join(["\nc nodes first \n"])
  qubo_file.write(qubo_string_diag)

  # Write diag elements
  for i in range(qsize):
    entry = 0.0
    if qmatrix.get((i,i)):
      entry = -qmatrix[i,i]
    qubo_string_diag = "".join([str(i)+" "+str(i)+" "+str(entry)+"\n"])
    qubo_file.write(qubo_string_diag)

  # Write coupler header
  qubo_string_couplers  = "".join(["\nc couplers \n"])
  qubo_file.write(qubo_string_couplers)

  # Write couplers
  for i in range(qsize):
    for j in range(i, qsize):
      if i != j:
         if qmatrix.get((i,j)):
           entry = -qmatrix[i,j]
           qubo_string_couplers = "".join([str(i)+" "+str(j)+" "+str(entry)+"\n"])
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
  qsol = {}
  line_count = 0

  for lines in myFile:
    line_count += 1
    if line_count == 2:
      bit_string = lines
      bit_string =  bit_string.strip()
      for i in range(len(bit_string)):
        qsol[i] = int(bit_string[i])
    if line_count == 3:
      x = lines.split()
      energy = float(x[0])
      break

  return qsol, energy


def getEmbedding(qsize):

  qsystem = DWaveSampler()
  ksize = nx.complete_graph(qsize).edges()
  embedding = minorminer.find_embedding(ksize, qsystem.edgelist)
  print('\nembedding done')

  return embedding


def runDwave(Q, embedding, run_label, run_profile, result):

  sampler  = FixedEmbeddingComposite(DWaveSampler(), embedding)

  t0 = dt.datetime.now()
  #response = QBSolv().sample_qubo(Q, solver=sampler, label=run_label)
  wtime = dt.datetime.now() - t0
  print('\n wall clock time = ', wtime)
  result['wall_clock_time'] = wtime

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

  print('\n qbsolv response:')
  print(response)
  ss = response.samples()
  #print("\n qbsolv samples=" + str(list(response.samples())))
  print(flush=True)
  
  return ss
 
  
def runDwaveDirect(Q, run_label, run_profile, result):

  # Use D-Wave directly
  sampler = DWaveCliqueSampler(profile=run_profile)
  chain_strength = partial(uniform_torque_compensation, prefactor=2)
  num_reads=1000

  t0 = dt.datetime.now()
  solution = sampler.sample_qubo(Q, num_reads=num_reads, chain_strength=chain_strength, label=run_label)
  wtime = dt.datetime.now() - t0
  result['wall_clock_time'] = wtime

  print('\n wall clock time = ', wtime)
  print('\nPercentage of samples with high rates of breaks (> 0.10) is ',
        np.count_nonzero(solution.record.chain_break_fraction > 0.10)/num_reads*100)

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


def process_solution(ss, num_nodes):

  # Determine sizes of 2 clusters
  cc = np.zeros([num_nodes])
  cc2 = {}
  c0 = 0
  c1 = 0
  for i in range(num_nodes):
    if ss[0,i] == 0:
      c0 = c0 + 1
      cc[i] = 0
      cc2[i] = 0
    else:
      c1 = c1 + 1
      cc[i] = 1
      cc2[i] = 1

  print('\n2 clusters of size:', c0, c1)

  cldet = [[] for i in range(2)]
  cdet = np.zeros([num_nodes])
  for i in range(num_nodes):
    cdet[i] = i

  cldet[0] = cdet[cc == 0]
  cldet[1] = cdet[cc == 1]
  for i in range(2):
    print('cldet ', i, cldet[i])
  print(flush=True)

  return cc2, cldet


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


def runDwaveHybrid(Q, num_nodes, sub_qsize, run_label, run_profile, result):

  bqm = dimod.BQM.from_qubo(Q)

  rparams = {}
  rparams['label'] = run_label

  # QPU sampler with timing
  QPUSubSamTime = QPUTimeSubproblemAutoEmbeddingSampler(num_reads=100, sampling_params=rparams)

  # define the workflow
  iteration = hybrid.Race(
    hybrid.InterruptableTabuSampler(),
    #hybrid.EnergyImpactDecomposer(size=sub_qsize, rolling=True, rolling_history=0.15)
    hybrid.EnergyImpactDecomposer(size=sub_qsize, rolling=True, rolling_history=1.00)
    #| hybrid.QPUSubproblemAutoEmbeddingSampler(num_reads=100, profile=run_profile, sampling_params=rparams)
    | QPUSubSamTime
    | hybrid.SplatComposer()
  ) | hybrid.MergeSamples(aggregate=True)
  workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=3)

  # run the workflow
  init_state = hybrid.State.from_problem(bqm)
  t0 = dt.datetime.now()
  solution = workflow.run(init_state).result()
  wtime = dt.datetime.now() - t0
  print('\n wall clock time = ', wtime)

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


def makeAdj(H):

  hsize = H.shape[1]
  A = np.zeros([hsize,hsize])
  for i in range(hsize):
    for j in range(hsize):
      if i != j:
        A[i,j] = abs(H[i,j])

  #print('\nA = ', '\n', A)

  return A


def lowEnergy(cldet, H):

  clind = 0
  lowe = np.zeros([2])
  for i in range(2):
    e, v = la.eigh(H[cldet[i],:][:,cldet[i]])
    lowe[i] = e[0]
    print('cluster energy:', i, lowe[i])
    if lowe[i] < lowe[clind]:
      clind = i

  print('clind = ', clind)
  print(flush=True)

  return clind


def cluster(B, embedding, run_label, run_profile, result):

  # Create the qubo from the modularity matrix
  Q = makeQubo(B)

  # Cluster into 2 parts using DWave
  cldet = runDwave(Q, embedding, run_label, run_profile, result)

  # Determine which of the 2 clusters has the lower energy
  #clind = lowEnergy(cldet, H)

  return cldet


def clusterSA(B, num_nodes, result):

  # Start with B
  bsize = B.shape[1]
  print('\n B size = ', bsize)

  # Create the qubo from the modularity matrix
  Q = makeQubo(B)

  # Cluster into 2 parts usingSA 
  cldet = runSA(Q, num_nodes, result)

  # Determine which of the 2 clusters has the lower energy
  #clind = lowEnergy(cldet, H)

  return cldet


def clusterHybrid(B, num_nodes, qsize, run_label, run_profile, result):

  # Create the qubo from the modularity matrix
  Q = makeQubo(B)

  # Cluster into 2 parts using DWave
  cldet = runDwaveHybrid(Q, num_nodes, qsize, run_label, run_profile, result)

  # Determine which of the 2 clusters has the lower energy
  #clind = lowEnergy(cldet, H)

  return cldet 


def clusterDirect(B, run_label, run_profile, result):

  # Create the qubo from the modularity matrix
  Q = makeQubo(B)

  # Cluster into 2 parts using DWave
  cldet = runDwaveDirect(Q, run_label, run_profile, result)

  # Determine which of the 2 clusters has the lower energy
  #clind = lowEnergy(cldet, H)

  return cldet


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

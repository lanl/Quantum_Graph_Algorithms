import numpy as np
import numpy.linalg as la
import networkx as nx

from dwave_qbsolv import QBSolv
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
from dimod.reference.samplers import ExactSolver
import dimod
import hybrid
import minorminer

from scipy.io import mmread
import matplotlib.pyplot as plt
import argparse
import logging
import os
import sys
import random


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
        for ii in range(0,Dim):
          for jj in range(0,Dim):
            if(ii != jj):
              Deg[ii] = Deg[ii] + Adj[ii, jj]
          M = M + Deg[ii]
              #M = M + 1
          #print ii,Deg[ii]
          mtotal = M/2.0

        #print ("\n Number of edges:", M)

        Mod = np.zeros((Dim,Dim))

        # Calc modularity matrix
        for ii in range(0,Dim):
          for jj in range(0,Dim):
            Mod[ii,jj] = Adj[ii,jj] - (Deg[ii]*Deg[jj])/M
            if(ii == jj):
              Mod[ii,jj] = - (Deg[ii]*Deg[jj])/M

        # Calc modularity degrees
        DegM = np.zeros((Dim))
        #print ("\n Mod degrees: ")
        for ii in range(0,Dim):
          for jj in range(0,Dim):
           if(ii != jj):
              DegM[ii] = DegM[ii] + Mod[ii,jj]
              M = M + DegM[ii]
              #print ii,DegM[ii]

        # Threshold mod
        nonzeros = 0
        no_edges = 0
        for ii in range(0,Dim):
          for jj in range(0,Dim):
            if ii != jj:
              if(abs(Mod[ii,jj]) > thresh):
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


def getEmbedding():

  subqubo_size = 64
  qsystem = DWaveSampler()
  k64 = nx.complete_graph(64).edges()
  embedding = minorminer.find_embedding(k64, qsystem.edgelist)
  print('\nembedding done')

  return embedding


def runDwave(Q, num_nodes, use_dwave, embedding, qsize, run_label):

  if use_dwave == 1 or use_dwave == 2:
    # Using D-Wave/qbsolv
    # Needed when greater than the number of nodes/variables that fit on D-Wave
    if use_dwave == 1:
      sampler = EmbeddingComposite(DWaveSampler())
    else:
      sampler  = FixedEmbeddingComposite(DWaveSampler(), embedding)

    subqubo_size = qsize
    response = QBSolv().sample_qubo(Q, solver=sampler,
                                       label=run_label)

    print('\n qbsolv response:')
    print(response)
    ss = response.samples()
    #print("\n qbsolv samples=" + str(list(response.samples())))
    print(flush=True)
 
    # Determine sizes of 2 clusters
    cc = np.zeros([num_nodes])
    c0 = 0
    c1 = 0
    for i in range(num_nodes):
      if ss[0,i] == 0:
        c0 = c0 + 1
      else:
        c1 = c1 + 1
        cc[i] = 1
 
    print('\ndwave  2 clusters of size:', c0, c1)
    print('\ncc = ', cc)
  
  elif use_dwave == 3:
    # Use command-line qbsolv+D-Wave
    qubo2file(Q, num_nodes)
    rval = random.randint(1,1000)
    estring = "qbsolv -r " + str(rval) + " -i graph.qubo -o dwave_output.out"
    print(estring)
    os.system(estring)
    ss, qmin = get_qubo_solution()
    #print('\nss = ', ss)
    #print('\nqmin = ', qmin)
    print(flush=True)

    # Determine sizes of 2 clusters
    cc = np.zeros([num_nodes])
    c0 = 0
    c1 = 0
    for i in range(num_nodes):
      if ss[i] == 0:
        c0 = c0 + 1
      else:
        c1 = c1 + 1
        cc[i] = 1

    print('\ndwave  2 clusters of size:', c0, c1)

  cldet = [[] for i in range(2)]
  cdet = np.zeros([num_nodes])
  for i in range(num_nodes):
    cdet[i] = i

  cldet[0] = cdet[cc == 0]
  cldet[1] = cdet[cc == 1]
  print('\ndwave  2 clusters of size:', c0, c1)
  for i in range(2):
    print('cldet ', i, cldet[i])
  print(flush=True)

  return cldet


def runDwaveHybrid(Q, num_nodes, use_dwave, sub_qsize, run_label):

  if use_dwave == 3:
    print('\nThe hybrid version of 2-clustering does not work with option 3')
    exit(1)

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

  # run the workflow
  init_state = hybrid.State.from_problem(bqm)
  solution = workflow.run(init_state).result()

  print(solution.samples)
  ss = np.zeros([1,num_nodes])
  for i in range(num_nodes):
    ss[0,i] = solution.samples.first.sample[i]

  # Determine sizes of 2 clusters
  cc = np.zeros([num_nodes])
  c0 = 0
  c1 = 0
  for i in range(num_nodes):
    if ss[0,i] == 0:
      c0 = c0 + 1
    else:
      c1 = c1 + 1
      cc[i] = 1

  print('\ndwave  2 clusters of size:', c0, c1)
  print('\ncc = ', cc)

  cldet = [[] for i in range(2)]
  cdet = np.zeros([num_nodes])
  for i in range(num_nodes):
    cdet[i] = i

  cldet[0] = cdet[cc == 0]
  cldet[1] = cdet[cc == 1]
  print('\ndwave  2 clusters of size:', c0, c1)
  for i in range(2):
    print('cldet ', i, cldet[i])
  print(flush=True)

  return cldet


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


def cluster(B, use_dwave, embedding, qsize, run_label):

  # Start with B
  bsize = B.shape[1]
  print('\n B size = ', bsize)

  # Create the qubo from the modularity matrix
  Q = makeQubo(B)

  # Cluster into 2 parts using DWave
  cldet = runDwave(Q, bsize, use_dwave, embedding, qsize, run_label)

  # Determine which of the 2 clusters has the lower energy
  #clind = lowEnergy(cldet, H)

  return cldet


def clusterHybrid(B, use_dwave, qsize, run_label):

  # Start with B
  bsize = B.shape[1]
  print('\n B size = ', bsize)

  # Create the qubo from the modularity matrix
  Q = makeQubo(B)

  # Cluster into 2 parts using DWave
  cldet = runDwaveHybrid(Q, bsize, use_dwave, qsize, run_label)

  # Determine which of the 2 clusters has the lower energy
  #clind = lowEnergy(cldet, H)

  return cldet


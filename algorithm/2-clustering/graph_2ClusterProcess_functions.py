import numpy as np
import numpy.linalg as la
import networkx as nx

from scipy.io import mmread
import matplotlib.pyplot as plt
import argparse
import logging
import os
import sys
import random
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
  Q = np.zeros((num_nodes, num_nodes))
  for i in range(num_nodes):
    Q[i,i] = -C[i,i]
    for j in range(i+1,num_nodes):
      if i != j:
        Q[i,j] = -C[i,j]
        Q[j,i] = -C[j,i]

  #print('\nQ = \n', Q)

  return Q


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

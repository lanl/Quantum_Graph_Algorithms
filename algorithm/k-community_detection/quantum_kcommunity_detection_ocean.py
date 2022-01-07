#!/usr/bin/env python

# K_modularity using weighted edges

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
import urllib3

import graph_kClusterAlgorithm_functions as QCD
import graphFileUtility_functions as GFU

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


if __name__== '__main__':

  urllib3.disable_warnings()

  parser = argparse.ArgumentParser(description='Quantum Community Detection')
  parser.add_argument('-nparts', type=int, default=2, help='number of parts')
  parser.add_argument('-pflag', type=int, default=0, help='plot flag, 0-no 1-yes')
  parser.add_argument('-ifile', help='input filename')
  parser.add_argument('-ftype', default='mtx', help='input file type (mtx, umtx, zmtx, gml, konect, net, mi')
  parser.add_argument('-beta', type=int, default=1, help='beta penalty constant: minimize edge cut')
  parser.add_argument('-gamma', type=int, default=-5, help='gamma penalty constant: each node in 1 part')
  parser.add_argument('-threshold', type=float, default=0.00, help='threshold value')
  parser.add_argument('-label', default='qcd_qbsolv', help='label for run')
  parser.add_argument('-qsize', type=int, default=64, help='qbsolv sub-qubo size')

  args = parser.parse_args()

  print('number parts = ', args.nparts)
  print('plot flag = ', args.pflag)
  print('input file = ', args.ifile)
  print('file type = ', args.ftype)
  print('beta = ', args.beta)
  print('gamma = ', args.gamma)
  print('threshold = ', args.threshold)
  print('label = ', args.label)
  print('qsize = ', args.qsize)
  print('\n')

  num_parts = args.nparts
  pflag = args.pflag
  ifilename = args.ifile
  ftype = args.ftype
  beta0 = args.beta
  gamma0 = args.gamma
  threshold = args.threshold
  run_label = args.label
  qsize = args.qsize

  ####
  # NOTE: node and matrix indexing starts from 0
  #
  ###

  # Read and generate graph
  graph = GFU.generateGraph(ftype, ifilename, threshold)

  A = nx.adjacency_matrix(graph)
  print ('\nAdjacency matrix:\n', A.todense())

  num_blocks = num_parts 
  num_nodes = nx.number_of_nodes(graph)
  num_edges = nx.number_of_edges(graph)
  print ("\n\t Quantum Community Detection: up to %d communities...\n" %num_parts)
  print ("Graph has %d nodes and %d edges" %(num_nodes, num_edges))

  # Collect results to dictionary
  result = {}
  result['alg'] = 'LANL_QCD'
  result['num_clusters'] = num_parts
  result['name'] = ifilename
  result['nodes'] = num_nodes
  result['edges'] = num_edges
  result['size'] = num_nodes * num_parts
  result['run_arch'] = 'DWAVE_Ocean_Qbsolv'
  result['subqubo_size'] = qsize

  beta, gamma, GAMMA  = QCD.set_penalty_constant(num_nodes, num_blocks, beta0, gamma0)

  mtotal,modularity = QCD.build_mod(A, threshold, num_edges)

  print ("\nModularity matrix: \n", modularity)
  
  print ("min value = ", modularity.min())
  print ("max value = ", modularity.max())
  
  print ("threshold = ", threshold)

  Q = QCD.makeQubo(graph, modularity, beta, gamma, GAMMA, num_nodes, num_parts, num_blocks, threshold)

  # Create embedding for D-Wave
  embedding = QCD.getEmbedding(qsize)

  # Run k-clustering with qbsolv/D-Wave using ocean
  ss = QCD.cluster(Q, num_parts, embedding, qsize, run_label, result)

  # Process solution
  part_number = QCD.process_solution(ss, graph, num_blocks, num_nodes, num_parts, result)

  mmetric = QCD.calcModularityMetric(mtotal, modularity, part_number)
  print ("\nModularity metric = ", mmetric)
  result['modularity_metric'] = mmetric

  GFU.write_partFile(part_number, num_nodes, num_parts) 

  GFU.write_resultFile(result)

  if pflag == 1:
    GFU.showClusters(part_number, graph)

  exit(0)

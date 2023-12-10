import numpy as np
import numpy.linalg as la
import networkx as nx

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
from dimod.reference.samplers import ExactSolver
import minorminer

from scipy.io import mmread
import matplotlib.pyplot as plt
import argparse
import logging
import os
import sys
import random
import urllib3

import graph_2ClusterAlgorithm_functions as QCD
import graphFileUtility_functions as GFU


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

  parser = argparse.ArgumentParser(description='Quantum Community Detection 2-clustering - hybrid workflow')
  parser.add_argument('-ifile', help='input filename')
  parser.add_argument('-ftype', default='umtx', help='input file type (mtx, umtx, 0mtx, mi)')
  parser.add_argument('-pflag', type=int, default=0, help='plot flag, 0-no 1-yes')
  parser.add_argument('-nparts', type=int, default=2, help='number of parts')
  parser.add_argument('-label', default='q2cd_hybrid', help='label for run')
  parser.add_argument('-qsize', type=int, default=64, help='hybrid sub-qubo size')
  parser.add_argument('-myprofile', default='', help='D-Wave profile for run')

  args = parser.parse_args()

  print('input file  = ', args.ifile)
  print('file type = ', args.ftype)
  print('plot flag = ', args.pflag)
  print('number parts = ', args.nparts)
  print('label = ', args.label)
  print('qsize = ', args.qsize)
  print('myprofile = ', args.myprofile)

  ifile = args.ifile
  ftype = args.ftype
  pflag = args.pflag
  nparts = args.nparts
  run_label = args.label
  qsize = args.qsize
  run_profile = args.myprofile
 
  # Read in file as graph
  threshold = 0.0
  graph = GFU.createGraph(ftype, ifile, threshold) 
  
  num_nodes = nx.number_of_nodes(graph)
  num_edges = nx.number_of_edges(graph)
  print ("\n\t community detection: up to %d communities...\n" %nparts)
  print ("Graph has %d nodes and %d edges" %(num_nodes, num_edges))

  # Collect results to dictionary
  result = {}
  result['alg'] = 'LANL_2CD'
  result['num_parts'] = nparts
  result['dataset'] = ifile
  result['nodes'] = num_nodes
  result['edges'] = num_edges
  result['size'] = num_nodes
  result['solver'] = 'DWAVE_HYBRID'
  result['subqubo_size'] = qsize

  # Create Adjacency matrix A
  A = nx.adjacency_matrix(graph)
  print ('\nAdjacency matrix:\n', A.todense())

  # Calculate Modularity matrix B
  mtotal,B = QCD.buildMod(A, threshold)
  print('\nModularity matrix:\n', B)

  # Cluster into 2 parts
  ss = QCD.clusterHybrid(B, num_nodes, qsize, run_label, run_profile,  result)

  # Process solution
  part_number,cdet = QCD.process_solution(ss, num_nodes)

  # Calculate modularity meetric
  mmetric = QCD.calcModularityMetric(mtotal, B, part_number)
  print('\nmodularity metric = ', mmetric)
  result['modularity_metric'] = mmetric
  mcut = QCD.calc_cut(graph, part_number)
  print('\nMin cut = ', mcut)
  result['minimum_cut'] = mcut
  result['num_clusters_found'] = nparts
  print(flush=True)

  GFU.write_partFile(part_number, num_nodes, nparts)

  GFU.write_resultFile(result)

  # Show plot of parts if requested
  if pflag == 1:
    GFU.showClusters(part_number, graph)

  exit(0)

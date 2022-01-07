#!/usr/bin/env python

import matplotlib.pyplot as plt
import re, os, sys
import numpy as np

from dwave_qbsolv import QBSolv
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
import minorminer

import networkx as nx
from numpy import linalg as la
from networkx.generators.atlas import *
import random, copy
import math
import argparse
from scipy.sparse import csr_matrix
import urllib3

import graph_kPartitionAlgorithm_functions as QGP
import graphFileUtility_functions as GFU

#
# The Quantum Graph Partitioning Algorithm has been described
# in the following publications. Please cite in your publication.
#
# H. Ushijima-Mwesigwa, C. F. A. Negre, S. M. Mniszewski,
# 2017, Graph Partitioning using Quantum Annealing on the
# D-Wave System, Proceedings of the 2nd International
# Workshop on Post Moore’s Era Supercomputing (PMES), 22-29.
#


if __name__== '__main__':

  urllib3.disable_warnings()

  parser = argparse.ArgumentParser(description='Quantum Graph Partitioning - Hybrid')
  parser.add_argument('-nparts', type=int, default=2, help='number of parts')
  parser.add_argument('-pflag', type =int, default=0, help='0 - no plot, 1 - show plot')
  parser.add_argument('-ifile', help='input filename in mtx format')
  parser.add_argument('-beta', type=int, default=1, help='beta penalty constant: minimize edge cut')
  parser.add_argument('-alpha', type=int, default=1000, help='alpha penalty constant: balancing')
  parser.add_argument('-gamma', type=int, default=5000, help='gamma penalty constant: each node in 1 part')
  parser.add_argument('-label', default='qgp_hybrid', help='label for run')
  parser.add_argument('-qsize', type=int, default=64, help='QPU sub-qubo size')

  args = parser.parse_args()
 
  print('number parts = ', args.nparts)
  print('plot flag = ', args.pflag)
  print('mtx file = ', args.ifile)
  print('beta = ', args.beta)
  print('alpha = ', args.alpha)
  print('gamma = ', args.gamma)
  print('label = ', args.label)
  print('qsize = ', args.qsize)
  print('\n')

  num_parts = args.nparts
  pflag = args.pflag
  ifilename = args.ifile
  beta0 = args.beta
  alpha0 = args.alpha
  gamma0 = args.gamma
  run_label = args.label
  qsize = args.qsize

  ####
  # NOTE: node and matrix indexing starts from 0
  #
  ###

  # Read in graph from mtx file
  graph = nx.Graph() 
  graph = GFU.read_graph_file(graph, ifilename)

  num_blocks = num_parts 
  num_nodes = nx.number_of_nodes(graph)
  num_edges = nx.number_of_edges(graph)
  print("\n\t Partitioning into %d parts...\n" %num_parts)
  print("Graph has %d nodes and %d edges" %(num_nodes, nx.number_of_edges(graph)))

  # Collect results to dictionary
  result = {}
  result['alg'] = 'LANL_QGP'
  result['num_parts'] = num_parts
  result['name'] = ifilename
  result['nodes'] = num_nodes
  result['edges'] = num_edges
  result['size'] = num_nodes * num_parts
  result['run_arch'] = 'DWAVE_Hybrid'
  result['subqubo_size'] = qsize

  # Set penalty constants
  beta, alpha, gamma, GAMMA  = QGP.set_penalty_constant(num_nodes, num_blocks, beta0, alpha0, gamma0)

  # Create and write QUBO matrix to file
  laplacian = nx.laplacian_matrix(graph)
  Q = QGP.makeQubo(laplacian, alpha, beta, gamma, GAMMA, graph, num_nodes, num_parts, num_blocks)
  
  # Run k-partitioning with Hybrid/D-Wave using ocean
  ss = QGP.partitionHybrid(Q, num_parts, qsize, run_label, result)

  # Process solution
  part_number = QGP.process_solution(ss, graph, num_blocks, num_nodes, num_parts, result)

  GFU.write_partfile(graph, part_number, num_nodes, num_parts)

  # Get results and compare to other tools (if available)
  min_cut = QGP.compare_with_metis_and_kahip_ocean(graph, part_number, num_nodes, num_parts, num_blocks, result)
  result['min_cut_metric'] = min_cut

  GFU.write_resultFile(result)

  # Show plot of clusters if requested
  if pflag == 1:
    GFU.show_partitions(graph, part_number)

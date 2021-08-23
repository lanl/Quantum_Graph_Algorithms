#!/usr/bin/env python

import matplotlib.pyplot as plt
import re, os, sys
import numpy as np
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
  
  parser = argparse.ArgumentParser(description='Quantum Graph Partitioning')
  parser.add_argument('-nparts', type=int, default=2, help='number of parts')
  parser.add_argument('-pflag', type =int, default=0, help='0 - no plot, 1 - show plot')
  parser.add_argument('-ifile', help='input filename in mtx format')
  parser.add_argument('-beta', type=int, default=1, help='beta penalty constant: minimize edge cut')
  parser.add_argument('-alpha', type=int, default=1000, help='alpha penalty constant: balancing')
  parser.add_argument('-gamma', type=int, default=5000, help='gamma penalty constant: each node in 1 part')

  args = parser.parse_args()
 
  print('number parts = ', args.nparts)
  print('plot flag = ', args.pflag)
  print('mtx file = ', args.ifile)
  print('beta = ', args.beta)
  print('alpha = ', args.alpha)
  print('gamma = ', args.gamma)
  print('\n')

  num_parts = args.nparts
  pflag = args.pflag
  ifilename = args.ifile
  beta0 = args.beta
  alpha0 = args.alpha
  gamma0 = args.gamma

  ####
  # NOTE: node and matrix indexing starts from 0
  #
  ###

  # Read in graph from mtx file
  graph = nx.Graph() 
  graph = GFU.read_graph_file(graph, ifilename)

  num_blocks = num_parts 
  num_nodes = nx.number_of_nodes(graph)
  print("\n\t Partitioning into %d parts...\n" %num_parts)
  print("Graph has %d nodes and %d edges" %(num_nodes, nx.number_of_edges(graph)))

  # Set penalty constants
  beta, alpha, gamma, GAMMA  = QGP.set_penalty_constant(num_nodes, num_blocks, beta0, alpha0, gamma0)

  # Create and write QUBO matrix to file
  laplacian = nx.laplacian_matrix(graph)
  QGP.write_qubo_file(laplacian, alpha, beta, gamma, GAMMA, graph, num_nodes, num_parts, num_blocks)

  # Run using qbsolv/D-Wave
  QGP.run_qbsolv()

  # Process results
  bit_string = QGP.process_solution_qbsolv(graph, num_blocks, num_nodes, num_parts)

  # Get results and compare to other tools (if available)
  part_number = QGP.compare_with_metis_and_kahip(graph, bit_string, num_nodes, num_parts, num_blocks)

  # Show plot of clusters if requested
  if pflag == 1:
    GFU.show_partitions(graph, part_number)

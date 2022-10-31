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

import graph_2PartitionAlgorithm_functions as QGP
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

  parser = argparse.ArgumentParser(description='Quantum Graph 2-Partitioning - ocean/qbsolv')
  parser.add_argument('-nparts', type=int, default=2, help='number of parts')
  parser.add_argument('-pflag', type =int, default=0, help='0 - no plot, 1 - show plot')
  parser.add_argument('-ifile', help='input filename in mtx format')
  parser.add_argument('-ftype', default='mtx', help='file type: mtx, 0mtx')
  parser.add_argument('-beta', type=int, default=1, help='beta penalty constant: minimize edge cut')
  parser.add_argument('-alpha', type=int, default=1000, help='alpha penalty constant: balancing')
  parser.add_argument('-gamma', type=int, default=5000, help='gamma penalty constant: each node in 1 part')
  parser.add_argument('-label', default='q2gp_qbsolv', help='label for run')
  parser.add_argument('-qsize', type=int, default=64, help='qbsolv sub-qubo size')

  args = parser.parse_args()
 
  print('number parts = ', args.nparts)
  print('plot flag = ', args.pflag)
  print('mtx file = ', args.ifile)
  print('file type = ', args.ftype)
  print('beta = ', args.beta)
  print('alpha = ', args.alpha)
  print('gamma = ', args.gamma)
  print('label = ', args.label)
  print('qsize = ', args.qsize)
  print('\n')

  num_parts = args.nparts
  pflag = args.pflag
  ifilename = args.ifile
  ftype = args.ftype
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
  threshold = 0.0
  graph = GFU.createGraph(ftype, ifilename, threshold)

  num_parts = 2
  num_nodes = nx.number_of_nodes(graph)
  num_edges = nx.number_of_edges(graph)
  print("\n\t Partitioning into %d parts...\n" %num_parts)
  print("Graph has %d nodes and %d edges" %(num_nodes, nx.number_of_edges(graph)))
  print(flush=True)

  # Collect results to dictionary
  result = {}
  result['alg'] = 'LANL_2GP'
  result['num_parts'] = num_parts
  result['dataset'] = ifilename
  result['nodes'] = num_nodes
  result['edges'] = num_edges
  result['size'] = num_nodes
  result['solver'] = 'DWAVE_Ocean_Qbsolv'
  result['subqubo_size'] = qsize

  # Set penalty constants
  alpha = alpha0
  beta = beta0
  gamma = gamma0

  # Create QUBO matrix 
  laplacian = nx.laplacian_matrix(graph)
  Q = QGP.makeQubo(laplacian, alpha, beta, gamma, graph, num_nodes, num_parts)
  print('QUBO created')
  print(flush=True)
  
  # Create embedding for D-Wave
  embedding = QGP.getEmbedding(qsize)
  #embedding = {}
  print('Embedding done')
  print(flush=True)

  # Run 2-partitioning with qbsolv/D-Wave using ocean
  ss = QGP.partition(Q, num_parts, embedding, qsize, run_label, result)
  print('Partitioning done')
  print(flush=True)

  # Process solution
  part_number,cdet = QGP.process_solution(ss, graph, num_nodes, num_parts, result)
  print('Postprocessing done')
  print(flush=True)

  GFU.write_partfile(graph, part_number, num_nodes, num_parts)

  # Get results and compare to other tools (if available)
  min_cut = QGP.compare_with_metis_and_kahip(graph, part_number, num_nodes, num_parts, result)
  result['min_cut_metric'] = min_cut

  GFU.write_resultFile(result)

  # Show plot of parts if requested
  if pflag == 1:
    GFU.show_partitions(graph, part_number)

import numpy as np
import numpy.linalg as la
import networkx as nx

#from dwave_qbsolv import QBSolv
#from dwave.system.samplers import DWaveSampler
#from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
#from dimod.reference.samplers import ExactSolver
#import minorminer

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

def read_graph_file(G, prot_file, threshold):

  gfile = open(prot_file, "r")
  line = gfile.readline()
  line = gfile.readline()
  x = line.split()
  n = int(x[0])
  nedges = int(x[2])
  print ("graph ", n, " nodes ", nedges, " non-zeroes")
  nedges2 = (nedges - n)
  print ("nedges = ", nedges2)

  # Add all nodes
  for i in range(n):
    G.add_node(i)

  # Add all edges
  for i in range(nedges):
    line = gfile.readline()
    x = line.split()
    n0 = int(x[0]) - 1
    n1 = int(x[1]) - 1
    eweight = abs(float(x[2]))
    if n0 != n1:
      if abs(eweight) > threshold:
        G.add_edge(n0,n1,weight=eweight)

  gfile.close()

  print ("graph size =", G.size())

  return G


def read_graph_file_unweighted(G, data_file):

  gfile = open(data_file, "r")
  line = gfile.readline()
  line = gfile.readline()
  x = line.split()
  n = int(x[0])
  nedges = int(x[2])
  print ("graph ", n, " nodes ", nedges, " non-zeroes")
  nedges2 = (nedges - n)
  print ("nedges = ", nedges2)

  # Add all nodes
  for i in range(n):
    G.add_node(i)

  # Add all edges
  for i in range(nedges):
    line = gfile.readline()
    x = line.split()
    n0 = int(x[0]) - 1
    n1 = int(x[1]) - 1
    if n0 != n1:
      G.add_edge(n0,n1)

  gfile.close()

  print ("graph size =", G.size())

  return G


def read_mi_file(G, mfile, threshold):
  gfile = open(mfile, "r")
  line = gfile.readline()
  line = gfile.readline()
  x = line.split()
  n = int(x[0])
  nedges = int(x[2])
  print ("graph ", n, " nodes ", nedges, " non-zeroes")
  nedges2 = (nedges - n)
  print ("nedges = ", nedges2)

  # Add all nodes
  for i in range(n):
    G.add_node(i)

  # Add all edges
  for i in range(nedges):
    line = gfile.readline()
    x = line.split()
    n0 = int(x[0])
    n1 = int(x[1])
    eweight = abs(float(x[2]))
    if n0 != n1:
      if abs(eweight) > threshold:
        G.add_edge(n0,n1,weight=eweight)

  gfile.close()

  print ("graph size =", G.size())

  return G


def read_graph_file_noweights(G, prot_file):

  gfile = open(prot_file, "r")
  line = gfile.readline()
  line = gfile.readline()
  x = line.split()
  n = int(x[0])
  nedges = int(x[2])
  print("graph ", n, " nodes ", nedges, " elements")

  for i in range(n):
    G.add_node(i)

  for i in range(0, nedges):
    line = gfile.readline()
    x = line.split()
    n0 = int(x[0]) - 1
    n1 = int(x[1]) - 1
    if n0 != n1:
      G.add_edge(n0,n1)

  gfile.close()

  print("graph size =", G.size())

  return G


def read_graph_file_zerobased(G, prot_file):

  gfile = open(prot_file, "r")
  line = gfile.readline()
  line = gfile.readline()
  x = line.split()
  n = int(x[0])
  nedges = int(x[2])
  print("graph ", n, " nodes ", nedges, " elements")

  # Create nodes
  for i in range(n):
    G.add_node(i)

  # Add edges
  for i in range(nedges):
    line = gfile.readline()
    x = line.split()
    n0 = int(x[0])
    n1 = int(x[1])
    if n0 != n1:
      G.add_edge(n0,n1)

  gfile.close()

  print("graph size =", G.size())

  return G  


def createGraph(ftype, ifile, threshold):

  # Read in file as graph
  graph = nx.Graph()
  # Weighted mtx
  if ftype == 'mtx':
    graph = read_graph_file(graph, ifile, threshold)

  # Zero-based mtx
  elif ftype == '0mtx':
    graph = read_graph_file_zerobased(graph, ifile)

  # Unweighted mtx or no weights
  elif ftype =='umtx':
    graph = read_graph_file_unweighted(graph, ifile)

  # mutual information
  elif ftype == 'mi':
    graph = read_mi_file(graph, ifile, threshold)

  return graph


def showClusters(part_number, graph):

    #drawing
    color = {0:'red', 1:'blue', 2:'green', 3:'turquoise', 4:'yellow', 5:'orange', 6:'violet', 7:'pink', 8:'grey', 9:'black', 10:'teal', 11:'purple', 12:'cyan', 13:'magenta', 14:'brown'}
    partition = part_number
    size = float(len(set(partition.values())))
    pos = nx.spring_layout(graph)
    count = 0.
    for com in set(partition.values()) :
      count = count + 1.
      print (com)
      list_nodes = [nodes for nodes in partition.keys()
                                  if partition[nodes] == com]
      nx.draw_networkx_nodes(graph, pos, list_nodes, node_size = 80,
                                  node_color = color[com] ) #str(count / size))

    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    plt.show()


def calc_density(graph, num_nodes, result):

    # Get adjacency matrix
    A = nx.adjacency_matrix(graph)
    A.todense()

    nnz = A[A != 0].size
    #nnz = np.count_nonzero(A)
    print('\nNNZ = ', nnz)
    result['NNZ'] = nnz
    density = float(nnz/(num_nodes*num_nodes))
    sparsity = float(1.0 - density)
    print('\ndensity = ', density)
    print('\nsparsity= ', sparsity)
    result['density'] = density
    result['sparsity'] = sparsity


def write_partFile(part_num, num_nodes, nparts):

    pname = "comm" + str(nparts) + ".txt"
    PartFile = open(pname, "w")
    string=str(num_nodes)+"       "+'\n'
    PartFile.write(string)
    for i in range(num_nodes):
      string = str(i)+"  "+str(part_num[i])+"\n"
      PartFile.write(string)

    PartFile.close()


def write_resultFile(result):

    print('\nResult info:\n', result)

    resultFile = open("result.txt", "w")

    result_string=str(result)+'\n'
    resultFile.write(result_string)

    resultFile.close()


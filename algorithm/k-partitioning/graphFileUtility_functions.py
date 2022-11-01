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

#
# The Quantum Graph Partitioning Algorithm has been described
# in the following publications. Please cite in your publication.
#
# H. Ushijima-Mwesigwa, C. F. A. Negre, S. M. Mniszewski,
# 2017, Graph Partitioning using Quantum Annealing on the
# D-Wave System, Proceedings of the 2nd International
# Workshop on Post Moore’s Era Supercomputing (PMES), 22-29.
#

def write_graph_file(graph):

  gfile = open("graphx.txt", "w")
  n = str(nx.nx.number_of_nodes(graph))
  m = str(nx.nx.number_of_edges(graph))
  out = " ".join([n, m, "\n"])
  gfile.write(out)
  for edge in graph.edges():
      u,v = edge
      out = " ".join([str(u), str(v), "\n"])
      gfile.write(out)

  gfile.close()


def write_partfile(graph, part_number, num_nodes, num_parts):

  fname = "part" + str(num_parts) + ".txt"
  pfile = open(fname, "w")
  out = " ".join([str(num_nodes), "\n"])
  pfile.write(out)
  for node in graph.nodes():
    out = " ".join([str(node), str(part_number[node]), "\n"])
    pfile.write(out)

  pfile.close()


def write_mtx(graph):

  gfile = open("graphx.mtx", "w")
  header = "%%MatrixMarket matrix coordinate real general\n"
  gfile.write(header)

  n = nx.nx.number_of_nodes(graph)
  e = nx.nx.number_of_edges(graph)
  nonz = n+e+e
  out = " ".join([str(n), str(n), str(nonz), "\n"])
  gfile.write(out)

  for node in range(1,n+1):
    out = " ".join([str(node), str(node), "1.0\n"])
    gfile.write(out)
 
  for edge in graph.edges():
      u, v = edge
      u = u + 1
      v = v + 1
      out = " ".join([str(u), str(v), "1.0\n"])
      gfile.write(out)
      out = " ".join([str(v), str(u), "1.0\n"])
      gfile.write(out)

  gfile.close()
    

def qubo2mtx(qubofile):
 
  mfile = open("qubo.mtx", "w")
  header = "%%MatrixMarket matrix coordinate real general\n"
  mfile.write(header)

  qfile = open(qubofile, "r")
  line = qfile.readline()
  x = line.split()
  nnodes = int(x[4])
  nedges = int(x[5])
  nonz = nnodes + nedges + nedges
  out = " ".join([str(nnodes), str(nnodes), str(nonz), "\n"])
  mfile.write(out)
  
  line = qfile.readline()
  line = qfile.readline()
  for i in range(nnodes):
    line = qfile.readline()
    x = line.split()
    n0 = int(x[0]) + 1
    n1 = int(x[1]) + 1
    val = float(x[2])
    out = " ".join([str(n0), str(n1), str(val), "\n"])
    mfile.write(out)

  line = qfile.readline()
  line = qfile.readline()
  for i in range(nedges):
    line = qfile.readline()
    x = line.split()
    n0 = int(x[0]) + 1
    n1 = int(x[1]) + 1
    val = float(x[2])
    out = " ".join([str(n0), str(n1), str(val), "\n"])
    mfile.write(out)
    out = " ".join([str(n1), str(n0), str(val), "\n"])
    mfile.write(out)

  mfile.close()

def read_graph_file(G, prot_file):

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
  for i in range(0, nedges):
    line = gfile.readline()
    x = line.split()
    n0 = int(x[0]) - 1
    n1 = int(x[1]) - 1
    eweight = float(x[2])
    if n0 != n1:
      G.add_edge(n0, n1, weight=eweight)

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
    eweight = float(x[2])
    if n0 != n1:
      G.add_edge(n0, n1, weight=eweight)

  gfile.close()

  print("graph size =", G.size())

  return G


def createGraph(ftype, ifile, threshold):

  # Read in file as graph
  graph = nx.Graph()
  # Weighted mtx
  if ftype == 'mtx':
    graph = read_graph_file(graph, ifile)

  # Unweighted mtx
  if ftype == 'umtx':
    graph = read_graph_file_unweighted(graph, ifile)

  # Zero-based mtx
  elif ftype =='0mtx':
    graph = read_graph_file_zerobased(graph, ifile)

  return graph


def read_graph_file_unweighted(G, prot_file):

  threshold =  0.0
  gfile = open(prot_file, "r")
  line = gfile.readline()
  line = gfile.readline()
  x = line.split()
  n = int(x[0])
  nedges = int(x[2])
  print("graph ", n, " nodes ", nedges, " edges")

  # Create nodes
  for i in range(n):
    G.add_node(i)

  # Add edges
  for i in range(nedges):
    line = gfile.readline()
    x = line.split()
    n0 = int(x[0]) - 1
    n1 = int(x[1]) - 1

    if n0 != n1:
      G.add_edge(n0,n1)

  gfile.close()

  print("graph size =", G.size())

  return G


def write_metis_graph_file(graph):
    
    gfile = open("graph.txt", "w")
    n = str(nx.nx.number_of_nodes(graph))
    m = str(nx.nx.number_of_edges(graph))
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


def write_resultFile(result):

    print('\nResult info:\n', result)

    resultFile = open("result.txt", "w")

    result_string=str(result)+'\n'
    resultFile.write(result_string)

    resultFile.close()


def show_partitions(graph, part_number):

  color = {0:'red', 1:'blue', 2:'green', 3:'black', 4:'yellow', 5:'orange', 6:'violet', 7:'pink', 8:'grey', 9:'turquoise', 10:'teal', 11:'purple', 12:'cyan', 13:'magenta', 14:'brown'}

  partition = part_number
  size = float(len(set(partition.values())))
  #pos = nx.spring_layout(graph, 1.0)
  pos = nx.spring_layout(graph)
  count = 0.
  for com in set(partition.values()) :
    count = count + 1.
    print (com)
    list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
    nx.draw_networkx_nodes(graph, pos, list_nodes, node_size = 30,
                                node_color = color[com] ) #str(count / size))

  nx.draw_networkx_edges(graph, pos, alpha=0.5)
  plt.show()

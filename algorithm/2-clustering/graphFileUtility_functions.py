import numpy as np
import numpy.linalg as la
import networkx as nx

from dwave_qbsolv import QBSolv
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
    #else:
    #  G.add_node(n0)

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
  threshold = 0.0
  for i in range(nedges):
    line = gfile.readline()
    x = line.split()
    n0 = int(x[0]) - 1
    n1 = int(x[1]) - 1
    eweight = float(x[2])
    if n0 != n1:
      if abs(eweight) > threshold:
        #G.add_edge(n0,n1,weight=float(1.0))
        G.add_edge(n0,n1)
    #else:
    #  G.add_node(n0)

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
    #else:
    #  G.add_node(n0)

  gfile.close()

  print ("graph size =", G.size())

  return G


def read_graph_file_noweights(G, prot_file):

  gfile = open(prot_file, "r")
  line = gfile.readline()
  line = gfile.readline()
  x = line.split()
  n = x[0]
  nedges = int(x[2])
  print("graph ", n, " nodes ", nedges, " elements")

  for i in range(0, nedges):
    line = gfile.readline()
    x = line.split()
    n0 = int(x[0]) - 1
    n1 = int(x[1]) - 1
    if n0 != n1:
      G.add_edge(n0,n1)
    else:
      G.add_node(n0)

  gfile.close()

  print("graph size =", G.size())

  return G


def createGraph(ftype, ifile, threshold):

  # Read in file as graph
  graph = nx.Graph()
  # Weighted mtx
  if ftype == 'mtx':
    graph = read_graph_file(graph, ifile, threshold)

  # Unweighted mtx
  elif ftype =='umtx':
    graph = read_graph_file_unweighted(graph, ifile)

  # No weights
  elif ftype == 'nmtx':
     graph = read_graph_file_noweights(graph, ifile)

  # mutual information
  elif ftype == 'mi':
    graph = read_mi_file(graph, ifile, threshold)

  return graph

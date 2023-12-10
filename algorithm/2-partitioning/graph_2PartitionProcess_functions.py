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
import datetime as dt

#
# The Quantum Graph Partitioning Algorithm has been described
# in the following publications. Please cite in your publication.
#
# H. Ushijima-Mwesigwa, C. F. A. Negre, S. M. Mniszewski,
# 2017, Graph Partitioning using Quantum Annealing on the
# D-Wave System, Proceedings of the 2nd International
# Workshop on Post Moore’s Era Supercomputing (PMES), 22-29.
#


def makeQubo(laplacian, alpha, beta, gamma, graph, num_nodes, num_parts):

  # Create QUBO matrix
  n = num_nodes
  Q = np.zeros((n,n))
  
  # Create Adjacency matrix A
  A = nx.adjacency_matrix(graph)

  # Calculate degree for each node
  g = np.zeros([n])
  g = A.sum(1)

  for i in range(n):
    for j in range(i, n):
      if i == j:
        entry = beta * g[i] - alpha * (n - 1)
        Q[i,i] = entry
      else:
        if A[i,j] > 0:
          entry = alpha - beta
        else:
          entry = alpha
        Q[i,j] = entry
        Q[j,i] = entry

  return Q


def compute_cut(graph, part_number, num_nodes, num_parts):

  cut = 0
  for edge in graph.edges():
    u, v = edge
    if part_number[u] != part_number[v]:
      cut += 1

  return cut


def write_metis_graph_file(graph):
    
  gfile = open("graph.txt", "w")
  n = str(nx.number_of_nodes(graph))
  m = str(nx.number_of_edges(graph))
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


def get_metis_edgecut():

  myFile = open("metis_output.out", 'r')
  
  metis_cost = 0
  for lines in myFile:
    lines_split = lines.split()
    if(len(lines_split) > 1):
      if lines_split[1] == "Edgecut:":
        metis_cost = lines_split[2].replace(',', '')
        metis_cost.strip()
        break
    
  return metis_cost


def get_kaffpa_edgecut():

  myFile = open("kaffpa_output.out", 'r')
  
  kaffpa_cut = 0
  for lines in myFile:
    lines_split = lines.split()
    if(len(lines_split) > 1):
      if lines_split[0] == "cut":
        kaffpa_cost = lines_split[1]
        kaffpa_cost.strip()
        break
        
  return kaffpa_cost


def compare_with_metis_and_kahip(graph, part_number, num_nodes, num_parts, result):

  qubo_cut = compute_cut(graph, part_number, num_nodes, num_parts)

  write_metis_graph_file(graph)
  #os.system("./gpmetis -ufactor=1 graph.txt "+str(num_parts)+"  > metis_output.out") 
  metis_cut = 0
  #metis_cut = get_metis_edgecut()
  #os.system("./kaffpa graph.txt " + "--k=" +str(num_parts)+"  > kaffpa_output.out")

  kaffpa_cut = 0
  #kaffpa_cut = get_kaffpa_edgecut()

  fedges = float(nx.number_of_edges(graph))
  print('\nEdge cut results:')
  print("dwave cut: ", qubo_cut, " fraction: ", float(qubo_cut)/fedges)
  if metis_cut > 0:
    print("metis cut: ",metis_cut, " fraction: ", float(metis_cut)/fedges)
  if kaffpa_cut > 0:
    print("kahip:", kaffpa_cut, " fraction: ", float(kaffpa_cut)/fedges)
  if metis_cut > 0 and qubo_cut > 0:
    print("difference: ", abs(int(float(metis_cut) - float(qubo_cut))), " fraction: ", abs(float(metis_cut)/fedges - float(qubo_cut)/fedges))

  return qubo_cut


def process_solution(ss, graph, num_nodes, num_parts, result):

  # Determine sizes of 2 parts
  cc = np.zeros([num_nodes])
  c0 = 0
  c1 = 0
  for i in range(num_nodes):
    if ss[0,i] == 0:
      c0 = c0 + 1
    else:
      c1 = c1 + 1
      cc[i] = 1

  print('\n 2 parts of size:', c0, c1)
  if c0 < c1:
    balance = float(c0/c1)
  else:
    balance = float(c1/c0)
  print('\nbalance = ', balance)
  result['balance_metric'] = balance

  cldet = [[] for i in range(2)]
  cdet = np.zeros([num_nodes])
  for i in range(num_nodes):
    cdet[i] = i

  cldet[0] = cdet[cc == 0]
  cldet[1] = cdet[cc == 1]

  for i in range(2):
    print('cldet ', i, cldet[i])
  print(flush=True)

  return cc, cldet


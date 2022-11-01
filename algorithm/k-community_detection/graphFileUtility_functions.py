#!/usr/bin/env python

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


def read_mtx(fname, tolerance):

  eq_tolerance = tolerance
  gfile = open(fname, "r")

  # Read header and size
  line = gfile.readline()
  line = gfile.readline()
  x =  line.split()
  nsize = int(x[0])
  esize = int(x[2])
  print('\nDimension = ', nsize, ' Number elements = ', esize)

  garray = np.zeros([nsize,nsize])
	 
  for i in range(esize):
    line = gfile.readline()
    x = line.split()
    ii = int(x[0]) - 1
    jj = int(x[1]) - 1

    if ii < 0 or ii > (nsize-1):
      print('\nIllegal ii = ', ii)
      exit(0)
    if jj < 0 or jj > (nsize-1):
      print('\nIllegal jj = ', jj)
      exit(0)

    evalue = float(x[2])
    if abs(evalue) > eq_tolerance:
      garray[ii,jj] = evalue

  gfile.close()

  return garray


def write_mtx(garray, fname, isize, tolerance):

  eq_tolerance = tolerance
  gfile = open(fname, "w")
  header = "%%MatrixMarket matrix coordinate real general\n"
  gfile.write(header)

  vcount = 0
  for i in range(isize):
    for j in range(isize):
      if abs(garray[i,j]) > eq_tolerance:
        vcount = vcount + 1

  out = ' '.join([str(isize), str(isize), str(vcount), "\n"])
  gfile.write(out)

  for i in range(isize):
    for j in range(isize):
      if abs(garray[i,j]) > eq_tolerance:
        out = ' '.join([str(i+1), str(j+1), str(garray[i,j]), "\n"])
        gfile.write(out)

  gfile.close()


def gershgorin(hmatrix, hsize):

  emin = 99999 
  emax = -99999

  for i in range(hsize):
    radius = 0.0
    for j in range(hsize):
      absham = abs(hmatrix[i,j])
      radius = radius + absham

    dvalue = hmatrix[i,i]
    radius = radius - abs(dvalue)

    if (dvalue + radius) > emax:
      emax = dvalue + radius
    if (dvalue - radius) < emin:
      emin = dvalue - radius 

  return emin, emax


def normalize(hmatrix, hsize):

  gmatrix = np.zeros([hsize,hsize])

  emin, emax = gershgorin(hmatrix, hsize)
  print('\nemin = ', emin, 'emax = ', emax)

  maxminusmin = emax - emin
  hbeta = emax / maxminusmin
  halpha = -1.0 / maxminusmin
  print('\nmaxminusmin = ', maxminusmin, 'beta = ', hbeta, 'alpha = ', halpha)

  gmatrix = halpha * hmatrix

  gmatrix = gmatrix + hbeta * np.identity(hsize)

  return gmatrix


def read_graph_file_posweighted(G, prot_file, threshold):

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
    eweight = float(x[2])
    if n0 != n1:
      if eweight > threshold:
        G.add_edge(n0,n1,weight=eweight)
    #else:
    #  G.add_node(n0)

  gfile.close()

  print ("graph size =", G.size())

  return G


def import_konect(fpath):
    G = nx.convert_node_labels_to_integers(nx.read_edgelist(fpath, comments='%', data=False))
    logging.info("Imported graph: {}".format(nx.info(G)))
    return G


def write_partFile(part_num,Dim,nparts):

  pname = "comm" + str(nparts) + ".txt"
  PartFile = open(pname, "w")
  string=str(Dim)+"       "+'\n'
  PartFile.write(string)
  for i in range(Dim):
    string = str(i)+"  "+str(part_num[i])+"\n"
    PartFile.write(string)

  PartFile.close()


def write_modFile(Mod,Dim):

  WriteFile = open("mod.txt", "w")
  string=str(Dim)+"       "+'\n'
  WriteFile.write(string)

  for ii in range(0,Dim):
    string=str(ii)+"       "+str(ii)+"       "+str(Mod[ii,ii])+"       "+'\n'
    WriteFile.write(string)

  for ii in range(0,Dim):
    for jj in range(ii+1,Dim):
      if ii != jj:
        string=str(ii)+"       "+str(jj)+"       "+str(2*Mod[ii,jj])+"       "+'\n'
        WriteFile.write(string)

  WriteFile.close()


def read_modFile(mfile):

  gfile = open(mfile, "r")
  line = gfile.readline()
  x = line.split()
  n = int(x[0])
  nedges = n*n
  print ("graph ", n, " nodes ", nedges, " non-zeroes")

  Mod = np.zeros((n, n))

  for i in range(0, n):
    for j in range(ii+1,n):
      line = gfile.readline()
      x = line.split()
      n0 = int(x[0])
      n1 = int(x[1])
      eweight = float(x[2])
      Mod[n0,n1] = eweight

  gfile.close()

  return Mod


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

def read_graph_file_zerobased(G, prot_file, threshold):

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
    #else:
    #  G.add_node(n0)

  gfile.close()

  print("graph size =", G.size())

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


def read_net_file(data_file):

  G = nx.Graph() 
  gfile = open(data_file, "r")

  # Read Vertices line
  line = gfile.readline()
  x = line.split()
  field0 = x[0]
  n = int(x[1])

  # Read Arcs and Edges lines
  while field0 != '*Edges':
    line = gfile.readline()
    x = line.split()
    field0 = x[0]

  # Read in all edges with weights
  nedges = 0
  line = gfile.readline()
  while line != '':
    x = line.split()
    n0 = int(x[0]) - 1
    n1 = int(x[1]) - 1
    eweight = float(x[2])
    G.add_edge(n0,n1,weight=eweight)
    nedges += 1
    line = gfile.readline()

  print ("nedges = ", nedges)

  gfile.close()

  print ("graph size =", G.size())

  return G


def generateGraph(ftype, ifilename, threshold):

  # Weighted mtx
  if ftype == 'mtx':
    graph = nx.Graph()
    graph = read_graph_file(graph, ifilename, threshold)

  # Unweighted or no weights mtx
  elif ftype =='umtx':
    graph = nx.Graph()
    graph = read_graph_file_unweighted(graph, ifilename)

  # Zero-based
  elif ftype == '0mtx':
    graph = nx.Graph()
    graph = read_graph_file_zerobased(graph, ifilename, threshold)

  # mutual information
  elif ftype == 'mi':
    graph = nx.Graph()
    graph = read_mi_file(graph, ifilename, threshold)

  # gml
  elif ftype == 'gml':
    graph = nx.read_gml(ifilename, 'id', None)

  # konect
  elif ftype == 'konect':
    graph = import_konect(ifilename)

  # net
  elif ftype == 'net':
    graph = read_net_file(ifilename)

  return graph


def write_resultFile(result):

    print('\nResult info:\n', result)

    resultFile = open("result.txt", "w")

    result_string=str(result)+'\n'
    resultFile.write(result_string)

    resultFile.close()


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


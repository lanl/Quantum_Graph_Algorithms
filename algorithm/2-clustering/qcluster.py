# useful additional packages
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigs, eigsh
from numpy.linalg import eig

from qiskit import IBMQ
from qiskit.quantum_info import Pauli, SparsePauliOp

from qiskit import Aer
from qiskit.tools.visualization import plot_histogram
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram
from qiskit.opflow.primitive_ops import PauliSumOp

import argparse
import logging

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

  print("graph size =", G.size())

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


def createOrReadGraph(ifile, ftype, threshold):

  # Weighted mtx
  if ftype == 'mtx':
    graph = nx.Graph()
    graph = read_graph_file(graph, ifile, threshold)

  # Unweighted mtx
  else:
    if ftype =='umtx':
      graph = nx.Graph()
      graph = read_graph_file_unweighted(graph, ifile)

  # mutual information
    else:
      if ftype == 'mi':
        graph = nx.Graph()
        graph = read_mi_file(graph, ifile, threshold)

  # test graph
      else:
        if ftype == 'test':
          graph = nx.connected_caveman_graph(2,4)
          #graph = generateGraph()

  print("graph = ", graph.size(), graph.nodes(), graph.edges())

  return graph

# Generating a graph of 4 nodes
def generateGraph():
  n=4 # Number of nodes in graph
  G=nx.Graph()
  G.add_nodes_from(np.arange(0,n,1))
  elist=[(0,1,1.0),(0,2,1.0),(0,3,1.0),(1,2,1.0),(2,3,1.0)]
  # tuple is (i,j,weight) where (i,j) is the edge
  G.add_weighted_edges_from(elist)

  return G

def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    plt.show()

def calcBruteForce(w, nnodes):
  best_cost_brute = 0
  print('\nBrute force results:')
  for b in range(2**nnodes):
    x = [int(t) for t in reversed(list(bin(b)[2:].zfill(nnodes)))]
    cost = 0
    for i in range(nnodes):
      for j in range(nnodes):
        cost = cost + w[i,j]*x[i]*x[j]
        #cost = cost + w[i,j]*x[i]*(1-x[j])
    if best_cost_brute < cost:
      best_cost_brute = cost
      xbest_brute = x

    print('case = ' + str(x)+ ' cost = ' + str(cost))

  print('\nBest solution = ' + str(xbest_brute) + ' cost = ' + str(best_cost_brute))
  return xbest_brute, best_cost_brute

def build_mod(Adj, thresh, num_edges):
        """Builds the modularity matrix from the Adjacency matrix.

        Given an adj matrix, it constructs the modularity matrix and its graph.

        """

        Dim = Adj.shape[1]
        #print ("\n Dim = ", Dim)

        #print ("\n Computing modularity matrix ...")

        Deg = np.zeros([Dim])

        M = 0.0

        #print ("\n Adj degrees: ")
        for ii in range(0,Dim):
          for jj in range(0,Dim):
            if(ii != jj):
              Deg[ii] = Deg[ii] + Adj[ii, jj]
          M = M + Deg[ii]
              #M = M + 1
          #print ii,Deg[ii]
          mtotal = M/2.0

        #print ("\n Number of edges:", M, num_edges)

        Mod = np.zeros([Dim,Dim])

        # Calc modularity matrix
        for ii in range(0,Dim):
          for jj in range(0,Dim):
            Mod[ii,jj] = Adj[ii,jj] - (Deg[ii]*Deg[jj])/M
            if(ii == jj):
              Mod[ii,jj] = - (Deg[ii]*Deg[jj])/M

        # Calc modularity degrees
        DegM = np.zeros([Dim])
        #print ("\n Mod degrees: ")
        for ii in range(0,Dim):
          for jj in range(0,Dim):
           if(ii != jj):
              DegM[ii] = DegM[ii] + Mod[ii,jj]
              M = M + DegM[ii]
              #print ii,DegM[ii]

        # Threshold mod
        nonzeros = 0
        no_edges = 0
        for ii in range(0,Dim):
          for jj in range(0,Dim):
            if ii != jj:
              if(abs(Mod[ii,jj]) > thresh):
                nonzeros =  nonzeros + 1
              else:
                Mod[ii,jj] = 0.0

        #print ("\n Mod matrix:")
        #print(Mod)

        return mtotal, Mod


# Computing the weight matrix from the random graph
def computeWeightMatrix(G,n):
  w = np.zeros([n,n])
  for i in range(n):
    for j in range(n):
      temp = G.get_edge_data(i,j,default=0)
      if temp != 0:
          w[i,j] = temp['weight']

  return w

def calcBruteForce(w,n):
  best_cost_brute = 0
  print('\nBrute force results:')
  for b in range(2**n):
    x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]
    cost = 0
    for i in range(n):
      for j in range(n):
        #cost = cost + w[i,j]*x[i]*(1-x[j])
        cost = cost + w[i,j]*x[i]*x[j]
    if best_cost_brute < cost:
      best_cost_brute = cost
      xbest_brute = x
    #print('case = ' + str(x)+ ' cost = ' + str(cost))

  print('\nBest solution = ' + str(xbest_brute) + ' cost = ' + str(best_cost_brute))

def calcPaulisShift(w,nnodes):
  pauli_list = []
  coef_list = []
  shift = 0
  for i in range(nnodes):
    for j in range(i):
      if w[i, j] != 0:
        x_p = np.zeros(nnodes, dtype=bool)
        z_p = np.zeros(nnodes, dtype=bool)
        z_p[i] = True
        z_p[j] = True
        pauli_list.append((-0.5 * w[i, j], Pauli((z_p, x_p))))
        coef_list.append(-0.5 * w[i, j])
        shift -= w[i, j]
  
  print('\npauli_list = ', pauli_list)
  
  #return SparsePauliOp(pauli_list), shift
  return PauliSumOp(SparsePauliOp(pauli_list, coef_list)), shift

def calc_paulis_shift(w, nnodes):
  pauli_list = []
  cost_shift = 0

  for i in range(nnodes):
    for j in range(nnodes):
      if w[i,j] != 0:
        cost_shift = cost_shift - w[i,j]
        wp = np.zeros(nnodes)
        vp = np.zeros(nnodes)
        vp[i] = 1
        vp[j] = 1
        pauli_list.append((-w[i,j],Pauli(vp,wp)))

  print("cost_shift =", cost_shift)
  #print('pauli_list = ', len(pauli_list.paulis), '\n', pauli_list.print_details())

  return WeightedPauliOperator(paulis=pauli_list)




if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Test clustering')
  parser.add_argument('-nnodes', type=int, default=10, help='number of nodes')
  parser.add_argument('-nparts', type=int, default=2, help='number of parts')
  parser.add_argument('-pflag', type=int, default=0, help='0-no plot, 1-plot')
  parser.add_argument('-partfile', default='comm2.txt', help='comm file name')
  parser.add_argument('-ifile', help='input filename')
  parser.add_argument('-ftype', default='test', help='input file type (test, mtx, umtx, mi')
  parser.add_argument('-threshold', type=float, default=0.00, help='threshold value')

  args = parser.parse_args()

  print('mumber nodes = ', args.nnodes)
  print('number parts = ', args.nparts)
  print('plot flag = ', args.pflag)
  print('partfile = ', args.partfile)
  print('input file = ', args.ifile)
  print('file  type= ', args.ftype)
  print('threshold = ', args.threshold)
  print('\n')

  nnodes = args.nnodes
  nparts = args.nparts
  pflag = args.pflag
  partfile = args.partfile
  ifile = args.ifile
  ftype = args.ftype
  threshold = args.threshold

  MY_TOKEN = 'd14c0ed83858ee6971324bdc5462b02f629e4cc00f1e722dc0632ca2e701191a367992e0c1a195a2dc7588c6a249ee9990f2997908df27f9c67ac35179c5c00c'
  provider1 = IBMQ.enable_account(MY_TOKEN)

  provider = IBMQ.get_provider(hub='ibm-q-lanl', group='lanl', project='quantum-optimiza')

  # Make graph
  G = createOrReadGraph(ifile, ftype, threshold)
  nnodes = nx.number_of_nodes(G)
  num_edges = nx.number_of_edges(G)
  print('\nnnodes = ', nnodes, ' num_edges = ', num_edges)
  
  #if pflag == 1:
  #  colors = ['r' for node in G.nodes()]
  #  pos = nx.spring_layout(G)
  #  draw_graph(G, colors, pos)

  # Get Adjacency matrix
  A = nx.adjacency_matrix(G)
  for i in range(nnodes):
    for j in range(nnodes):
      if abs(A[i,j]) > 0:
        A[i,j] = abs(A[i,j])
  print ("Adjacency matrix:")
  print (A.todense())

  # Calculate modularity matrix
  mtotal,w = build_mod(A, threshold, num_edges)
  print('\nModularity Matrix:\n', w)

  # Calculate brute force approach
  calcBruteForce(w, nnodes)

  # Create cluster as quadratic program
  #qcluster,shift = calcPaulisShift(w,nnodes)
  #print('\nqcluster = ', qcluster)
  #print('shift = ', shift)
  #print('qubits = ', qcluster.num_qubits)
  #print('coefs = ', qcluster.coeffs)
  #qspmat = qcluster.to_spmatrix()
  #qmat = qcluster.to_matrix()
  #print('\nqmat = ', qmat)
  #w = eigsh(qmat, k=1),
  #ww = eig(qmat),
  #leig = ww[0][0][0]
  #print('energy = ', leig)

  # Make quadratic program
  qp = QuadraticProgram('2-cluster')
  for i in range(nnodes):
    qp.binary_var()
  #print(qp.export_as_lp_string())

  lin = np.zeros(nnodes, dtype=float)
  for i in range(nnodes):
    lin[i] = w[i,i]
  #print('\nlin = ', lin)

  qmat = np.zeros([nnodes, nnodes])
  for i in range(nnodes):
    for j in range(nnodes):
      if i != j:
        qmat[i,j] = w[i,j]
  #print('\nqmat = ', qmat)

  qp.maximize(constant = 0, linear=lin, quadratic=qmat)
  print('\n',qp.export_as_lp_string())
  #print('\nqp as cplex:')
  #qp.to_docplex().prettyprint()

  # Convert to qubit operator
  qubitOp, offset = qp.to_ising()
  print('\nqubitOp = ')
  print('num_qubits = ', qubitOp.num_qubits)
  print('Offset:', offset)
  print('Ising Hamiltonian:')
  print(str(qubitOp))

  # solving Quadratic Program using exact classical eigensolver
  exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
  result = exact.solve(qp)
  print('\nExact qp result = ', result)


  # setup to run on simulator
  algorithm_globals.random_seed = 123
  seed = 10598
  backend = Aer.get_backend('statevector_simulator')
  quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)

  # construct VQE
  spsa = SPSA(maxiter=300)
  #ry = TwoLocal(qubitOp.num_qubits, 'ry', 'cz', reps=2, entanglement='linear')
  ry = TwoLocal(qubitOp.num_qubits, 'ry', 'cz', reps=5, entanglement='full')
  vqe = VQE(ry, optimizer=spsa, quantum_instance=quantum_instance)
  result = vqe.compute_minimum_eigenvalue(qubitOp)
  print('\nVQE sim qubitOp result = ')
 
  # print results
  print('\nSimulator results:')
  print('energy:', result.eigenvalue.real)
  print('time:', result.optimizer_time)
  print('objective:', result.eigenvalue.real + offset)
  #print('solution:', result.x)
  #print('solution objective:', qp.objective.evaluate(x))

  # create minimum eigen optimizer based on VQE
  vqe_optimizer = MinimumEigenOptimizer(vqe)

  # solve quadratic program
  result = vqe_optimizer.solve(qp)
  print('\nVQE sim qp results:')
  print('result =',result)
  print('optimal results = ', result.x)

  if pflag == 1:
    colors = ['r' if result.x[i] == 0 else 'c' for i in range(nnodes)]
    pos = nx.spring_layout(G)
    draw_graph(G, colors, pos)

  exit(0)

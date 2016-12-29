#!/usr/bin/env python
import networkx as nx
import os.path
import numpy
import sys
import math
import collections
from collections import Counter

from random import choice, sample

import time
from subprocess import call

import numpy.linalg as LA
from scipy.optimize import minimize


global algorithm # the algorithm created as global variable

DIMENSION = 7 # the number of features used  
SAMPLE_SIZE = 100 # the number of trials to construct a pair of ground truth communities randomly 
SHARP = 50.0 # the parameter to increase robustness against outliers of the sampled ground truth pairs  
LAMBDA1 = 0.1  # the regularization parameter for weight variance, see paper lambda_1
               # NOTE: high LAMBDA1 results in a small variance of edge weights 
LAMBDA2 = 100.0 / SAMPLE_SIZE # the co-efficient for the penality terms, see lambda_2 in the paper
AVE = 1 # the suggested average edge weight, a parameter in the regularization term   

# wrapper function for the objective function 
def f_wrapper(p):
  return algorithm.f(p)

# wrapper function for the gradient of the objective function 
def f_der_wrapper(p):
  return algorithm.f_der(p)

# main optimization process based on scipy 
def solve():
  starttime = time.time()
  res = minimize(f_wrapper, algorithm.p, method="BFGS", jac=f_der_wrapper, \
                options={'maxiter': 8, 'gtol': 1e-4, 'disp': True})
  endtime = time.time()
  print "time of convergence:",
  print(endtime - starttime)
  print "final result: p=", (res.x)
  return res

# sigmoid function 
def sigmoid(x):
    x = SHARP * x
    if x > 10: x = 10.0  # exp(10) is too large, impossible 
    if x < -10: x = -10.0
    return 1.0 / ( 1.0 + math.exp(- x) ) 

# the f-measure
def f1score(nodes, heatnodes):
    common = len(nodes.intersection(heatnodes))
    precision = 1.0 * common / len(nodes)
    recall = 1.0 * common / len(heatnodes)
    if precision + recall == 0:
        return 0
    else:
        return ( 2.0 * precision * recall / (precision + recall) )

# the first order derivative of the sigmoid function
def sigmoid_der(x):
    return SHARP * sigmoid(x) * ( 1.0 - sigmoid(x) ) 


class AdaptiveModularity:
    def __init__(self, target):
        self.target = target
        self.loadTargetGraph()

        # TRAINING DATA CHOSEN HERE

        # Sample parameters of the target graph,
        #       i.e degrees, clustering coefficient
        degs, ccs = self.sample_graph_parameter()
        self.loadArtifialGraph(degs, ccs) 

        # Alternatively, use football network as the training dataset
        #self.loadFootball()

    def loadTargetGraph(self):
        if self.target == "football":
            self.loadFootball()
        elif self.target == "LFR":
            self.loadLFR("mu4.5")
        elif self.target == "amazon":
            self.loadAmazon()
        elif self.target == "dblp":
            self.loadDBLP()

    def sample_graph_parameter(self):
        bunch = sample(self.G.nodes(), max(100, int(numpy.sqrt(len(self.G)))) )
        degs = self.G.degree(bunch).values()
        ccs = nx.clustering(self.G, nodes = bunch).values()
        return degs, ccs

    def loadArtifialGraph(self, degs, ccs):
        N = 500
        self.G = nx.Graph(gnc = {}, membership = {})
        for index, mu in enumerate([0.3,0.32,0.35,0.37,0.4]):
            #params = ["./LFR_benchmark/benchmark","-N",str(N),"-k", str(1.0 * sum(degs) / len(degs)),\
            #          "-maxk",str(max(degs)),"-mu","0.4","-minc","20","-maxc","50","-C", str(1.0 * sum(ccs) / len(ccs))]
            params = ["./LFR_benchmark/benchmark","-N",str(N),"-k", str( max(1.0 * sum(degs) / len(degs), 10) ),\
                      "-maxk",str(max(max(degs), 50)),"-mu",str(mu), "-minc","20","-maxc","100","-C", str(1.0 * sum(ccs) / len(ccs))]

            print "*" * 20
            print " ".join(params)
            print "*" * 20
            call(params)
            time.sleep(3)
            with open("./network.dat", "r") as f:
                for line in f:
                    seg = line.split()
                    l, r = index * N + int(seg[0]), index * N + int(seg[1])
                    self.G.add_edge(l, r)
            with open("./community.dat", "r") as fconf:
                for line in fconf:
                    seg = line.split()
                    l, comm = index * N + int(seg[0]), index * N + int(seg[1])
                    self.G.graph["membership"][l] = [comm] 
                    if comm in self.G.graph["gnc"]:
                        self.G.graph["gnc"][ comm ].append( l ) 
                    else:
                        self.G.graph["gnc"][ comm ] = [ l ]
        call(["rm", "community.dat", "network.dat", "time_seed.dat", "statistics.dat"])


    def loadFootball(self):
        self.G = nx.Graph(gnc = {}, membership = {})
        with open("./input/football_raw_data/footballTSEinputEL.dat", "r") as f:
            for line in f:
                seg = line.split()
                self.G.add_edge( int(seg[0]), int(seg[1]) )
        with open("./input/football_raw_data/footballTSEinputConference.clu", "r") as fconf:
            for i, line in enumerate(fconf):
                self.G.graph["membership"][i] = [ str(line.strip()) ]
                if str(line.strip()) in self.G.graph["gnc"]:
                    self.G.graph["gnc"][str(line.strip())].append(i) 
                else:
                    self.G.graph["gnc"][str(line.strip())] = [ i ]

    def loadLFR(self, mu): 
        self.G = nx.Graph(gnc = {}, membership = {})
        with open("./input/LFR_raw_data/%s/network.dat" % (mu), "r") as f:
            for line in f:
                seg = line.split()
                self.G.add_edge( int(seg[0]), int(seg[1]) )
        with open("./input/LFR_raw_data/%s/community.dat" % (mu), "r") as fconf:
            for line in fconf:
                seg = line.split()
                self.G.graph["membership"][int(seg[0])] = [int(seg[1])] 
                if int(seg[1]) in self.G.graph["gnc"]:
                    self.G.graph["gnc"][ int(seg[1]) ].append( int(seg[0]) ) 
                else:
                    self.G.graph["gnc"][ int(seg[1]) ] = [ int(seg[0]) ]
        with open("./input/LFR_raw_data/%s/LFR.group" % (mu), "w+") as txt:
            for key in self.G.graph["gnc"].keys():
                txt.write(" ".join([str(_) for _ in self.G.graph["gnc"][key]]) + "\n")

    def loadAmazon(self):
        if os.path.isfile("./input/amazon/amazon.gpickle"): # fast reload graph 
            self.G = nx.read_gpickle("./input/amazon/amazon.gpickle")
        else:
            self.G = nx.Graph(gnc = {}, membership = {}, top5000 = {}, top5000_membership = {})
            with open("./input/amazon/com-amazon.ungraph.txt", "r") as txt:
                for line in txt:
                    if not line[0] == '#':
                        e = line.split()
                        self.G.add_edge(int(e[0]), int(e[1]))
            with open("./input/amazon/com-amazon.top5000.cmty.txt", "r") as txt:
                count = 0
                for line in txt:
                    if not line[0] == '#':
                        e = line.split()
                        self.G.graph["top5000"][count] = [int(_) for _ in e]
                        for n in self.G.graph["top5000"][count]:
                            if n in self.G.graph["top5000_membership"]:
                                self.G.graph["top5000_membership"][n].append( count )
                            else:
                                self.G.graph["top5000_membership"][n] = [ count ]
                        count += 1
            with open("./input/amazon/com-amazon.all.dedup.cmty.txt", "r") as txt:
                count = 0
                for line in txt:
                    if not line[0] == '#':
                        e = line.split()
                        self.G.graph["gnc"][count] = [int(_) for _ in e]
                        for n in self.G.graph["gnc"][count]:
                            if n in self.G.graph["membership"]:
                                self.G.graph["membership"][n].append( count )
                            else:
                                self.G.graph["membership"][n] = [ count ]
                        count += 1
            print "write gpickle file.."
            nx.write_gpickle(self.G, "./input/amazon/amazon.gpickle")

    def loadDBLP(self):
        if os.path.isfile("./input/dblp/dblp.gpickle"):
            self.G = nx.read_gpickle("./input/dblp/dblp.gpickle")
        else:
            self.G = nx.Graph(gnc = {}, membership = {}, top5000 = {})
            with open("./input/dblp/com-dblp.ungraph.txt", "r") as txt:
                for line in txt:
                    if not line[0] == '#':
                        e = line.split()
                        self.G.add_edge(int(e[0]), int(e[1]))
            with open("./input/dblp/com-dblp.top5000.cmty.txt", "r") as txt:
                count = 0
                for line in txt:
                    if not line[0] == '#':
                        e = line.split()
                        self.G.graph["top5000"][count] = [int(_) for _ in e]
                        count += 1
            with open("./input/dblp/com-dblp.all.cmty.txt", "r") as txt:
                count = 0
                for line in txt:
                    if not line[0] == '#':
                        e = line.split()
                        self.G.graph["gnc"][count] = [int(_) for _ in e]
                        for n in self.G.graph["gnc"][count]:
                            if n in self.G.graph["membership"]:
                                self.G.graph["membership"][n].append( count )
                            else:
                                self.G.graph["membership"][n] = [ count ]
                        count += 1
            print "write gpickle file.."
            nx.write_gpickle(self.G,"./input/dblp/dblp.gpickle")

    def edge_feature(self, e):
        return numpy.array([
                            math.sqrt(float(len(set(self.G[e[0]]).intersection(self.G[e[1]])))), \
                            float(abs(nx.clustering(self.G, e[0]) - nx.clustering(self.G, e[1]))), \
                            float(list(nx.jaccard_coefficient(self.G, [(e[0], e[1])]))[0][2]), \
                            float(list(nx.resource_allocation_index(self.G, [(e[0], e[1])]))[0][2]),\
                            float(min(len(self.G[e[0]]), len(self.G[e[1]]))) / float(max(len(self.G[e[0]]), len(self.G[e[1]]))), \
                            float(list(nx.adamic_adar_index(self.G, [(e[0], e[1])]))[0][2]), \
                            1.0
                           ])

    def preprocess(self):
        self.dimension = DIMENSION 
        self.numiter = 0 
        self.p = numpy.ones(self.dimension)
        self.edges_involved = set() 
        self.comm = {}
        self.pairs = set() 
        for t in range(SAMPLE_SIZE):
            i = choice(self.G.graph["gnc"].keys())
            nodes_i = self.G.graph["gnc"][i]
            if len(nodes_i) > 10 and len(nodes_i) < 100: 
                self.preprocess_helper(i, nodes_i)
                memberships = [self.G.graph["membership"][e[1]]
                                for e in self.comm[i]["eout"]
                                if e[1] in self.G.graph["membership"] 
                            ] 
                if not memberships: continue
                neighbor_comms = reduce(lambda x,y: list(x+y), memberships)
                for _ in range(min(len(neighbor_comms), 30)):
                    j = choice(neighbor_comms)
                    nodes_j = self.G.graph["gnc"][j]
                    if len(nodes_j) > 5 and len(nodes_j) < 100 and f1score(set(nodes_i), set(nodes_j)) < 0.4 :
                        print "comm",i,"(",len(nodes_i),"nodes)",": comm",j,"(",len(nodes_j),"nodes)"        
                        self.preprocess_helper(j, nodes_j)
                        self.preprocess_helper((i,j), list(set(nodes_i + nodes_j)))
                        self.pairs.add((i,j)) 
                        break

        self.feature_normalization()
        self.ein_eout_vectors()

        return

    def ein_eout_vectors(self):
        for i,j in self.pairs:
            self.preprocess_vector(i)
            self.preprocess_vector(j)
            self.preprocess_vector((i,j))

        self.ave_edge_feature = 1.0 * numpy.sum([ self.G.edge[e[0]][e[1]]["feature"] for e in self.edges_involved], axis=0) \
                                        / float(len(self.edges_involved))
        self.sum_edge_feature = self.ave_edge_feature * self.G.number_of_edges()
        self.cov_edge_feature = numpy.zeros((self.dimension, self.dimension)) 
        for e in self.edges_involved:
            self.cov_edge_feature = numpy.add(self.cov_edge_feature,\
                                            numpy.dot(\
                                                self.G.edge[e[0]][e[1]]["feature"][numpy.newaxis].T,\
                                                self.G.edge[e[0]][e[1]]["feature"][numpy.newaxis])\
                                        )
        self.cov_edge_feature /= float(len(self.edges_involved))
        print "covariance matrix", self.cov_edge_feature
        print "Average edge feature:", self.ave_edge_feature
 
    def feature_normalization(self):
        self.maxf = {di : -99 for di in range(self.dimension - 1)} 
        self.minf = {di : 999999 for di in range(self.dimension - 1)} 
        for e in self.edges_involved:
            for di in range(self.dimension - 1):
                self.maxf[di] = max(self.maxf[di], self.G.edge[e[0]][e[1]]["feature"][di])
                self.minf[di] = min(self.minf[di], self.G.edge[e[0]][e[1]]["feature"][di])
        print "Feature Min:", self.minf
        print "Feature Max:", self.maxf
        for e in self.edges_involved:
            for di in range(self.dimension - 1):
                self.G.edge[e[0]][e[1]]["feature"][di] = \
                    (self.G.edge[e[0]][e[1]]["feature"][di] - self.minf[di])/ \
                    (self.maxf[di] - self.minf[di])

    def preprocess_helper(self, i, nodes): 
        self.comm[i] = {}

        edges = self.G.edges(nodes)
        for e in edges:
            self.G.edge[e[0]][e[1]]["feature"] = self.edge_feature(e) 
            self.edges_involved.add(e) # used for statistics

        self.comm[i]["ein"] = [e for e in edges if e[0] in nodes and e[1] in nodes]
        self.comm[i]["eout"] = [e for e in edges if not e in self.comm[i]["ein"]]


    def preprocess_vector(self, i):
        self.comm[i]["in"] = numpy.sum([ self.G.edge[e[0]][e[1]]["feature"] for e in self.comm[i]["ein"] ], axis=0)
        self.comm[i]["cut"] = numpy.sum([ self.G.edge[e[0]][e[1]]["feature"] for e in self.comm[i]["eout"] ], axis=0)
        self.comm[i]["vol"] = 2.0 * self.comm[i]["in"] + self.comm[i]["cut"] # vol = 2 ein + eout

    def update(self, p, i, j):
        vol_i_j = numpy.inner(p, self.comm[(i,j)]["vol"])
        cut_i_j = numpy.inner(p, self.comm[(i,j)]["cut"])
        in_i_j = numpy.inner(p, self.comm[(i,j)]["in"])
         
        vol_i = numpy.inner(p, self.comm[i]["vol"])
        cut_i = numpy.inner(p, self.comm[i]["cut"])
        in_i = numpy.inner(p, self.comm[i]["in"])

        vol_j = numpy.inner(p, self.comm[j]["vol"])
        cut_j = numpy.inner(p, self.comm[j]["cut"])
        in_j = numpy.inner(p, self.comm[j]["in"])
        return vol_i_j, cut_i_j, in_i_j, vol_i, cut_i, in_i, vol_j, cut_j, in_j

    def f(self, p):
        '''
        @return the objective function 
        '''
        ret = 0
        total_negative = 0
        self.p = p
        #print "p = ", p 
        ave_w = numpy.inner(p, self.ave_edge_feature)
        print "ave_w = ", ave_w
        W = ave_w * self.G.number_of_edges()
        for i,j in self.pairs:
            vol_i_j, cut_i_j, in_i_j, vol_i, cut_i, in_i, vol_j, cut_j, in_j = self.update(p, i, j)
            m_i = (in_i)/W - (vol_i / 2.0 / W) ** 2.0
            m_j = (in_j)/W - (vol_j / 2.0 / W) ** 2.0
            m_i_j = (in_i_j)/W - (vol_i_j / 2.0 / W) ** 2.0
            if vol_i < 0 or vol_j < 0:
                total_negative += 1
            ret += sigmoid( m_i_j - m_i - m_j ) 
        var = numpy.dot( p, numpy.dot(p, self.cov_edge_feature) ) - ave_w ** 2.0
        print "f=" , (ave_w - AVE) ** 2.0, "+", LAMBDA1 * (var) , "+" ,  LAMBDA2 * ret,
        ret = (ave_w - AVE) ** 2.0 + LAMBDA1 * ( var )  +  LAMBDA2 * ret
        print "==" , ret 

        self.validate()
        return ret

    def f_der(self, p):
        '''
        @return the first order derivetive of the objective function 
        '''
        # maximum number of iterations = 25 
        if self.numiter >= 25: return numpy.zeros(self.dimension)
        self.numiter += 1
        ret = 0
        self.p = p
        ave_w = numpy.inner(p, self.ave_edge_feature)
        W = ave_w * self.G.number_of_edges()
        for i,j in self.pairs:
            vol_i_j, cut_i_j, in_i_j, vol_i, cut_i, in_i, vol_j, cut_j, in_j = self.update(p, i, j)
            m_i = (in_i)/W - (vol_i / 2.0 / W) ** 2.0
            m_j = (in_j)/W - (vol_j / 2.0 / W) ** 2.0
            m_i_j = (in_i_j)/W - (vol_i_j / 2.0 / W) ** 2.0
            #print m_i_j - m_i - m_j
            ret += sigmoid_der( m_i_j - m_i - m_j )  * \
                     ( 
                       ( (self.comm[(i,j)]["in"] * W -  in_i_j * self.sum_edge_feature)  \
                       - 0.5 * vol_i_j / W * (self.comm[(i,j)]["vol"] * W - vol_i_j * self.sum_edge_feature) )\
                       / (W ** 2.0)\
                     - ( (self.comm[i]["in"] * W -  in_i * self.sum_edge_feature)  \
                       - 0.5 * vol_i / W * (self.comm[i]["vol"] * W - vol_i * self.sum_edge_feature) )\
                       / (W ** 2.0)\
                     - ( (self.comm[j]["in"] * W -  in_j * self.sum_edge_feature)  \
                       - 0.5 * vol_j / W * (self.comm[j]["vol"] * W - vol_j * self.sum_edge_feature) )\
                       / (W ** 2.0)\
                     )
        ret = 2.0 * (ave_w - AVE) * self.ave_edge_feature + 2.0 * LAMBDA1 * ( numpy.dot(p, self.cov_edge_feature) - ave_w * self.ave_edge_feature ) * self.ave_edge_feature  + 2.0 * LAMBDA2 * ret
        return ret

    def validate(self):
        p = self.p
        count = 0
        good = 0
        W = float( self.G.number_of_edges() )
        for i,j in self.pairs:
            vol_i_j, cut_i_j, in_i_j, vol_i, cut_i, in_i, vol_j, cut_j, in_j = self.update(p, i, j)
            m_i = (in_i)/W - (vol_i / 2.0 / W) ** 2.0
            m_j = (in_j)/W - (vol_j / 2.0 / W) ** 2.0
            m_i_j = (in_i_j)/W - (vol_i_j / 2.0 / W) ** 2.0
        print "=" * 20
        print "p=", self.p
        #print "Correctness : ", ( 100.0 * good / count )
        print "=" * 20

        ave_w = numpy.inner(p, self.ave_edge_feature)
        return ave_w  

    def convert(self, path):
        '''
        convert an unweighted graph to weighted one
        '''
        self.loadTargetGraph()
        print "Convert with p=", self.p
        filename = path + self.target + ".wpairs"
        print "total #edges:", self.G.number_of_edges()
        with open(filename, "w+") as txt:
            for e in self.G.edges():
                feature =  self.edge_feature(e)
                for di in range(self.dimension - 1):
                    feature[di] = \
                        (feature[di] - self.minf[di])/ \
                        (self.maxf[di] - self.minf[di])
                w = numpy.inner(self.p, feature)
                txt.write("%d %d %f\n" % (e[0], e[1], w))
        print "write to file:", filename

if __name__ == "__main__":
    print "start"
    #algorithm = AdaptiveModularity(target = "football")
    #algorithm = AdaptiveModularity(target = "amazon")
    algorithm = AdaptiveModularity(target = "dblp")

    print "preprocessing graph"
    algorithm.preprocess()
   
    ave_w = 0
    while (ave_w < 0.2 or ave_w > 0.9):
      # initlization
      algorithm.p = numpy.zeros(algorithm.dimension)
      algorithm.p[-1] = 1
      ave_w = algorithm.validate()
      # optimization
      res = solve()
      ave_w = algorithm.validate()

    print "output weighted graph"
    algorithm.convert("./output/")

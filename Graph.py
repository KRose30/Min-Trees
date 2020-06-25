import random

from numpy.core import double

from DisjointSet import DisjointSet as ds
import math
import numpy as np
import matplotlib.pyplot as plt

class Node:

    def __init__(self, alias: int):
        self.data = None
        self.alias = alias


class Edge:

    def __init__(self, src: Node, destination: Node, w):
        self.src = src
        self.dest = destination
        self.w = w

    # Compares edges by weight
    def __lt__(self, other):
        return self.w < other.w

    def get_name(self):
        return (f"({self.src},{self.dest})")


# Implementation of graph using adjacency matrix
class Graph:

    def __init__(self):
        self.adj_matrix = []
        self.E = []
        self.V = []
        self.size = 0

    def add(self, x: Node):
        self.V.append(x)

    def connect(self, x: Node, y: Node, weight):
        new_edge = Edge(x, y, weight)
        self.E.append(new_edge)

    def make_complete_graph(self, size):
        self.size = size

        for i in range(self.size):
            self.add(Node(i))

        for i in range(self.size):
            self.adj_matrix.append([])
            for j in range(self.size):
                if i == j:
                    self.adj_matrix[i].append(0)
                else:
                    w = np.random.uniform(low=0.0, high=2.0)
                    self.adj_matrix[i].append(w)
                    # Connects vertices and adds this new edge to edge list
                    self.connect(self.V[i], self.V[j], weight=w)

    def uneven_graph(self, size):
        self.size = size

        for i in range(self.size):
            self.add(Node(i))

        for i in range(self.size):
            self.adj_matrix.append([])
            for j in range(self.size):
                if i == j:
                    self.adj_matrix[i].append(0)
                elif j == i + 1:
                    w = np.random.uniform(low=0.0, high=2.0)
                    self.adj_matrix[i].append(w)
                    self.connect(self.V[i], self.V[j], weight=w)
                else:
                    w = np.random.uniform(low=10.0, high=11.0)
                    self.adj_matrix[i].append(w)
                    self.connect(self.V[i], self.V[j], weight=w)


# Returns the minimum spanning tree using kruskal's algorithm
def kruskals(graph: Graph) -> list:
    s = sorted(graph.E)
    return spanning_tree(graph, s)


def random_tree(graph: Graph) -> list:
    s = graph.E
    random.shuffle(s)
    return spanning_tree(graph, s)


# Returns spanning tree by picking the edges in decreasing order from the given distribution
def mst_from_freq(freq: {}, graph: Graph) -> []:
    # S is a list of edges sorted in decreasing order by the amount of times they occur
    S, freq = map(list, zip(*(sorted(freq.items(), key=lambda item: item[1], reverse=True))))
    return spanning_tree(graph, S)


'''
Creates a spanning tree of a graph given a list of the graph's edges sorted in a particular order. The function utilizes
the disjoint set data structure to add each edge to the graph in the given order and check if there are cycles.
Returns a list of edges in the spanning tree
'''


def spanning_tree(graph: Graph, S: []) -> list:
    F = ds()
    spanning_tree = []

    # Make set for each node in graph and add it to the forest
    for i in range(graph.size):
        F.make_set(graph.V[i])

    # For each edge in increasing order, check whether adding the edge would make a cycle
    # If not, add edge to the minimum spanning tree
    for edge in S:
        source = F.get_tree(edge.src)
        dest = F.get_tree(edge.dest)
        src_root = F.find(source)
        dest_root = F.find(dest)

        if src_root is not dest_root:
            F.union(src_root, dest_root)
            spanning_tree.append(edge)

    return spanning_tree


def maximum_entropy(graph: Graph, n, B, show) -> list:
    """ Computes the MST of a graph by sampling from a maximum entropy distribution of the edges

        Parameters:
            graph (Graph) - Graph object on which we want to find the MST
            n (int) - number of iterations for the sampling algorithm
            B (double) - Parameter beta used when calculating acceptance probability

    """

    # Create dictionary to represent frequency of each edge
    edges_hist = {}  # Edge, integer pairs
    Tk = kruskals(graph)

    for edge in graph.E:
        edges_hist[edge] = 0

    r, l = [], []

    # Guess a tree structure T0 with w0
    T0 = random_tree(graph)

    for i in range(n):
        T0 = sample(T0, graph, edges_hist, B)
        r.append(get_variance(edges_hist))
        l.append(loss(Tk, edges_hist, graph))

    # Plot certain data
    if show:
        path= 'size{}_beta{}_iter{}'.format(graph.size, B, n)

        plot_edge_freq(edges_hist,path)
        plot_roughness(r,path)
        plot_loss(l,path)

    return mst_from_freq(edges_hist, graph)


def sample(T, graph, freq, B):
    w = total_weight(T)

    # Propose new tree
    Tn = random_tree(graph)
    wn = total_weight(Tn)

    delta_w = wn - w

    # Acceptance test
    if delta_w < 0 or math.exp(-B * delta_w) > random.random():
        add_to_hist(Tn, freq)
        T = Tn
    else:
        add_to_hist(T, freq)

    return T


def loss(Tk, freq: {}, graph):
    Tn = mst_from_freq(freq, graph)
    Tk, Tn = sorted(Tk), sorted(Tn)
    L = 0

    for i in range(graph.size - 1):
        k, n = Tk[i], Tn[i]
        L += k.w - n.w

    return L


# Takes in a list of edges and finds the sum of the edge weights
def total_weight(edges: list):
    total = 0
    for edge in edges:
        total += edge.w

    return total


def add_to_hist(T, hist):
    for edge in T:
        hist[edge] += 1


def get_variance(hist):
    return np.array([hist[e] for e in hist]).var()


def plot_roughness(r: [], path: str):
    plt.plot(r)
    plt.title("Roughness")
    plt.savefig("images/roughness_{}.png".format(path))
    plt.show()


def plot_loss(l: [], path: str):
    plt.plot(l)
    plt.title("Loss")
    plt.savefig('images/loss_{}.png'.format(path))
    plt.show()


def plot_edge_freq(freq: {}, path:str):
    plt.bar(range(len(freq)), list(freq.values()), align='center')
    plt.title("Edge Histogram")
    plt.xticks(range(len(freq)), list(element.get_name for element in freq.keys()))
    plt.savefig('images/edge_hist_{}.png'.format(path))
    plt.show()


# Function to print out list of edges in the tree and the minimum cost
def print_edges(edges: list):
    for edge in edges:
        print(f"Edge from {edge.src.alias} to {edge.dest.alias} with weight {edge.w:.4}")

    print(f"The total cost of the tree is {total_weight(edges):.5}")


def compare_trees(graph, n, beta, show):
    Tk = kruskals(graph)
    Tm = maximum_entropy(graph, n, beta, show)

    return Tk, Tm


def compare_algorithms(size, n):
    graph = Graph()
    graph.make_complete_graph(size)

    b = np.arange(0, 10.0, 0.01)
    dif = np.empty(b.size)

    for i in range(len(b)):
        Tk, Tm = compare_trees(graph, n, b[i], show=False)
        wk, wm = total_weight(Tk), total_weight(Tm)
        dif[i] = wk - wm

    plt.plot(b, dif)
    plt.savefig('final_loss_v_beta_size{}.png'.format(graph.size))
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Kruskals algroithm implementation')
    parser.add_argument('--size', type=int, default=6,
                        help='Number of vertices in the graph.')
    parser.add_argument('--beta', type=double, default=0.75,
                        help='Parameter beta for use in maximum entropy calculation of MST')
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Iterations for sampling algorithm')

    args = parser.parse_args()

    # Makes a complete graph and prints out the resulting tree
    compare_algorithms(args.size, args.iterations)

    # graph = Graph()
    # graph.uneven_graph(args.size)
    #
    # Tk, Tn = compare_trees(graph, args.iterations, args.beta, show=True)
    # print("Kruskal's")
    # print_edges(Tk)
    # print("Maximum Entropy")
    # print_edges(Tn)

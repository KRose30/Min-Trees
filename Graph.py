import random
from DisjointSet import DisjointSet as ds

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
                    w = random.random()
                    self.adj_matrix[i].append(w)
                    # Connects vertices and adds this new edge to edge list
                    self.connect(self.V[i], self.V[j], weight=w)


# Kruskal's algorithm for finding the minimum spanning tree, following psuedocode on Wikipedia
def kruskals(graph: Graph) -> list:
    F = ds()
    min_tree = []

    # Make set for each node in graph and add it to the forest
    for i in range(graph.size):
        F.make_set(graph.V[i])

    # Create a sorted list of edges
    S = sorted(graph.E)

    # For each edge in increasing order, check whether adding the edge would make a cycle
    # If not, add edge to the minimum spanning tree
    for edge in S:
        source = F.get_tree(edge.src)
        dest = F.get_tree(edge.dest)
        src_root = F.find(source)
        dest_root = F.find(dest)

        if src_root is not dest_root:
            F.union(src_root, dest_root)
            min_tree.append(edge)

    return min_tree

# Function to print out list of edges in the tree and the minimum cost
def print_edges(edges: list):
    sum = 0
    for edge in edges:
        sum+= edge.w
        print(f"Edge from {edge.src.alias} to {edge.dest.alias} with weight {edge.w}")

    print(f"The total cost of the tree is {sum}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Kruskals algroithm implementation')
    parser.add_argument('--size', type=int, default=10,
                        help='Number of vertices in the graph.')

    args = parser.parse_args()

    # Makes a complete graph and prints out the resulting tree
    graph = Graph()
    graph.make_complete_graph(args.size)
    print_edges(kruskals(graph))

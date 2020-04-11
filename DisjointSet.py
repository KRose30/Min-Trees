class Tree:
    def __init__(self):
        self.parent = self
        self.rank = 0
        self.size = 1


class DisjointSet:
    '''
    Implementation of the disjoint-set data structure following psuedocode on Wikipedia.
    A DisjointSet is a forest of elements
    '''

    def __init__(self):
        self.elements = {} # Maps each value to an element
        self.forest = [] # Contains all the elements of the disjoint set

    def get_tree(self, x) -> Tree:
        return self.elements[x]

    def make_set(self, x):
        # If x is not already present in element list, then add it as an element
        if x not in self.forest:
            new_tree = Tree()
            self.forest.append(new_tree)
            self.elements[x] = new_tree

    def find(self, x: Tree):
        # Recursively find the parent of x
        if x.parent is not x:
            x.parent = self.find(x.parent)
        return x.parent

    # Union by size
    def union(self, x: Tree, y: Tree) -> bool:
        x_root = self.find(x)
        y_root = self.find(y)

        # If x and y root are the same, then they are in the same set
        if x_root is y_root:
            return False

        # Swap roots so that tree does not become unbalanced
        if x_root.size < y_root.size:
            x_root, y_root = y_root, x_root

        # Merge trees
        y_root.parent = x_root
        x_root.size = y_root.size + x_root.size

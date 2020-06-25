Kruskal's Algorithm vs. Maximum Entropy Distribution for MST 
Author: Katherine Seeman

Attached is a program to compute minimum spanning trees using two separate
approaches. It contains an implementation of Kruskal's algorithm which is straight-
forward and follows common pseudocode found on Wikipedia. It utilizes the disjoint
set data structure and the operations make-set, find, and union.

It also contains the method to find the tree using a maximum entropy distribution
on the edges. The distribution is obtained through a Metropolis-Hastings sampling
algorithm.

The main method of Graph.py will create trees using both of these methods and print out
the resulting MSTs. The user can compare the total cost of the tree from each method.

To run:
By default, the program will run on a complete graph of 6 vertices with weights
initialized to random numbers between 0 and 2.0 and a beta of 0.75. The user can
enter arguments as follows :
--size [number of vertices]
--beta [double hyperparameter]
--iterations [iterations for sampling]
#! /opt/local/bin/python3
# -*- coding: utf-8 -*-


# Tarjan's algorithm computes the strongly connected components of a graph.
# File downloaded fro GitHub.
# In order to use the following graph algorithms, the graph should be given
# as a dictionary of the form {node:successor_list, ...}
# Example: G = {1:[2],2:[1,5],3:[4],4:[3,5],5:[6],6:[7],7:[8],8:[6,9],9:[]}


def SCC(graph):
	"""
	Tarjan's Algorithm (named for its discoverer, Robert Tarjan) is a graph theory algorithm
	for finding the strongly connected components of a graph.
	
	Based on: http://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
	"""

	index_counter = [0]
	stack = []
	lowlinks = {}
	index = {}
	result = []
	
	def strongconnect(node):
		# set the depth index for this node to the smallest unused index
		index[node] = index_counter[0]
		lowlinks[node] = index_counter[0]
		index_counter[0] += 1
		stack.append(node)
	
		# Consider successors of `node`
		try:
			successors = graph[node]
		except:
			successors = []
		for successor in successors:
			if successor not in lowlinks:
				# Successor has not yet been visited; recurse on it
				strongconnect(successor)
				lowlinks[node] = min(lowlinks[node],lowlinks[successor])
			elif successor in stack:
				# the successor is in the stack and hence in the current strongly connected component (SCC)
				lowlinks[node] = min(lowlinks[node],index[successor])
		
		# If 'node' is a root node, pop the stack and generate an SCC
		if lowlinks[node] == index[node]:
			connected_component = []
			
			while True:
				successor = stack.pop()
				connected_component.append(successor)
				if successor == node: break
			component = tuple(connected_component)
			# storing the result
			result.append(component)
	
	for node in graph:
		if node not in lowlinks:
			strongconnect(node)
	
	return result


# ********* PATCH ********* #
def is_an_SCC(scc):
	""""
	Sometimes the algo finds scc which are isolated nodes.
	They cause problems when applying the algorithm below.
	So we remove them...
	"""
	if len(scc) > 1:
		return True
	else:
		return False


# There's a problem in the algo:
# When a node doen not belong to an SCC, then it is considered as a single node SCC automatically!
# I still don't k ow where is the prolem, but I make a function to patch this.
def check_scc_1(SCC, G):
	"""
	checks if what is found as a single node SCC in a graph G is really an SCC or not
	Recall that SCC is a tuple of nodes and 
	G is given in its dico form
	"""
	node = SCC[0]
	if G[node] == [node]: # i.e., node is 1-point attractor, which would mean that it is truly an SCC
		return True
	else:
		return False
# ******* END PATCH ******* #




# ********************************************************** #
# ********************************************************** #
# ********************************************************** #



# Johnson's algorithm computes the simple cycles of a graph.
# It is is based on Tarjan's algorithm (above).
# File downloaded fro GitHub and patched.


from collections import defaultdict
from itertools import chain



# ********* PATCH ********* #
# MY PREMILINARY FUNCTION
def reorder(c):
	"""reorders a cycle so that it always begins with the node of smalles values"""
	n = c.index(min(c))
	c2 = [c[(j + n) % len(c)] for j in range(len(c))]
	return c2
# ******* END PATCH ******* #



def grab_cycles(graph): # expects graph to be strongly connected
	cycles = set()
	blocked = defaultdict(bool)
	B = defaultdict(list)
	stack = []

	def find_cycles(v, s):
		f = False
		stack.append(v)
		blocked[v] = True
		for w in graph[v]:
			if w == s:
				# ***** PATCH ***** #
				#cycles.add(tuple(stack))
				if tuple(reorder(stack)) not in cycles:
					cycles.add(tuple(reorder(stack)))
				# *** END PATCH *** #
				f = True
			elif not blocked[w]:
				if find_cycles(w, s):
					f = True
		if f:
			unblock(v)
		else:
			for w in graph[v]:
				if v not in B[w]:
					B[w].append(v)
		stack.remove(v)
		return f

	def unblock(node):
		blocked[node] = False
		Bnode = B[node]
		while Bnode:
			w = Bnode.pop(0)
			if blocked[w]:
				unblock(w)

	# *** PATCH ***
	#start = min(graph) # arbitrary
	#find_cycles(start, start)
	# END ORIGINAL - BEGINNING MY CODE
	for node in graph.keys():
		find_cycles(node, node)
	# *** END PATCH ***
	return tuple(cycles)

def get_subgraph(graph, vertices):
	#returns subgraph induced by vertices
	return {vert : {child
					for child in graph[vert]
					if child in vertices}
			for vert in vertices}

def get_elementary_cycles(graph):
	#returns all elementary cycles in a graph
	return tuple(chain.from_iterable(grab_cycles(get_subgraph(graph, scc)) for scc in SCC(graph)))




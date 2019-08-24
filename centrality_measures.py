import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import operator
import seaborn as sns

def makeGraph(csvFile):
	reader = csv.reader(open(csvFile, 'rU'), dialect=csv.excel_tab, 
						delimiter=",")
	x = list(reader)
	dataFile = np.array(x).astype("float")
	graph = nx.from_numpy_matrix(dataFile)
	return graph

def degree(graph):
	'''returns list'''
	degree = []
	i = 0
	while i<nx.number_of_nodes(graph):
		degree.append(graph.degree[i])
		i += 1
	return degree

def getMinCentrality(centrality):
	location = min(centrality.iteritems(), 
				key=operator.itemgetter(1))[0]
	lowest = centrality[location]
	return location, lowest

def getMaxCentrality(centrality):
	location = max(centrality.iteritems(), 
				key=operator.itemgetter(1))[0]
	highest = centrality[location]
	return location, highest;

def eigenvector(graph):
	'''returns dictionary'''
	centrality = nx.eigenvector_centrality(graph)
	return centrality

def betweenness(graph):
	'''returns dictionary'''
	between = nx.betweenness_centrality(G=graph, normalized=True)
	return between 

def closeness(graph):
	'''returns dictionary'''
	closeness = nx.closeness_centrality(G=graph, normalized=True)
	return closeness

def flow(graph):
	'''returns dictionary'''
	flow = nx.current_flow_betweenness_centrality(G=graph, normalized=True)
	return flow

def pageRank(graph):
	'''returns dictionary'''
	page = nx.pagerank(G=graph)
	return page

def hits(graph):
	'''returns two-tuple of dictionaries'''
	hub, authority = nx.hits(G=graph, normalized=True)
	return hub, authority

japGraph = makeGraph("Japanais_littlneck_clam.csv")
pacGraph = makeGraph("Pacific_littlneck_clam.csv")
redGraph = makeGraph("Red_rock_crab.csv")

Japanais_littlneck_clamCentrality = eigenvector(japGraph)
Pacific_littlneck_clamCentrality = eigenvector(pacGraph)
Red_rock_crabCentrality = eigenvector(redGraph)

getMinCentrality(Japanais_littlneck_clamCentrality)
getMaxCentrality(Japanais_littlneck_clamCentrality)

japBetween = betweenness(japGraph)
japFlow = flow(japGraph)
japPage = pageRank(japGraph)
japHub, japAuthority = hits(japGraph)

excel = pd.DataFrame([japBetween, japFlow, japPage, japHub, japAuthority], 
			index=["Betweenness", "Flow", "Page Rank", "Hub", "Authority"])
excel.to_excel('Japanese.xlsx')

# helix = pd.read_csv('heatmap_japanese.csv')
# ax = sns.heatmap(helix, cmap = 'Blues_r')
# plt.show()

# helix1 = pd.read_csv('heatmap_crab.csv')
# ax = sns.heatmap(helix1, cmap = 'Blues_r')
# plt.show()

# helix2 = pd.read_csv('heatmap_pacific.csv')
# ax = sns.heatmap(helix2, cmap = 'Blues_r')
# plt.show()

# helix3 = pd.read_csv('output.csv')
# ax=sns.heatmap(helix3, cmap='Blues_r')
# ax.set_yticklabels(['Japanese LC', 'Pacific LC', 'RR Crab'], va='center')
# plt.show()

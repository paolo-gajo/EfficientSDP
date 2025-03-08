"""
	Visualize the output flow graphs, as well as create compact flow graphs
	using pydot and graphviz!
"""

"""
	This will take parser output in CoNLLU format and plot flow graphs out of it. 
"""

from stepparser.utils import make_dir, is_dir, build_augmented_recipe
from tqdm import tqdm
import os
import pydot, copy, spacy, string
import networkx as nx
import torch
import matplotlib as plt
import seaborn as sns
nlp = spacy.load('en_core_web_md')

def remove_invalid_characters(sentence):
	"""
		remove invalid characters which are not printable!
	"""
	valid_chars = string.printable.replace("\"", "") 
	return ''.join([char for char in sentence if char in valid_chars ])

def merge_nodes(G,nodes, new_node, **attr):
	"""
	Merges the selected `nodes` of the graph G into one `new_node`,
	meaning that all the edges that pointed to or from one of these
	`nodes` will point to or from the `new_node`.
	attr_dict and **attr are defined as in `G.add_node`.
	"""
	
	G.add_node(new_node, **attr) # Add the 'merged' node
	# for node_id, att_d in G.nodes(data=True):
		# print(node_id, att_d)
	thingsToChange = []
	for n1, n2, data in G.edges(data=True):
		# For all edges related to one of the nodes to merge,
		# make an edge going to or coming from the `new gene`.
		# print(n1, n2)
		if n1 in nodes:
			#print('n1')
			#print(new_node, n2)
			thingsToChange.append((new_node, n2))
			#G.add_edge(new_node,n2)
		elif n2 in nodes:
			#print('n2')
			#print(new_node, n2)
			#G.add_edge(n1,new_node)
			thingsToChange.append((n1, new_node))
	for edge in thingsToChange:
		G.add_edge(edge[0], edge[1])
	for n in nodes: # remove the merged nodes
		G.remove_node(n)
	# print(G.nodes)

def merge_BIO(nodes):
	
	merged_nodes = []

	for node in nodes:
		nidx = node[0][0]
		node_name = node[0][1].split('_')[0] + '_' + '-'.join([n[1].split('_')[1] for n in node]) 
		node_label = node[0][2].split('-')[1]
		merged_nodes.append([(nidx, node_name, node_label)])
		
	return merged_nodes

def get_graph_root(tempG):
	graph_roots = [n for n,d in tempG.in_degree() if d==0]
	if len(graph_roots) > 1:
		print("Alert: multiple roots!: {}".format(graph_roots))
	graph_root = graph_roots[0]
	return graph_root

def get_compact_graph(orig_r, node_list, edges):
	
	FullG = nx.DiGraph()
	nodes = merge_BIO(node_list)

	for n in nodes:

		att = n[0]
		print(att)
		nk = int(att[0])
		step_id = int(att[1].split("_")[0])
		word = att[1].split("_")[1]
		# node_type = att[2].split("-")[1]
		node_type = att[2]
		print(step_id, word, node_type)
		FullG.add_node(nk, step_id=step_id, word=word, node_type=node_type)

	nx_edges = []
	for e in edges:
		nx_edges.append((e[0], e[1]))
	FullG.add_edges_from(nx_edges)

	orig_FullG = copy.deepcopy(FullG)
	# find root
	graph_roots = [n for n,d in FullG.in_degree() if d==0]
	if len(graph_roots) > 1:
		print("Alert: multiple roots!")
	
	graph_root = graph_roots[0]
	
	# build graph by steps
	G = copy.deepcopy(FullG)
	CollapsedG = nx.DiGraph()
	collapsed_edges = []
	
	postorder_nodes = list(nx.dfs_postorder_nodes(G, graph_root))
	print(len(postorder_nodes))
	print(postorder_nodes)
	for idx, node_id in enumerate(postorder_nodes):
		att_d = G.nodes[node_id]
		step_id = int(att_d['step_id'])
		
		if att_d['node_type'] == 'Ac':
			print("")
			print(node_id, G.nodes[node_id])
			
			# find predecessor action node in unvisited nodes
			unvisited = postorder_nodes[idx+1:]
			# we can do this because each node has one incoming edge
			Ac_found = False
			Ac_pred = node_id
			if node_id != graph_root:
				print("searching")
				print(list(G.predecessors(node_id)))
				pre_node_id = list(G.predecessors(node_id))[0]
				if G.nodes[pre_node_id]['node_type'] == 'Ac':
					Ac_found = True
					Ac_pred = pre_node_id
				while pre_node_id != graph_root:
					print(pre_node_id)
					pre_node_id = list(G.predecessors(pre_node_id))[0]
					if G.nodes[pre_node_id]['node_type'] == 'Ac':
						Ac_found = True
						Ac_pred = pre_node_id
					else:
						continue
					if Ac_found is True:
						break
				print(Ac_pred, pre_node_id, node_id, Ac_found)
				print(node_id, orig_r[step_id-1], step_id)
				CollapsedG.add_node(node_id, name = orig_r[step_id-1], step_id=step_id)
				if Ac_found is True:
					collapsed_edges.append((Ac_pred, node_id))
			else:
				print("root")
				print(node_id, orig_r[step_id-1], step_id)
				CollapsedG.add_node(node_id, name = orig_r[step_id-1], step_id=step_id)
	print(collapsed_edges)
	CollapsedG.add_edges_from(collapsed_edges)



	# combine redundant nodes
	# get a dictionary of step_id to nodes
	CG = copy.deepcopy(CollapsedG)
	CG_graph_root = get_graph_root(CG)
	postorder_nodes = list(nx.dfs_postorder_nodes(CG, CG_graph_root))
	# print(len(postorder_nodes))
	# print(postorder_nodes)
	id2node = dict()
	for idx, node_id in enumerate(postorder_nodes):
		att_d = G.nodes[node_id]
		step_id = att_d['step_id']
		if step_id in id2node:
			id2node[step_id].append(node_id)
		else:
			id2node[step_id] = [node_id]

	# merge nodes in the same step id
	for k, v in id2node.items():
		if len(v) > 1:
			new_node_id = '-'.join(str(n) for n in v)
			# print(k, v)
			
			#attr_dict['merged_nodes'] = '-'.join(str(n) for n in v)
			
			merge_nodes(CG, v, new_node_id, **CG.nodes[v[0]])

	# for node_id, att_d in CollapsedG.nodes(data=True):
	#     print(node_id, att_d)


	ridx = 0
	recipe_nodes = nodes
	recipe_edges = nx_edges
	graph = pydot.Dot("recipe_graph", graph_type='digraph')

	for node in node_list:
		nidx = node[0][0]
		label = ' '.join([n[1] for n in node]) + " (" + node[0][2].split('-')[1] + ")"

		## print actions in hexagon and other nodes in oval shape
		if "(Ac)" in label:
			shape = 'hexagon'
		else:
			shape = 'oval'
		graph.add_node(pydot.Node(f"{nidx}", shape = shape, label='"' + label + '"'))


	for edge in edges:
		src_idx, trg_idx, rel = edge
		graph.add_edge(pydot.Edge(f"{src_idx}", f"{trg_idx}", color="blue", label=rel))
		

	graph = pydot.Dot("recipe_graph", graph_type='digraph')
	graph_edges_list = []

	nidx_2_step_idx = {}
	for k, v in CG.nodes(data=True):
		print(k,v)
		nidx_2_step_idx[k] = v['step_id']

		nidx = k
		label = str(v['step_id']) + "-" + v['name']
		label = remove_invalid_characters(label)
		graph.add_node(pydot.Node(f"{nidx}", shape="oval", label=label))

	for edge in CG.edges(data=True):
		trg_idx, src_idx, _ = edge
		step_id_edge = [int(nidx_2_step_idx[src_idx]) - 1, int(nidx_2_step_idx[trg_idx]) - 1]

		if step_id_edge not in graph_edges_list and list(reversed(step_id_edge)) not in graph_edges_list:
			graph_edges_list.append(step_id_edge)
		graph.add_edge(pydot.Edge(f"{src_idx}", f"{trg_idx}", color="blue"))
		

	return graph, graph_edges_list

def get_nodes_and_edges(recipe, augmented_recipe):
	
	nodes = {}
	edges = []
	valid_nodes = {}

	assert len(recipe) == len(augmented_recipe)
	
	node_list = []
	open_tag = False
	
	## let's get the nodes
	for i, recipe_step in enumerate(recipe):
		recipe_split = recipe_step.split('\t')
		assert recipe_split[1] in augmented_recipe[i]
		if recipe_split[4] != 'O':
			if not(recipe_split[4].startswith('I-') and open_tag):
				open_tag = True
				node_list.append([])
			node_list[-1].append((recipe_split[0], augmented_recipe[i], recipe_split[4]))
		else:
			# node_list.append([])
			# node_list[-1].append((recipe_split[0], augmented_recipe[i], '-' + recipe_split[4]))
			open_tag = False


		nodes[recipe_split[0]] = augmented_recipe[i] + f' ({recipe_split[4]})'
		valid_nodes = [int(nd[0][0]) for nd in  node_list]

	## let's get the edges
	for recipe_step in recipe:
		
		recipe_split = recipe_step.split('\t')


		if recipe_split[6] == '0':
			continue
		else:        
			source_id = int(recipe_split[6])
			dest_id = int(recipe_split[0])
			relation = recipe_split[7]
			if source_id in valid_nodes and dest_id in valid_nodes:
				edges.append((source_id, dest_id, relation))
			# if int(recipe_split[6]) not in valid_nodes:
			#     valid_nodes[int(recipe_split[6])] = nodes[recipe_split[6]]    
			# if int(recipe_split[0]) not in valid_nodes:
			#     valid_nodes[int(recipe_split[0])] = nodes[recipe_split[0]]

	return node_list, edges

def plot_dot_from_nodes_and_edges(nodes, edges):
	
	graph = pydot.Dot("recipe_graph", graph_type='digraph')

	for node in nodes:
		nidx = node[0][0]
		label = ' '.join([n[1] for n in node]) + " (" + node[0][2].split('-')[1] + ")"
		## print actions in hexagon and other nodes in oval shape
		if "(Ac)" in label:
			shape = 'hexagon'
		else:
			shape = 'oval'
		graph.add_node(pydot.Node(f"{nidx}", shape = shape, label='"' + remove_invalid_characters(label) + '"'))

	for edge in edges:
		dest_idx, src_idx, rel = edge
		graph.add_edge(pydot.Edge(f"{src_idx}", f"{dest_idx}", color="blue", label='"' + remove_invalid_characters(rel) + '"'))

	return graph

def plot_single_recipe(recipe, augmented_recipe, recipe_as_list_of_steps, save_path):
	"""
		This function will take recipe (as a list in CoNLLU format) and augmented recipe,
		and save graphs in dot and PNG format at appropriate place.
	"""

	## let's get nodes and edges for this flow graph
	nodes, edges = get_nodes_and_edges(recipe, augmented_recipe)

	## plot the graph in dor format using nodes and edges
	graph = plot_dot_from_nodes_and_edges(nodes, edges)
	save_path = os.path.join(f'{save_path}.png')
	save_path_dot = os.path.join(f'{save_path}.dot')
	output_raw_dot = graph.to_string()
	graph.write_raw(save_path_dot) 
	graph.write_png(save_path) 

	## build a compact instruction level graph
	try:
		compact_graph, compact_graph_edges = get_compact_graph(recipe_as_list_of_steps, nodes, edges)
		save_path_compact = os.path.join(f'{save_path}_compact.png')
		save_path_dot_compact = os.path.join(f'{save_path}_compact.dot')
		output_raw_dot_compact = compact_graph.to_string()
		compact_graph.write_png(save_path_compact)
		compact_graph.write_raw(save_path_dot_compact)
	except:
		pass
	
	return save_path

def plot_from_conllu(output_file, save_dir):
	"""
		Output file: Output CoNLLU file with results
		save_dir: Directory where recipes are to be saved.
	"""
	## make directory to save the file
	if not is_dir(save_dir):
		make_dir(save_dir)

	## recipes
	recipes = open(output_file, "r", encoding="utf-8").read().split('\n\n')

	## split recipes
	recipes = [recipe.split('\n') for recipe in recipes if len(recipe) > 0]
	
	## let's go through recipes, plot and save them 1 at a time
	for i, recipe in enumerate(tqdm(recipes)):
		save_path = os.path.join(save_dir, f'{i+1}'.zfill(5))
		augmented_recipe, recipe_as_list_of_steps = build_augmented_recipe(recipe)
		plot_single_recipe(recipe, augmented_recipe, recipe_as_list_of_steps,  save_path)

def save_heatmap(matrix, filename="heatmap.pdf", cmap="viridis"):
    """
    Saves a heatmap of a square matrix as a PDF.

    Parameters:
    - matrix (torch.Tensor or np.ndarray): Square matrix to visualize.
    - filename (str): Output filename (default: "heatmap.pdf").
    - cmap (str): Matplotlib colormap (default: "viridis").
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()  # Convert PyTorch tensor to NumPy

    # if matrix.shape[0] != matrix.shape[1]:
    #     raise ValueError("Input matrix must be square.")

    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap=cmap, aspect="auto")
    plt.colorbar()
    plt.title("Heatmap")
    
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()

def show_attentions(model_out):
    # Extract attention matrices (Shape: num_layers, batch_size, num_heads, seq_len, seq_len)
    attentions = model_out.attentions

    # Get attention maps from the last layer
    last_layer_attentions = attentions[-1][0]  # Shape: (num_heads, seq_len, seq_len)
    num_heads = last_layer_attentions.shape[0]

    # Plot heatmaps for each attention head in the last layer
    fig, ax = plt.subplots(figsize=(20, 15))  # 12 heads (4 rows × 3 cols)

    sns.heatmap(last_layer_attentions[0].detach().cpu().numpy(), ax=ax, cmap="Blues")

    plt.tight_layout()
    plt.savefig('model_attentions.pdf', format='pdf')
'''

The following code will contain the functions, in some organized fashion TBD. 

*Table of Conents here?*

'''

import numpy as np
import networkx as nx
import itertools
import random
import scipy


def hp_decay_model(t,y,curr,const_state,r):
    '''
    This is a modification to the HP model, with a decay term.
    
    INPUTS
    t: Total time simulation vector, not used in this function
    y: State variables array [x], where each state variable is size N 
    v: Input voltage at time t, size Nx1 
    const: List of constants [μ,D,τ]
    r: Threshold resistances [R_on, R_off]

    OUTPUTS:
    states: State values
    
    '''
    μ, d, τ = np.vsplit(const_state,3)
    r_on, r_off = np.vsplit(r,2)

    f = np.where(curr<0, 1-(y-1)**2, 1-y**2)
    #f = 1 - (2*y -1)**2     # For comparison with Marcus' package
    
    dx = ((μ*r_on*curr)/(d*d))*f - y/τ
    
    return np.asarray(dx)



def matrix_A(G,node_list,source_nodes,drain_nodes):
    '''
    This function returns the A matrix
    '''
    
    n = len(node_list)
    m = len(source_nodes)

    L = nx.linalg.laplacian_matrix(G, nodelist=node_list, weight='Effective Conductance').astype(float)
    
    for node in drain_nodes:     # Ground reference node
        idx = node_list.index(node)
        L = L.tolil()     # Python suggested this

        L[idx,:] = 0
        L[:,idx] = 0
        L[idx,idx] = 1

    B = scipy.sparse.dok_matrix((n,m),dtype=float)
    '''
    for idx, node in enumerate(node_list):
        if G.nodes[node].get("Type") == "Source":
            if num_sources == 1:
                B[idx,0] = -1.0
        elif G.nodes[node].get("Type") == "Drain":
            if num_sources == 1:
                B[idx,0] = 0.0
    '''
    for idx, node in enumerate(source_nodes):
        node_idx = node_list.index(node)
        B[node_idx,idx] = -1
    C = B.T
    D = np.zeros((m,m))
    A = scipy.sparse.bmat([[L.tocsr(),B],[C,D]])

    return A.tocsr()


def matrix_z(G,node_list,source_nodes,drain_nodes,I_prev=None,graph=None):
    '''
    This function returns the z matrix at time t
    '''

    n = len(node_list)     # Number of nodes
    i = np.zeros(n, dtype=float)

    if I_prev is not None and graph is not None:
        edge_n1 = graph['Node 1 Indices']
        edge_n2 = graph['Node 2 Indices']
        
        np.add.at(i, edge_n1, -I_prev[0,:])
        np.add.at(i, edge_n2, +I_prev[0,:])

        # Make sure drain nodes have no current and are grounded
        node_idx = graph["Node Indices"]
        for node in drain_nodes:
            i[node_idx[node]] = 0.0
            
    
    e = []

    for node in source_nodes:
        e.append(G.nodes[node].get("Voltage",0.0))
    e = np.asarray(e, dtype=float)
    
    z = np.concatenate([i,e])
    
    return z


def node_solver(G, node_list,source_nodes,drain_nodes, I_prev=None, graph=None):
    '''
    Solve matrix equation for node voltages
    '''
    
    A = matrix_A(G,node_list,source_nodes,drain_nodes)
    z = matrix_z(G,node_list,source_nodes,drain_nodes,I_prev=I_prev,graph=graph)

    x = scipy.sparse.linalg.spsolve(A,z)
    return x

def memristor_voltages(G,node_list,source_nodes,drain_nodes, I_prev=None, graph=None):
    # Each memristor is an edge
    # The voltage across the memristor the is voltage difference between the nodes
        
    x = node_solver(G, node_list,source_nodes,drain_nodes, I_prev=I_prev, graph=graph)     
    node_voltages = x[:len(node_list)]     # Node voltages

    nx.set_node_attributes(G,{node_list[idx]: {"Voltage": node_voltages[idx]} for idx in range(len(node_list))})     # Setting voltages to nodes
    
    voltages_memristors = {}
    
    for n1,n2,attr in G.edges(data=True):
        # Have to call the voltages at each node
        v_diff = G.nodes[n1]["Voltage"] - G.nodes[n2]["Voltage"]
        voltages_memristors[(n1,n2)] = v_diff
        attr['Voltage Difference'] = v_diff
    return voltages_memristors, node_voltages

def evolution(t,dt, y, v_input, const_state, G, model, r, node_list, q_prev, graph):
    # Each memristor needs the states, and const

    edge_list = graph["Edge List"]
    source_nodes = graph["Source Nodes"]
    #source_nodes_idx = [graph["Node Indices"][source] for source in source_nodes]
    drain_nodes = graph["Drain Nodes"]
    
    if model == "Chen":
        for (n1,n2), state in zip(edge_list, y.T):        
            #print(state.shape)
            #print(state)
            G.edges[n1,n2]['x'],G.edges[n1,n2]['epsilon'],G.edges[n1,n2]['tau'] = state
            G.edges[n1,n2]["Conductance"] = memductance_function(G.edges[n1,n2]['x'],r)
    else:

        g_on = 1/(graph['R_on']*y + 1e-30)
        g_off = 1/(graph['R_off']*(1 - y) + 1e-30)
        c = graph['C_on']*y

        den = g_on + g_off + (c/dt)
        g_eff = g_on*(g_off + (c/dt))/den

        I_prev = g_on*(q_prev/dt)/den
        for j, (n1,n2) in enumerate(edge_list):
            G.edges[n1,n2]['x'] = y[0,j]
            G.edges[n1,n2]["Effective Conductance"] = g_eff[0,j]     # g_eff.shape = (1,len(edge_list))
    
    for idx, node in enumerate(source_nodes):
        G.nodes[node]["Voltage"] = v_input[idx]
    
    voltages, node_voltages = memristor_voltages(G,node_list,source_nodes,drain_nodes,I_prev=I_prev,graph=graph)     # node voltages and memristor voltages

    v_n1 = node_voltages[graph['Node 1 Indices']]
    v_n2 = node_voltages[graph['Node 2 Indices']]

    v_int = (v_n1*g_on + v_n2*(g_off + (c/dt)) + (q_prev/dt))/den     # Internal node
    mem_curr = g_on*(v_n1 - v_int)

    q_new = c*(v_int-v_n2)
    
    v = np.array([voltages[(n1,n2)] for (n1,n2) in edge_list])
    
    if model == "HP":

        # Use the HP function
        dy =  hp_model(t,y,mem_curr,const_state,r)
    elif model == "HP Decay":
        dy = hp_decay_model(t,y,mem_curr,const_state,r)
    elif model == "Chen":
        dy = chen_model(t,y,mem_curr,const_state,r)
    elif model == "Linear":
        dy = 0
    return dy, mem_curr, v, q_new

def graph_info(G, const_state, r, const_cap):
    node_list = list(G.nodes())
    # Dictionary to keep track of the nodes and their indices
    node_idx = {node: i for i, node in enumerate(node_list)}

    source_nodes = [node for node in node_list if G.nodes[node].get("Type") == "Source"]
    drain_nodes = [node for node in node_list if G.nodes[node].get("Type") == "Drain"]    
    
    edge_list = list(G.edges())
    num_edges = len(edge_list)

    edge_n1_idx = np.array([node_idx[n1] for n1, n2 in edge_list], dtype=np.int32)
    edge_n2_idx = np.array([node_idx[n2] for n1, n2 in edge_list], dtype=np.int32)

    # Assume that the parameters are sent in as arrays
    μ, D, τ = np.vsplit(const_state,3)
    r_on, r_off = np.vsplit(r,2)
    A, ε_r = np.vsplit(const_cap,2)

    # Some functions of parameters 
    c_on = A*ε_r*8.85e-12/D
    k = μ/(D*D)

    graph = {
        'Node List': node_list,
        'Node Indices': node_idx,
        'Source Nodes': source_nodes,
        'Drain Nodes': drain_nodes,
        'Edge List': edge_list,
        'Number of Edges': num_edges,
        'Node 1 Indices': edge_n1_idx,
        'Node 2 Indices': edge_n2_idx,
        'R_on': r_on,
        'R_off': r_off,
        'C_on': c_on,
        'k': k
    }
    return graph

def param_network(N, params, pert_step, max_pert, idx_array):
    params = np.asarray(params)
    P = params.size
    param_array = np.zeros((N, P), dtype=float)
    param_array[:] = params  # start with baseline for every device

    # percentage perturbation per device in [-max_pert, max_pert] stepping pert_step
    choices = np.random.choice([-1, 1], size=N)
    steps = np.random.randint(0, int(max_pert / max(pert_step, 1)) + 1, size=N)
    pert_perc = choices * steps * pert_step / 100.0  # e.g. [-0.15, 0.0, 0.2] etc

    for i in range(N):
        for idx in idx_array:
            param_array[i, idx] = params[idx] * (1.0 + pert_perc[i])
    return param_array.T

def param_normal_network(N,params,var):
    '''
    Take the values of the params and make them the means and then make variance a percentage?
    '''
    params = np.asarray(params)
    P = params.size
    #param_array = np.zeros((N, P), dtype=float)
    #param_array[:] = params  # start with baseline for every device

    # percentage perturbation per device in [-max_pert, max_pert] stepping pert_step
    param_norm_array = np.zeros((N, P), dtype=float)
    for idx,param in enumerate(params):
        param_norm_array[:,idx] = np.random.normal(loc=param,scale=param*var,size=N)
    param_norm_array[0,:] = params
    return param_norm_array.T

def shift(t,v,nu,N):
    shifts = np.zeros(N)
    v_list = np.zeros((N,len(t)))
    for i in range(N):
        shift = nu*np.random.random_sample() # Random number [0,1]
        shift_idx = round(shift/(t[1] - t[0]))
        shift_list = np.zeros(shift_idx)
        v_shifted = np.concatenate((shift_list,v))[:len(t)]
        shifts[i] = shift_idx
        v_list[i] = v_shifted
    return v_list, shifts

def random_clustered_ring_graph(num_cluster,cluster_size,avg_degree,max_degree,max_connecting_nodes):
    G = nx.Graph()
    for i in range(num_cluster):
        nodes = np.arange(i * cluster_size, (i + 1) * cluster_size)
        possible_edges = itertools.combinations(nodes, 2)
        possible_edges = list(possible_edges)
        ## I have to make sure that each node is mentioned at least once
        degree_array = np.minimum(np.random.poisson(avg_degree, size=cluster_size),max_degree - 1) # Within the community
        #print(degree_array)
        seq_check = list(degree_array.copy())
        seq_check.sort(reverse=True)
        #print(seq_check)
        if sum(degree_array)%2 != 0:     # Sum of degrees has to be even 
            for idx in range(cluster_size):
                if degree_array[idx] < cluster_size:
                    degree_array[idx] += 1
                    #print("worked!", degree_array[idx])
                    break
        #print('degree array: ',degree_array)
        
        while True:

            edges = []
            np.random.shuffle(possible_edges)
            
            current_degree = {n: 0 for n in nodes}
            for u, v in possible_edges:
                #print(u,v)
                if (current_degree[u] < degree_array[u-i*cluster_size] and current_degree[v] < degree_array[v-i*cluster_size]):
                    edges.append((u,v))
                    current_degree[u] += 1
                    current_degree[v] += 1
                if all([current_degree[n] == degree_array[n-i*cluster_size] for n in nodes]):
                    #print('For loop breaks')
                    break
            #print(current_degree,degree_array)        
            cluster = nx.Graph()
            cluster.add_edges_from(edges)
            
            if nx.is_connected(cluster):
                break
        G.add_edges_from(edges)

        nx.set_node_attributes(G, {n: {"Type": "Cluster", "Cluster": i+1} for n in nodes})

    connecting_nodes = [max(G.nodes)+(i+1) for i in range(num_cluster)]
    G.add_nodes_from(connecting_nodes, Type="Connecting")
    #print('connecting nodes: ',connecting_nodes)

    for i in range(num_cluster):
        #print(G.nodes[i * cluster_size + 1])
        #print(G.nodes[(i + 1) * cluster_size % (num_cluster * cluster_size)])
        G.add_edge(i * cluster_size + 1, connecting_nodes[i])
        G.add_edge(connecting_nodes[i], (i + 1) * cluster_size % (num_cluster * cluster_size))

    if len(list(nx.connected_components(G))) > 1:     # Keep largest component
        comps = list(nx.connected_components(G))
        largest_comp = max(comps,key=len)
        G.remove_nodes_from([node for node in G if node not in largest_comp])     # Remove isolated nodes
        print("Removing nodes (isolated components): ",[comp for comp in comps if comp is not largest_comp])
    
    if len([node for node, degree in G.degree() if degree < 2]) > 0:     # Remove nodes with k < 2
        remove_nodes = [node for node, degree in G.degree() if degree < 2]
        print("Removing nodes (less than 1 degree): ",remove_nodes)
        G.remove_nodes_from(remove_nodes)

    return G

def random_modular_graph(num_cluster,cluster_size,avg_degree,max_degree,connect_list,max_connecting_nodes=2,
                                series_or_parallel=None,num_connections=None,ran_cluster_connect=None):
    '''
    This is an updated version of the code for my ring graph, but with more user creation
    
    ''' 
    G = nx.Graph()
    for i in range(num_cluster):
        nodes = np.arange(i * cluster_size, (i + 1) * cluster_size)
        possible_edges = itertools.combinations(nodes, 2)
        possible_edges = list(possible_edges)
        # I have to make sure that each node is mentioned at least once
        degree_array = np.minimum(np.random.poisson(avg_degree, size=cluster_size),max_degree - 1) # Within the community

        if sum(degree_array)%2 != 0:     # Sum of degrees has to be even 
            for idx in range(cluster_size):
                if degree_array[idx] < cluster_size:
                    degree_array[idx] += 1
                    break
        
        while True:
            edges = []
            np.random.shuffle(possible_edges)           
            current_degree = {n: 0 for n in nodes}

            for u, v in possible_edges:
                if (current_degree[u] < degree_array[u-i*cluster_size] and current_degree[v] < degree_array[v-i*cluster_size]):
                    edges.append((u,v))
                    current_degree[u] += 1
                    current_degree[v] += 1
                    
                if all([current_degree[n] == degree_array[n-i*cluster_size] for n in nodes]):
                    break
                   
            cluster = nx.Graph()
            cluster.add_edges_from(edges)
            
            if nx.is_connected(cluster):
                break

        G.add_edges_from(edges)
        nx.set_node_attributes(G, {n: {"Type": "Cluster", "Cluster": i+1} for n in nodes})


    # Define connecting nodes
    # Let's get user to define how many modules, which ones are connected (i.e. tuple: (1,2) ), and modules are numbered from left-to-right and top-to-bottom
    # User can have random number of connections (with maximum being max_connecting_nodes) 
    ## for each tuple combination or set number (i.e. define num_connections=True for random amounts)
    # Let 'connections' be a list of tuples (i.e. [(1,2),(1,3),...]), must handle for duplicate connections (i.e. [(1,2),...,(1,2),...], or [(1,2),...,(2,1),...])

    # This cleanup of the connect_list works better for smaller amount of modules
    clean_up = [tuple(sorted(connect)) for connect in connect_list if connect[0] != connect[1]]    # sort tuples and remove redundant self-connections
    new_connect_list = list(set(clean_up))     # remove duplicates using set()

    # Number of connections for each connection defined
    if num_connections:    # User input of number array
        if len(new_connect_list) != len(num_connections):
            raise Exception(f"Connections and number of connections are not the same dimension: {len(new_connect_list)} and {len(num_connections)}",
                            "\nThe original connection list may have been cleaned up:",f"\n\t User input: {connect_list} \n\t Cleaned list: {new_connect_list}")
        
        else:
            num_connect_array = num_connections.copy()
            
    else:    # Random number of connections with a max of user input
        num_connect_array = np.random.randint(1, max_connecting_nodes + 1, size=len(new_connect_list))
        
    # Define new nodes to connect modules
    connecting_nodes = [max(G.nodes)+(i+1) for i in range(sum(num_connect_array))]
    G.add_nodes_from(connecting_nodes, Type="Connecting")

    # Connecting the clusters together
    # Note that this can choose random nodes within the clusters to connect (which can also be the same nodes with connections in series/parallel, or 
    ## different cluster connecting nodes), choose the same nodes for each connection in series/parallel
    if ran_cluster_connect:    # Will work on user input later
        raise Exception('Must be "mix", "series", or "parallel"')
    
    else:  # Random nodes from modules are chosen
        count = 0
        
        for connect, num in zip(new_connect_list, num_connect_array):
            # Determine series/parallel configuration
            if series_or_parallel:
                series_parallel = 1 if series_or_parallel == 'series' else 0
            else:
                series_parallel = np.random.choice([0, 1])
            
            cluster_1, cluster_2 = connect
            this_connect = connecting_nodes[count:count+num]
            
            if series_parallel == 1 or series_or_parallel == 'series':  # Series
                cluster_1_node = np.random.choice([n for n, v in G.nodes(data=True) if v.get('Cluster') == cluster_1])
                cluster_2_node = np.random.choice([n for n, v in G.nodes(data=True) if v.get('Cluster') == cluster_2])
                
                if num == 1:
                    # Single connecting node connects both clusters
                    G.add_edges_from([(cluster_1_node, this_connect[0]), (cluster_2_node, this_connect[0])])
                else:
                    # Chain of connecting nodes between the two clusters
                    G.add_edge(cluster_1_node, this_connect[0])
                    G.add_edge(cluster_2_node, this_connect[-1])
                    for j in range(num - 1):
                        G.add_edge(this_connect[j], this_connect[j + 1])
            
            else:  # Parallel or random
                same = np.random.choice([0, 1])
                
                if same == 1 or series_or_parallel == 'parallel':  # Same connections (parallel)
                    cluster_1_node = np.random.choice([n for n, v in G.nodes(data=True) if v.get('Cluster') == cluster_1])
                    cluster_2_node = np.random.choice([n for n, v in G.nodes(data=True) if v.get('Cluster') == cluster_2])
                    result = list(itertools.product([cluster_1_node, cluster_2_node], this_connect))
                    G.add_edges_from(result)
                
                else:  # Different random connections
                    for j in range(num):
                        cluster_1_node = np.random.choice([n for n, v in G.nodes(data=True) if v.get('Cluster') == cluster_1])
                        cluster_2_node = np.random.choice([n for n, v in G.nodes(data=True) if v.get('Cluster') == cluster_2])
                        G.add_edges_from([(cluster_1_node, this_connect[j]), (cluster_2_node, this_connect[j])])
            
            count += num

    if len(list(nx.connected_components(G))) > 1:     # Keep largest component
        comps = list(nx.connected_components(G))
        largest_comp = max(comps,key=len)
        G.remove_nodes_from([node for node in G if node not in largest_comp])     # Remove isolated nodes
    
    if len([node for node, degree in G.degree() if degree < 2]) > 0:     # Remove nodes with k < 2
        remove_nodes = [node for node, degree in G.degree() if degree < 2]
        G.remove_nodes_from(remove_nodes)

    return G
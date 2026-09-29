import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
import scipy.sparse as sparse


def network_memristive_state(t, G, v_in, R_on, R_off, tau, k, dt, nt, window_function,
                            source_nodes=None, drain_nodes=None):
    """
    Simulates the algebraic network over time.

    Parameters:
    - t: Simulation time
    - G: The graph representing the network.
    - v_in: Input voltage function.
    - R_on: On resistance.
    - R_off: Off resistance.
    - tau: Time constant.
    - k: Scaling factor.
    - dt: Time step size.
    - nt: Number of time steps.
    - window_function: Window function
    - source_nodes: Nodes to be used as sources
    - drain_nodes: Nodes to be used as drains

    Returns:
    - store_x: Array storing the state of the memristors over time.
    """
    node_list = np.asarray(G)
    edge_list = np.asarray(G.edges())
    
    num_sources = len(source_nodes) if source_nodes is not None else 2
    num_drains = len(drain_nodes) if drain_nodes is not None else 2

    if source_nodes is None:
        source_nodes = [node_list[random.randint(0, len(node_list)-1)] for _ in range(num_sources)]

    if drain_nodes is None:
        drain_nodes = [node_list[random.randint(0, len(node_list)-1)] for _ in range(num_drains)]

    nodes_2 = [node for node in node_list if node not in source_nodes and node not in drain_nodes]
    nodes_part = np.concatenate([source_nodes, drain_nodes, nodes_2])

    B_part = nx.incidence_matrix(G,
                                 nodelist=nodes_part,
                                 edgelist=edge_list,
                                 oriented=True
                                ).tocsr()

    x = np.zeros((len(edge_list), 1)) + 5e-2

    store_x = np.zeros((len(edge_list), nt))
    store_x[:, 0] = x.flatten()

    part = num_sources + num_drains


    for i in range(1, nt):
        # Update conductance matrix based on memristive state
        Gx = 1/(x.flatten()*(R_on - R_off) + R_off + 1e-30)

        # Define Partitioned Laplacian matrix for example circuit
        L = (B_part @ (Gx[:, None] * B_part.T)).tocsr() 

        L21 = L[part:, :part]
        L22 = L[part:, part:]

        # Defining V1 at time i
        v_i = np.array([v_in(t[i])]*num_sources + [0]*num_drains)

        # Internal node voltages can be solved using the partitioned Laplacian
        v2 = sparse.linalg.spsolve(L22, -(L21 @ v_i))

        # New V vector is the combination of input voltage and internal node voltages above
        v_combined = np.concatenate([v_i, v2]).reshape(-1, 1)

        f = window_function(x)     # J. Window function
        v_edges = B_part.T @ v_combined
        dx = f * k * R_on * Gx[:, None] * v_edges  - x/tau 

        x = np.clip(x + dx*dt, 0, 1)

        store_x[:, i] = x.flatten()

    return store_x
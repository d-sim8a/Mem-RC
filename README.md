# Memristive Reservoir Computing

In this repository, you can find the results of my M.Sc. Thesis, co-supervised by Dr. Claudia Gomes da Rocha & Dr. Wilten Nicola. The thesis can be split into 3 different topics: Non-Coupled Nonlinear  Memristive System, "Random" Nanowire Networks with Inherent Capactive Effects, and Memristive-Based Integrate and Fire Neurons. Training of networks consisted of using the memristive currents as a basis set, and used Tikhonov's Regularization scheme<sup>[1](#neural_net_txtbook)</sup>

## Non-Coupled Nonlinear Memristive System

This topic introduced some early network training considerations. I first considered a simple network, that consists of memristive units, that are not connected with each other in any fashion such that they do not "interact". Each memristive unit was governed by the following relations<sup>[2](#ref1)</sup>:

$$
I = \alpha (1-x)\left( 1-e^{-\beta V} \right) + \gamma x \sinh{\delta V}
$$

where the first term corresponds to the Schottky current, and the second term corresponds to the tunneling current. This form is easily implemented for a non-coupled approach because of the non-Ohmic nature of the device. There was also the addition of a rectifying current, which was introduced in a paper with contributions from my supervisor <sup>[3](#ref2)</sup>:

$$
I = \alpha_{1} (1-x)\left( 1-e^{-\beta_{1} V} \right) + \gamma x \sinh{\delta V} + \alpha_{2}\left( 1-e^{-\beta_{2} V} \right)
$$
## Nanowire Networks with Inherent Capactive Effects

Nanowire networks are interesting due to the fact that the connections, or junctions, between the conductive wires are memristive devices. With this in mind, one may use Modified Nodal Analysis (MNA) <sup>[4](#mna_paper),</sup><sup>[5](#mna_site)</sup> as I did here, because the network can be translated into a graph representation; where wires are the nodes, and edges are the memristors. Using a forward Euler integration scheme, the network's dynamics can be simulated; each time step needs the static solution of MNA to solve for the voltage difference across each memristor. Hypothetically, any kind of memristor model can be used here, but for simplicity the Decay HP model <sup>[6](#decay_model)</sup> is used, which is described by:

$$
\frac{dx}{dt} = \frac{\mu_{\nu}R_{on}}{D^{2}}i(t) - \frac{x}{\tau} =  kR_{on}\frac{v(t)}{R_{M}(x)} - \frac{x}{\tau}
$$

where the model closely resembles the classic HP model <sup>[7](#hp_model)</sup> but with an additional term to allow for a relaxation to the thermodynamically favourable high resistive state. 


In a recent paper <sup>[8](#inherent_cap)</sup>, the authors introduce the idea of an inherent capacitive effect for 2-terminal resistive switching devices. They introduce their proposed equivalent circuit, which now has a capacitor in parallel with the off-resistor, which is in series with the on-resistor. This means that the current is now represented by:

$$
I = G_{on}(x)(V_{in}-V_{i}) = G_{off}(x)(V_{i}-V_{out}) + \frac{d}{dt}\left( C(x)\left( V_{i}-V_{out}\right) \right)
$$

where $G_{on}(x)$, $G_{off}(x)$, and $C(x)$ are the state-dependent on-conductance, off-conductance, and capacitances respectively. There is also the introduction of an internal node with a voltage of $V_{i}$. With this new internal node, brings more problems; an additional element with each memristive edge. This calls for a simplification of the model, so that we may bypass this internal node and solve for the nodal voltages at each time-step. The solution can be worked out, but the final expression for the current at discrete time-step $n+1$ is:

$$
I^{n+1} = G_{eff}^{n+1}(V_{in}^{n+1}-V_{out}^{n+1}) - I_{C}^{n}
$$

where
$$
G^{n+1}_{eff}=\frac{G^{n+1}_{on}\left(G^{n+1}_{off}+\frac{C^{n+1}}{\Delta t}\right)}{G^{n+1}_{on}+G^{n+1}_{off}+\frac{C^{n+1}}{\Delta t}}
$$

$$
I^{n}_{C}=\frac{G^{n+1}_{on}\frac{Q^{n}_{C}}{\Delta t}}{G^{n+1}_{on}+G^{n+1}_{off}+\frac{C^{n+1}}{\Delta t}}
$$


This form of equations is much easier to implement into an MNA pipeline than the previous one.

## Memristive-Based Integrate and Fire Neurons

Neuronal dynamics presents itself in the form of an \textit{action potential}. This is the voltage difference of the inside and outside of the cell, which itself happens because of ion movement through membrane-embedded \textit{ion gates}. The simplest circuit of this action potential includes a resistor and capacitor in parallel, with a voltage threshold switch to force the voltage to return to some \textit{reset} when it reaches some defined \textit{threshold} value.

$$
C\frac{dV}{dt} = I - \frac{V}{R}
V^{-}=V_{thresh} \rightarrow V^{+}=V_{reset}
$$

This circuit can exhibit firing, with defined and solvable firing rates, which depends on the input current and the values imposed by the resistor, capacitor, and threshold switch. 

Suppose we switch the resistor with a memristor, we would now have the following equation for our circuit

$$
C\frac{dV}{dt} = I - \frac{V}{R_{M}(x)}
\frac{dx}{dt} = f(V,x)
V^{-}=V_{thresh} \rightarrow V^{+}=V_{reset}
$$

where we can choose some accepted state equation for our memristor state. 

## References

<a id="neural_net_txtbook"></a>[1] Haykin, S. Neural networks and learning machines, 3/E (Pearson Education, 2009).

<a id="ref1"></a>[2] Chang, T. et al. Synaptic behaviors and modeling of a metal oxide memristive device. Applied Physics A 102, 857–863 (2011).

<a id="ref2"></a>[3] Alialy, S., Esteki, K., Ferreira, M. S., Boland, J. J. & da Rocha, C. G. Nonlinear ion drift-diffusion memristance description of tio 2 rram devices. Nanoscale Advances 2, 2514–2524 (2020).

<a id="mna_paper"></a>[4] Ho, C.-W., Ruehli, A. & Brennan, P. The modified nodal approach to network analysis. IEEE Transactions on Circuits and Systems 22, 504–509 (1975).

<a id="mna_site"></a>[5] Cheever, E. Analysis of circuits. https://lpsa.swarthmore.edu/Systems/Electrical/mna/MNA1.html (2022).

<a id="decay_model"></a>[6] Sillin, H. O. et al. A theoretical and experimental study of neuromorphic atomic switch networks for reservoir computing. Nanotechnology 24, 384004 (2013). URL https://dx.doi.org/10.1088/0957-4484/24/38/384004.

<a id="hp_model"></a>[7] Strukov, D. B., Snider, G. S., Stewart, D. R. & Williams, R. S. The missing memristor found. Nature 453, 80–83 (2008).

<a id="inherent_cap"></a>[8] eetendra Singh, Sanjeev Kumar Sharma, and Balwinder Raj. Investigation of inherent capacitive effects in linear memristor model. Silicon, 13(10):3423–3430, 2021.

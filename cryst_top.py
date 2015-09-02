#!/usr/bin/env sage-python
from Net import SystreDB, Net
from Visualizer import GraphPlot 
import numpy as np

np.set_printoptions(threshold=np.nan, precision=4, suppress=True, linewidth=185)

def qtz():
    # qtz net
    qtz = Net()

    qtz.graph = {'A':{'B':['e1','e6']}, 'B':{'C':['e5', 'e2']}, 'C':{'A':['e4', 'e3']}}
    qtz.voltage = np.matrix([[0, 1, 0],
                             [-1, -1, 1],
                             [1, 0, 0],
                             [0, 0, 0],
                             [0, 0, 1],
                             [0, 0, 0]])

    #qtz.graph.show(edge_labels=True)
    qtz.get_lattice_basis()
    qtz.get_cocycle_basis()
    qtz.get_cycle_basis()
    #qtz.lattice_basis = np.matrix([[0., 0., 1., -1., 0., 0.],
    #                               [1., 0., 0., 0., 0., -1.],
    #                               [0., 0., 0., 1., 1., 1.]])

    #qtz.cycle = np.matrix([[1., 0., 0., 0., 0., -1.],
    #                       [0., 1., 0., 0., -1., 0.],
    #                       [0., 0., 1., -1., 0., 0.],
    #                       [0., 0., 0., 1., 1., 1.]])

    #qtz.cocycle = np.matrix([[1., 0., -1., -1., 0., 1.],
    #                         [-1., 1., 0., 0., 1., -1.]])

    #qtz.cycle_rep = np.matrix([[0., 1., 0.],
    #                           [-1., -1., 0.],
    #                           [1., 0., 0.],
    #                           [0., 0., 1.]])

    qtz.barycentric_embedding()
    gp = GraphPlot(qtz)
    gp.view_placement(init=np.array([0., 0., 0.])) 

    #print qtz.lattice_arcs*qtz.metric_tensor*qtz.lattice_arcs.T

def bor():
    bor=Net()
    bor.graph = {'A':{'C':['e1'], 'B':['e2'], 'F':['e3']}, 
                 'B':{}, 
                 'C':{}, 
                 'D':{'C':['e7'], 'B':['e9'], 'F':['e11']}, 
                 'E':{'C':['e4'], 'B':['e5'], 'F':['e6']}, 
                 'F':{}, 
                 'G':{'B':['e10'], 'F':['e12'], 'C':['e8']}}
    bor.voltage = np.matrix([[0, 1, 1],
                             [1, 0, 1],
                             [1, 1, 1],
                             [0, 1, 1],
                             [0, 0, 1],
                             [0, 1, 0],
                             [0, 0, 0],
                             [0, 0, 1],
                             [0, 0, 0],
                             [1, 0, 1],
                             [0, 0, 0],
                             [1, 0, 0]])
    bor.get_cocycle_basis()
    # NB: lattice basis first because they are included in the cycle basis
    # if not included - then the projection does not work!
    bor.get_lattice_basis()
    bor.get_cycle_basis()
    bor.barycentric_embedding()
    bor_show = GraphPlot(bor)
    bor_show.view_placement(init=np.array([0.25, 0.25, 0.25]))
    #bor.graph.show(edge_labels=True)
    #raw_input("Type any key to continue...\n")

def alpha_crystobalite():
    acryst = Net()
    acryst.graph = {'A':{'B':['e1','e2'], 'D':['e5','e6']}, 
                    'B':{'C':['e7', 'e8']}, 
                    'C':{'D':['e3', 'e4']},
                    'D':{}}
    #acryst.graph.show(edge_labels=True)
    #raw_input("Press any key...\n")
    acryst.voltage = np.matrix([[0, -1, 0],
                                [0, 0, 0],
                                [0, 0, 0],
                                [0, 1, 0],
                                [0, 0, -1],
                                [-1, 0, -1],
                                [0, 0, 0],
                                [-1, 0, 0]])

    #acryst.graph.show(edge_labels=True)
    acryst.get_cocycle_basis()
    acryst.get_lattice_basis()
    acryst.get_cycle_basis()
    acryst.barycentric_embedding()
    acr_show = GraphPlot(acryst)
    acr_show.view_placement(init=np.array([0.25, 0.25, 0.]))


def hcb():
    hcb = Net()
    # hcb 2d net
    hcb.lattice_basis = np.matrix([[1., -1., 0.],
                                  [0., 1., -1.]])
    hcb_view = GraphPlot(params=hcb.get_2d_params(), two_dimensional=True)
    hcb_view.plot_2d_cell()
    hcb_view.view_placement()

def test(net, volt):
    testnet = Net()
    testnet.graph = net
    testnet.voltage = volt

    testnet.get_lattice_basis()
    testnet.get_cocycle_basis()
    testnet.get_cycle_basis()

    testnet.barycentric_embedding()
    
    #print testnet.lattice_arcs*testnet.metric_tensor*testnet.lattice_arcs.T

    show = GraphPlot(testnet)
    show.view_placement(init=np.array([0.5, 0.5, 0.5]))

def main():
    systre = SystreDB()
    test(systre['tbo'], systre.voltages['tbo'])
    #hcb()
    #qtz()
    #alpha_crystobalite()
    #bor()

if __name__=="__main__":
    main()


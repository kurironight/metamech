import numpy as np
import gym
from tools.graph import convert_edge_indices_to_adj, convert_adj_to_edge_indices
from tools.lattice_preprocess import make_main_node_edge_info
import networkx as nx
from FEM.make_structure import make_bar_structure
import matplotlib.pyplot as plt
from .gym_metamech import MetamechGym

MAX_NODE = 100
PIXEL = 100


class FEMGym(MetamechGym):
    # 定数定義

    # 初期化
    def __init__(self, node_pos, edges_indices, edges_thickness):
        super(FEMGym, self).__init__(node_pos, 0, 0,
                                     0, 0, 0, edges_indices, edges_thickness)

        self.pixel = PIXEL

    def extract_rho_for_fem(self):
        nodes_pos, edges_indices, edges_thickness = self.extract_node_edge_info()

        edges = [[self.pixel*nodes_pos[edges_indice[0]], self.pixel*nodes_pos[edges_indice[1]],
                  edge_thickness]
                 for edges_indice, edge_thickness in zip(edges_indices, edges_thickness)]

        rho = make_bar_structure(self.pixel, self.pixel, edges)

        return rho

    def calculate_simulation(self):
        rho = self.extract_rho_for_fem()

        return 0

    # 環境の描画
    def render(self, save_path="image.png"):
        rho = self.extract_rho_for_fem()

        ny, nx = rho.shape
        x = np.arange(0, nx+1)  # x軸の描画範囲の生成。
        y = np.arange(0, ny+1)  # y軸の描画範囲の生成。
        X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        _ = plt.pcolormesh(X, Y, rho, cmap="binary")
        plt.axis("off")
        fig.savefig(save_path)
        plt.close()

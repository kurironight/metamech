import numpy as np
import gym
from tools.graph import convert_edge_indices_to_adj, convert_adj_to_edge_indices
from tools.lattice_preprocess import make_main_node_edge_info
import networkx as nx
from FEM.make_structure import make_bar_structure
from FEM.fem import FEM, FEM_displacement
import matplotlib.pyplot as plt
from .gym_metamech import MetamechGym
import cv2

MAX_NODE = 100
PIXEL = 50
MAX_EDGE_THICKNESS = 5


class FEMGym(MetamechGym):
    # 定数定義

    # 初期化
    def __init__(self, node_pos, edges_indices, edges_thickness):
        super(FEMGym, self).__init__(node_pos, 0, 0,
                                     0, 0, 0, edges_indices, edges_thickness)

        self.pixel = PIXEL
        self.max_edge_thickness = MAX_EDGE_THICKNESS

        # condition for calculation
        ny = self.pixel
        nx = self.pixel
        Y_DOF = np.arange(2*(ny+1), 2*(nx+1)*(ny+1)+1, 2*(ny+1))
        X_DOF = np.arange(2*(ny+1)-1, 2*(nx+1)*(ny+1), 2*(ny+1))
        self.FIXDOF = np.concatenate([X_DOF, Y_DOF])
        self.displace_DOF = 2*nx*(ny+1)+2  # 強制変位を起こす場所

        F = np.zeros(2 * (nx + 1) * (ny + 1), dtype=np.float64)
        F[self.displace_DOF-1] = -1
        self.F = F

        disp = np.zeros((2 * (nx + 1) * (ny + 1)), dtype=np.float64)
        disp[self.displace_DOF-1] = -1
        self.disp = disp

        # 構造が繋がっているかを確認する時，確認するメッシュ位置のindex
        self.check_output_mesh_index = (0, 0)
        self.check_input_mesh_index = (0, nx-1)
        self.check_freeze_mesh_index = (ny-1, int(nx/2))

    def extract_rho_for_fem(self):
        nodes_pos, edges_indices, edges_thickness = self.extract_node_edge_info()

        edges = [[self.pixel*nodes_pos[edges_indice[0]], self.pixel*nodes_pos[edges_indice[1]],
                  self.max_edge_thickness*edge_thickness]
                 for edges_indice, edge_thickness in zip(edges_indices, edges_thickness)]

        rho = make_bar_structure(self.pixel, self.pixel, edges)

        return rho

    def calculate_simulation(self):
        rho = self.extract_rho_for_fem()
        #U = FEM(rho, self.FIXDOF, self.F)

        #displacement = np.array([U[0], U[1]])
        #output_vectors = np.array([1, 0])
        #efficiency = np.dot(output_vectors, displacement)
        #
        #print("力：\n", efficiency)
        U = FEM_displacement(rho, self.FIXDOF, np.zeros(
            (2 * (self.pixel + 1) * (self.pixel + 1)), dtype=np.float64), self.disp)

        # actuator.pyより引用
        displacement = np.array([U[0], U[1]])
        output_vectors = np.array([1, 0])
        efficiency = np.dot(output_vectors, displacement)

        print("変位版：\n", efficiency)

        return efficiency

    def confirm_graph_is_connected(self):
        # グラフが全て接続しているか確認

        rho = self.extract_rho_for_fem().astype(np.uint8)

        _, markers = cv2.connectedComponents(rho)
        if markers[self.check_output_mesh_index] == markers[self.check_input_mesh_index] & \
                markers[self.check_output_mesh_index] == markers[self.check_freeze_mesh_index]:

            self.render("image_connected.png")
            return True
        else:
            return False

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

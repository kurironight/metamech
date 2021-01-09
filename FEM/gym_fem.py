import numpy as np
import gym
from tools.graph import convert_edge_indices_to_adj, convert_adj_to_edge_indices
from tools.lattice_preprocess import make_main_node_edge_info
import networkx as nx
from FEM.make_structure import make_bar_structure
import matplotlib.pyplot as plt

MAX_NODE = 100
LINEAR_STIFFNESS = 10
ANGULAR_STIFFNESS = 0.2
PIXEL = 100


class MetamechGym(gym.Env):
    # 定数定義

    # 初期化
    def __init__(self, node_pos, input_nodes, input_vectors, output_nodes, output_vectors, frozen_nodes, edges_indices, edges_thickness):
        super(MetamechGym, self).__init__()

        # 初期条件の指定
        self.max_node = MAX_NODE  # ノードの最大数
        self.pixel = PIXEL

        self.first_node_pos = node_pos
        self.input_nodes = input_nodes
        self.input_vectors = input_vectors
        self.output_nodes = output_nodes
        self.output_vectors = output_vectors
        self.frozen_nodes = frozen_nodes
        self.first_edges_indices = edges_indices
        self.first_edges_thickness = edges_thickness

        # current_status
        self.current_obs = {}

        # 行動空間と状態空間の定義
        self.action_space = gym.spaces.Dict({
            'new_node':  gym.spaces.Box(low=0, high=1.0, shape=(1, 2), dtype=np.float32),
            'edge_thickness': gym.spaces.Box(low=np.array([-1]), high=np.array([1.0]), dtype=np.float32),
            'which_node': gym.spaces.MultiDiscrete([self.max_node-1, self.max_node]),
            'end': gym.spaces.Discrete(2),
        })

        self.observation_space = gym.spaces.Dict({
            # -1のところは，意味のないノードの情報
            'nodes':  gym.spaces.Box(low=-1, high=1.0, shape=(self.max_node, 2), dtype=np.float32),
            'edges': gym.spaces.Dict({
                'adj': gym.spaces.MultiBinary([self.max_node, self.max_node]),
                # -1のところは，意味の無いエッジの情報
                'thickness': gym.spaces.Box(low=-1, high=1.0, shape=(self.max_node*self.max_node, 1), dtype=np.float32)
            })
        })

    # 環境のリセット

    def reset(self):
        self._renew_current_obs(
            self.first_node_pos, self.first_edges_indices, self.first_edges_thickness)

        return self.current_obs

    def random_action(self):
        """強化学習を用いない場合に確認するための方針

        Returns:
            action
        """
        action = self.action_space.sample()

        # padding部分を排除した情報を抽出
        nodes_pos, adj, edges_thickness = self._extract_non_padding_status_from_current_obs()
        node_num = nodes_pos.shape[0]

        action['which_node'][0] = np.random.choice(np.arange(node_num))
        action['which_node'][1] = np.random.choice(
            np.delete(np.arange(node_num+1), action['which_node'][0]))

        return action

    def step(self, action):

        if action['end']:  # 終了条件を満たす場合
            # TODO 本来はこれは，外側の方で行うこと
            reward = 1
            obs = self.current_obs
            return obs, reward, True, {}

        # padding部分を排除した情報を抽出
        nodes_pos, edges_indices, edges_thickness = self.extract_info_for_lattice()
        node_num = nodes_pos.shape[0]

        if action['which_node'][1] == node_num:  # 新規ノードを追加する場合
            nodes_pos = np.concatenate([nodes_pos, action['new_node']])

        # TODO ここは本当はassert
        if action['which_node'][1] > node_num:
            print("actionで選ばれたノードが大きい")
            action['which_node'][1] = node_num
        if action['which_node'][0] > node_num:
            action['which_node'][0] = node_num-1

        edges_indices = np.concatenate([edges_indices, np.array(
            [[action['which_node'][0], action['which_node'][1]]])])

        edges_thickness = np.concatenate(
            [edges_thickness, action['edge_thickness']])

        self._renew_current_obs(nodes_pos, edges_indices, edges_thickness)

        reward = 0

        return self.current_obs, reward, False, {}

    def confirm_graph_is_connected(self):
        # グラフが全て接続しているか確認

        nodes_pos, adj, edges_thickness = self._extract_non_padding_status_from_current_obs()
        edges_indices = convert_adj_to_edge_indices(adj)

        G = nx.Graph()
        G.add_nodes_from(np.arange(len(nodes_pos)))
        G.add_edges_from(edges_indices)

        return nx.is_connected(G)

    def _extract_non_padding_status_from_current_obs(self):
        """self.current_obsのうち，PADDINGを除いた部分を抽出
        """
        nodes_mask = self.current_obs['nodes'][:, 0] != -1  # 意味を成さない部分を除外
        vaild_nodes = self.current_obs['nodes'][nodes_mask]

        # 意味を成さない部分を除外
        thickness_mask = self.current_obs['edges']['thickness'] != -1
        vaild_edges_thickness = self.current_obs['edges']['thickness'][thickness_mask]

        node_num = vaild_nodes.shape[0]
        valid_adj = self.current_obs['edges']['adj'][:node_num, :node_num]

        return vaild_nodes, valid_adj, vaild_edges_thickness

    def extract_info_for_lattice(self):
        nodes_pos, adj, edges_thickness = self._extract_non_padding_status_from_current_obs()
        edges_indices = convert_adj_to_edge_indices(adj)

        return nodes_pos, edges_indices, edges_thickness

    def extract_rho_for_fem(self):
        nodes_pos, edges_indices, edges_thickness = self.extract_info_for_lattice()

        edges = [[self.pixel*nodes_pos[edges_indice[0]], self.pixel*nodes_pos[edges_indice[1]],
                  5*edge_thickness]
                 for edges_indice, edge_thickness in zip(edges_indices, edges_thickness)]

        rho = make_bar_structure(self.pixel, self.pixel, edges)

        return rho

    def calculate_displacement(self):
        print("calculate_displacement_start")
        rho = self.extract_rho_for_fem()

        ny, nx = rho.shape
        x = np.arange(0, nx+1)  # x軸の描画範囲の生成。
        y = np.arange(0, ny+1)  # y軸の描画範囲の生成。
        X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        _ = plt.pcolormesh(X, Y, rho, cmap="binary")
        plt.axis("off")
        fig.savefig("image_.png")
        plt.close()

        # np.save("input_nodes", self.input_nodes)
        # np.save("input_vectors", self.input_vectors)
        # np.save("output_nodes", self.output_nodes)
        # np.save("output_vectors", self.output_vectors)
        # np.save("frozen_nodes", self.frozen_nodes)

        print("calculate_efficiency_end")

        return 0

    def _renew_current_obs(self, node_pos, edges_indices, edges_thickness):
        self.current_obs['nodes'] = np.pad(
            node_pos, ((0, self.max_node-node_pos.shape[0]), (0, 0)), constant_values=-1)
        adj = convert_edge_indices_to_adj(
            edges_indices, size=self.max_node)
        self.current_obs['edges'] = {
            'adj': adj,
            'thickness': np.pad(
                edges_thickness, (0, self.max_node*self.max_node-edges_thickness.shape[0]), constant_values=-1)}

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

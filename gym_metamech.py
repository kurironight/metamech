import numpy as np
import gym
from tools.graph import convert_edge_indices_to_adj, convert_adj_to_edge_indices
from metamech.lattice import Lattice
from metamech.actuator import Actuator

# 初期のノードの状態を抽出
new_node_pos = np.array([[6.5,    6.9282],
                         [7.,    7.79423],
                         [7.5,    6.9282],
                         [8.,    7.79423],
                         [0.,    7.79423],
                         [0.5,    6.9282],
                         [1.,    7.79423],
                         [1.5,    6.9282]])

new_input_nodes = np.array([0, 1, 2, 3])
new_input_vectors = np.array([
    [0., -0.1],
    [0., -0.1],
    [0., -0.1],
    [0., -0.1]
])
new_output_nodes = np.array([4, 5, 6, 7])
new_output_vectors = np.array([
    [-1, 0],
    [-1, 0],
    [-1, 0],
    [-1, 0],
])

edges_indices = np.array([
    [0,  1], [0,  2], [0, 18], [1,  2], [1,  3], [2, 18],
    [2,  3], [2,  4], [2, 20], [3,  5], [3,  4], [4,  5],
    [4, 22], [4, 20], [4,  6], [5,  7], [5,  6], [6, 22],
    [6,  7], [6, 24], [6,  8], [7,  9], [7,  8], [8, 10],
    [8, 24], [8,  9], [8, 26], [9, 10], [9, 11], [10, 26],
    [10, 11], [10, 12], [10, 28], [11, 12], [11, 13], [12, 28],
    [12, 13], [12, 14], [12, 30], [13, 15], [13, 14], [14, 15],
    [14, 32], [14, 30], [14, 16], [15, 16], [16, 32], [17, 19],
    [17, 18], [17, 35], [18, 19], [18, 20], [19, 35], [19, 21],
    [19, 20], [19, 37], [20, 22], [20, 21], [21, 22], [21, 39],
    [21, 37], [21, 23], [22, 24], [22, 23], [23, 24], [23, 39],
    [23, 41], [23, 25], [24, 26], [24, 25], [25, 27], [25, 26],
    [25, 41], [25, 43], [26, 27], [26, 28], [27, 43], [27, 29],
    [27, 28], [27, 45], [28, 29], [28, 30], [29, 45], [29, 31],
    [29, 30], [29, 47], [30, 32], [30, 31], [31, 32], [31, 49],
    [31, 47], [31, 33], [32, 33], [33, 49], [34, 35], [34, 36],
    [34, 52], [35, 36], [35, 37], [36, 52], [36, 37], [36, 38],
    [36, 54], [37, 39], [37, 38], [38, 39], [38, 56], [38, 54],
    [38, 40], [39, 41], [39, 40], [40, 56], [40, 41], [40, 58],
    [40, 42], [41, 43], [41, 42], [42, 44], [42, 58], [42, 43],
    [42, 60], [43, 44], [43, 45], [44, 60], [44, 45], [44, 46],
    [44, 62], [45, 46], [45, 47], [46, 62], [46, 47], [46, 48],
    [46, 64], [47, 49], [47, 48], [48, 49], [48, 66], [48, 64],
    [48, 50], [49, 50], [50, 66], [51, 53], [51, 52], [51, 69],
    [52, 53], [52, 54], [53, 69], [53, 55], [53, 54], [53, 71],
    [54, 56], [54, 55], [55, 56], [55, 73], [55, 71], [55, 57],
    [56, 58], [56, 57], [57, 58], [57, 73], [57, 75], [57, 59],
    [58, 60], [58, 59], [59, 61], [59, 60], [59, 75], [59, 77],
    [60, 61], [60, 62], [61, 77], [61, 63], [61, 62], [61, 79],
    [62, 63], [62, 64], [63, 79], [63, 65], [63, 64], [63, 81],
    [64, 66], [64, 65], [65, 66], [65, 83], [65, 81], [65, 67],
    [66, 67], [67, 83], [68, 69], [68, 70], [69, 70], [69, 71],
    [70, 71], [70, 72], [71, 73], [71, 72], [72, 73], [72, 74],
    [73, 75], [73, 74], [74, 75], [74, 76], [75, 77], [75, 76],
    [76, 78], [76, 77], [77, 78], [77, 79], [78, 79], [78, 80],
    [79, 80], [79, 81], [80, 81], [80, 82], [81, 83], [81, 82],
    [82, 83], [82, 84], [83, 84],
])
vectors = np.array([81, 82, 83, 84, 68, 69, 70, 71])

# edges_indicesのうち，初期エッジ情報のみを抽出する
edges_bool = np.isin(edges_indices, vectors)
edges_bool = edges_bool[:, 0] & edges_bool[:, 1]
edges_indices = edges_indices[edges_bool]

new_edges_indices = np.zeros(shape=edges_indices.shape, dtype=int)
for i, v in enumerate(vectors):
    mask = edges_indices == v
    new_edges_indices[mask] = i

new_edges_thickness = np.ones(len(new_edges_indices))

MAX_NODE = 20


class MetamechGym(gym.Env):
    # 定数定義

    # 初期化
    def __init__(self, node_pos, input_nodes, input_vectors, output_nodes, output_vectors, edges_indices, edges_thickness):
        super(MetamechGym, self).__init__()
        # 初期条件の指定
        self.max_node = MAX_NODE  # ノードの最大数

        self.node_pos = node_pos
        self.input_nodes = input_nodes
        self.input_vectors = input_vectors
        self.output_nodes = output_nodes
        self.output_vectors = output_vectors
        self.edges_indices = edges_indices
        self.edges_thickness = edges_thickness
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

    def renew_current_obs(self, node_pos, edges_indices, edges_thickness):
        self.current_obs['nodes'] = np.pad(
            node_pos, ((0, self.max_node-node_pos.shape[0]), (0, 0)), constant_values=-1)
        adj = convert_edge_indices_to_adj(
            edges_indices, size=self.max_node)
        self.current_obs['edges'] = {
            'adj': adj,
            'thickness': np.pad(
                self.edges_thickness, (0, self.max_node*self.max_node-self.edges_thickness.shape[0]), constant_values=-1)}

    # 環境のリセット
    def reset(self):
        self.renew_current_obs(
            self.node_pos, self.edges_indices, self.edges_thickness)
        return self.current_obs

    def _remove_padding_from_current_obs(self):
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

    # 環境の１ステップ実行
    def step(self, action):

        if action['end']:  # 終了条件を満たす場合
            reward = 1
            obs = self.current_obs
            return obs, reward, True, {}

        # padding部分を排除した情報を抽出
        nodes_pos, adj, edges_thickness = self._remove_padding_from_current_obs()
        node_num = nodes_pos.shape[0]
        edges_indices = convert_adj_to_edge_indices(adj)

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

        self.renew_current_obs(nodes_pos, edges_indices, edges_thickness)

        return self.current_obs, 1, False, {}

    # 環境の描画
    def render(self, mode='console', close=False):
        # TODO あとでレンダー出来るようにする


env = MetamechGym(new_node_pos, new_input_nodes, new_input_vectors,
                  new_output_nodes, new_output_vectors, new_edges_indices, new_edges_thickness)

# １エピソードのループ
state = env.reset()

while True:
    # ランダム行動の取得
    action = env.action_space.sample()
    # １ステップの実行
    state, reward, done, info = env.step(action)
    print('reward:', reward)
    # エピソード完了
    if done:
        print('done')
        break

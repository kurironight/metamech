

import argparse
import numpy as np
from itertools import count
from collections import namedtuple
from policy import model
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from env.gym_fem import FEMGym
from tools.graph import make_T_matrix, make_edge_adj, make_D_matrix
import torch.distributions as tdist
from tools.lattice_preprocess import make_main_node_edge_info

# Cart Pole

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

# 初期のノードの状態を抽出
origin_nodes_positions = np.array([
    [0., 0.86603], [0.5, 0.], [1., 0.86603], [1.5, 0.],
    [2., 0.86603], [2.5, 0.], [3., 0.86603], [3.5, 0.],
    [4., 0.86603], [4.5, 0.], [5., 0.86603], [5.5, 0.],
    [6., 0.86603], [6.5, 0.], [7., 0.86603], [7.5, 0.],
    [8., 0.86603], [0., 2.59808], [0.5, 1.73205], [1., 2.59808],
    [1.5, 1.73205], [2., 2.59808], [2.5, 1.73205], [3., 2.59808],
    [3.5, 1.73205], [4., 2.59808], [4.5, 1.73205], [5., 2.59808],
    [5.5, 1.73205], [6., 2.59808], [6.5, 1.73205], [7., 2.59808],
    [7.5, 1.73205], [8., 2.59808], [0., 4.33013], [0.5, 3.4641],
    [1., 4.33013], [1.5, 3.4641], [2., 4.33013], [2.5, 3.4641],
    [3., 4.33013], [3.5, 3.4641], [4., 4.33013], [4.5, 3.4641],
    [5., 4.33013], [5.5, 3.4641], [6., 4.33013], [6.5, 3.4641],
    [7., 4.33013], [7.5, 3.4641], [8., 4.33013], [0., 6.06218],
    [0.5, 5.19615], [1., 6.06218], [1.5, 5.19615], [2., 6.06218],
    [2.5, 5.19615], [3., 6.06218], [3.5, 5.19615], [4., 6.06218],
    [4.5, 5.19615], [5., 6.06218], [5.5, 5.19615], [6., 6.06218],
    [6.5, 5.19615], [7., 6.06218], [7.5, 5.19615], [8., 6.06218],
    [0., 7.79423], [0.5, 6.9282], [1., 7.79423], [1.5, 6.9282],
    [2., 7.79423], [2.5, 6.9282], [3., 7.79423], [3.5, 6.9282],
    [4., 7.79423], [4.5, 6.9282], [5., 7.79423], [5.5, 6.9282],
    [6., 7.79423], [6.5, 6.9282], [7., 7.79423], [7.5, 6.9282],
    [8., 7.79423]])

origin_nodes_positions = origin_nodes_positions/8

origin_edges_indices = np.array([
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

# origin_edges_indices = np.concatenate(
#    [origin_edges_indices, [[81, 68], [68, 9]]])
origin_input_nodes = [81, 82, 83, 84]
origin_input_vectors = np.array([
    [0., -0.1],
    [0., -0.1],
    [0., -0.1],
    [0., -0.1]
])

origin_output_nodes = [68, 69, 70, 71]
origin_output_vectors = np.array([
    [-1, 0],
    [-1, 0],
    [-1, 0],
    [-1, 0],
])

origin_frozen_nodes = [1, 3, 5, 7, 9, 11, 13, 15]

# gymに入力する要素を抽出
new_node_pos, new_input_nodes, new_input_vectors, new_output_nodes, new_output_vectors, new_frozen_nodes, new_edges_indices, new_edges_thickness = make_main_node_edge_info(origin_nodes_positions, origin_edges_indices, origin_input_nodes, origin_input_vectors,
                                                                                                                                                                            origin_output_nodes, origin_output_vectors, origin_frozen_nodes)
env = FEMGym(new_node_pos,
             new_edges_indices, new_edges_thickness)

Saved_prob_Action = namedtuple('SavedAction', ['log_prob', 'value', 'action'])
Saved_mean_std_Action = namedtuple(
    'SavedAction', ['mean', 'variance', 'value', 'action'])


# １エピソードのループ
state = env.reset()

# ポリシーモデル定義
node_out_features = 3
GCN = model.GCN_fund_model(2, 1, node_out_features, 3).double()
X_Y = model.X_Y_model(node_out_features, 2).double()  # なぜかdoubleが必要だった
Stop = model.Stop_model(node_out_features, 2).double()
Select_node1 = model.Select_node1_model(node_out_features, 2).double()
Select_node2 = model.Select_node2_model(2*node_out_features, 2).double()
Edge_thickness = model.Edge_thickness_model(2*node_out_features, 2).double()


def select_action(state):
    nodes_pos, edges_indices, edges_thickness, node_adj = env.extract_node_edge_info()

    # GCNの為の引数を作成
    T = make_T_matrix(edges_indices)
    edge_adj = make_edge_adj(edges_indices, T)
    D_e = make_D_matrix(edge_adj)
    D_v = make_D_matrix(node_adj)

    # GCNへの変換
    node = torch.from_numpy(nodes_pos).clone().double()
    node = node.unsqueeze(0)
    edge = torch.from_numpy(edges_thickness).clone().double()
    edge = edge.unsqueeze(0).unsqueeze(2)
    node_adj = torch.from_numpy(node_adj).clone().double()
    node_adj = node_adj.unsqueeze(0)
    edge_adj = torch.from_numpy(edge_adj).clone().double()
    edge_adj = edge_adj.unsqueeze(0)
    D_v = torch.from_numpy(D_v).clone().double()
    D_v = D_v.unsqueeze(0)
    D_e = torch.from_numpy(D_e).clone().double()
    D_e = D_e.unsqueeze(0)
    T = torch.from_numpy(T).clone().double()
    T = T.unsqueeze(0)

    # action求め
    emb_graph, state_value = GCN(node, edge, node_adj, edge_adj, D_v, D_e, T)
    coord = X_Y(emb_graph)
    stop = Stop(emb_graph)
    # ノード1を求める
    node1_prob = Select_node1(emb_graph)
    node1_categ = Categorical(node1_prob)
    node1 = node1_categ.sample()
    H1 = emb_graph[0][node1]
    H1 = H1.repeat(emb_graph.shape[1], 1)
    H1 = H1.unsqueeze(0)
    # HとH1のノード情報をconcat
    emb_graph_cat = torch.cat([emb_graph, H1], 2)
    # ノード2を求める
    node2_prob = Select_node2(emb_graph_cat)
    node2_categ = Categorical(node2_prob)
    node2 = node2_categ.sample()
    H2 = emb_graph[0][node2]
    H2 = H2.repeat(emb_graph.shape[1], 1)
    H2 = H2.unsqueeze(0)
    # H1とH2のノード情報をconcat
    emb_graph_cat2 = torch.cat([H1, H2], 2)
    edge_thickness = Edge_thickness(emb_graph_cat2)

    # 正規分布よりactionを選択
    coord_x_tdist = tdist.Normal(coord[0][0].item(), coord[0][1].item())
    coord_y_tdist = tdist.Normal(coord[0][2].item(), coord[0][3].item())
    edge_thickness_tdist = tdist.Normal(
        edge_thickness[0][0].item(), edge_thickness[0][1].item())

    coord_x_action = coord_x_tdist.sample()
    coord_y_action = coord_y_tdist.sample()
    edge_thickness_action = edge_thickness_tdist.sample()

    # save to action buffer
    Stop.saved_actions.append(Saved_prob_Action(
        torch.log(stop[0]), state_value, stop[0].item()))
    Select_node1.saved_actions.append(Saved_prob_Action(
        torch.log(node1_categ.log_prob(node1)), state_value, node1.item()))
    Select_node2.saved_actions.append(Saved_prob_Action(
        torch.log(node2_categ.log_prob(node2)), state_value, node2.item()))

    X_Y.saved_actions.append(Saved_mean_std_Action(
        coord[0][:2], coord[0][2:], state_value, [coord_x_action, coord_y_action]))
    Edge_thickness.saved_actions.append(Saved_mean_std_Action(
        edge_thickness[0][0], edge_thickness[0][1], state_value, edge_thickness_action))

    action = {}
    action["which_node"] = np.array(
        [Select_node1.saved_actions[-1].action, Select_node2.saved_actions[-1].action])
    action["new_node"] = np.array(X_Y.saved_actions[-1].action)
    action["edge_thickness"] = np.array(
        Edge_thickness.saved_actions[-1].action)
    action["end"] = np.array(Stop.saved_actions[-1].action)

    return action


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss
    returns = []  # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    running_reward = 10

    # run inifinitely many episodes
    for i_episode in count(1):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 10000):

            # select action from policy
            action = select_action(state)

            # take the action
            state, reward, done, _ = env.step(action)

            if args.render:
                env.render()

            GCN.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()

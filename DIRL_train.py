import random
import warnings

import torch.optim as optim

from entity.config_all import CONFIG_DICT_ALL
from env import PipelineEnv
from models import *
from tools import *
from heuristics import get_HeuA_utilization
from refinement import perform_swap

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

LMT1 = 1800
LMT2 = 3600
LMT3 = 7200
LMT4 = 10800


# LMT1 = 10
# LMT2 = 20
# LMT3 = 30
# LMT4 = 40


def train(_instance: Instance, instance_name: str, threshold: float):
    # 从训练配置中获取参数
    episodes = _instance.training_config['episodes']
    epochs = _instance.training_config['epochs']
    lr = _instance.training_config['learning_rate']
    resume = False
    visualize = _instance.training_config['visualize']
    checkpoint_path = _instance.training_config['checkpoint_path']
    result_path = _instance.training_config['results_filename']

    gamma = _instance.training_config['gamma']
    gae_lambda = _instance.training_config['gae_lambda']
    entropy_coefficient = _instance.training_config['entropy_coefficient']
    clip_epsilon = _instance.training_config['clip_epsilon']
    save_model_path = _instance.training_config['save_model_path']
    utilization_log_path = _instance.training_config['utilization_log_path']
    save_placement_path = _instance.training_config['save_placement_path']
    all_record_path = _instance.training_config['all_record_path']
    nfp_cache_path = _instance.training_config['nfp_cache_path']

    nfp_cache = load_nfp_cache(nfp_cache_path)

    # 初始化环境和模型
    env = PipelineEnv(_instance)
    gnn_model = GNNModel(input_dim=2, hidden_dim=64, output_dim=32).to(device)
    piece_selector = PieceSelector(len(env.input_pieces)).to(device)
    angle_selector = AngleSelector(len(_instance.rotations)).to(device)

    critic = Critic().to(device)

    models = {
        'piece_selector': piece_selector,
        'angle_selector': angle_selector,
        'critic': critic,
        'gnn_model': gnn_model
    }

    optimizer_ac = optim.Adam(
        list(piece_selector.parameters()) +
        list(angle_selector.parameters()) +
        list(critic.parameters()) +
        list(gnn_model.parameters()),
        lr=lr
    )

    start_episode = 0
    # if resume and os.path.exists(checkpoint_path):
    #     start_episode = load_checkpoint(models, optimizer_ac, checkpoint_path)

    episode_utilization = []
    max_recorded_length = 10  # NOTE：lyp modified. this value should be small.
    all_episode_records = []
    best_placement = ({}, 0)
    Z = []

    if start_episode == 0:
        with open(utilization_log_path, 'a') as log_file:
            log_file.write("Episode,Utilization\n")

    pattern_tracker = TopBottomTracker(k=max_recorded_length)
    """
    记录的格式：
    {
        "placement": placement,  # 存储选择序列和旋转角度记录（不需要记录具体坐标位置）
        "utilization": utilization,  # 用于堆排序
    }
    """
    all_placements = {}
    total_episode_num = 0

    current_selection_pattern = {}  # pairwise
    current_angle_pattern = {}
    current_selection_tabu = {}
    current_angle_tabu = {}

    max_utilization = 0
    max_non_improve = 10  # NOTE: lyp 若max_non_improve次都没有提升，说明搜索方向不好
    non_improve_count = 0

    seed_interval = 100  # 每100轮换一次随机种子
    pattern_start = 30  # NOTE
    swap_interval = 50  # NOTE

    update_bottom_k = 200  # TODO：每200轮更新bottom_k，不让已有的劣质解存留太久

    train_start_time = time.time()
    last_elapsed_time = 0

    placement_scheme = 'BL'

    for episode in range(start_episode, episodes):
        # 计时器
        current_time = time.time()
        elapsed_time = current_time - train_start_time
        if last_elapsed_time < LMT1 <= elapsed_time:
            # 存30分钟结果
            top_k_selection_seq, top_k_angle_seq = get_sequence(pattern_tracker.top_k())
            with open(f"NFP_records_LS_2/30/top_sequences/{instance_name}.json", "w") as fl:
                # noinspection PyTypeChecker
                json.dump({"top_k_selection_seq": top_k_selection_seq, "top_k_angle_seq": top_k_angle_seq}, fl, indent=4)

            save_best_placement(best_placement, result_path[0])
            save_all_placement(all_placements, save_placement_path[0], instance_name)
            print(f"Training time limit (30min) reached. Stopping training for instance {instance_name}.")

        if last_elapsed_time < LMT2 <= elapsed_time:
            # 存1小时结果
            top_k_selection_seq, top_k_angle_seq = get_sequence(pattern_tracker.top_k())
            with open(f"NFP_records_LS_2/60/top_sequences/{instance_name}.json", "w") as fl:
                # noinspection PyTypeChecker
                json.dump({"top_k_selection_seq": top_k_selection_seq, "top_k_angle_seq": top_k_angle_seq}, fl, indent=4)

            save_best_placement(best_placement, result_path[1])
            save_all_placement(all_placements, save_placement_path[1], instance_name)
            print(f"Training time limit (1h) reached. Stopping training for instance {instance_name}.")

        if last_elapsed_time < LMT3 <= elapsed_time:
            # 存2小时结果
            top_k_selection_seq, top_k_angle_seq = get_sequence(pattern_tracker.top_k())
            with open(f"NFP_records_LS_2/120/top_sequences/{instance_name}.json", "w") as fl:
                # noinspection PyTypeChecker
                json.dump({"top_k_selection_seq": top_k_selection_seq, "top_k_angle_seq": top_k_angle_seq}, fl, indent=4)

            save_best_placement(best_placement, result_path[2])
            save_all_placement(all_placements, save_placement_path[2], instance_name)
            print(f"Training time limit (2h) reached. Stopping training for instance {instance_name}.")

        if last_elapsed_time < LMT4 <= elapsed_time:
            # 存4小时结果
            top_k_selection_seq, top_k_angle_seq = get_sequence(pattern_tracker.top_k())
            with open(f"NFP_records_LS_2/240/top_sequences/{instance_name}.json", "w") as fl:
                # noinspection PyTypeChecker
                json.dump({"top_k_selection_seq": top_k_selection_seq, "top_k_angle_seq": top_k_angle_seq}, fl, indent=4)

            save_best_placement(best_placement, result_path[3])
            save_all_placement(all_placements, save_placement_path[3], instance_name)

            print(f"Training time limit (4h) reached. Stopping training for instance {instance_name}.")
            total_episode_num = episode
            return total_episode_num

        last_elapsed_time = elapsed_time

        if episode > 0 and episode % seed_interval == 0:  # 换随机种子
            new_seed = random.randint(0, 2 ** 32 - 1)
            print(f"Change into new seed {new_seed} in episode {episode}")

        # 重置环境，记录初始状态
        env.reset()
        state_graph = combine_graphs(
            get_pieces_from_id(env.placed_pieces, env.located_pieces),
            boundary=Polygon([(0, 0), (_instance.bin_width, 0), (_instance.bin_width, _instance.bin_height),
                              (0, _instance.bin_height)])
        )
        no_placed_state_graph = combine_graphs(
            get_pieces_from_id(env.no_placed_pieces, env.located_pieces)
        )

        # NOTE：尝试在pattern start episode中使用Heu A作为初始模式
        if episode == pattern_start:
            # 面积递减排序
            sorted_indices = sorted(range(len(env.input_pieces)), key=lambda i: env.input_pieces[i].area, reverse=True)

            heu_utilization = get_HeuA_utilization(env.input_pieces, sorted_indices, instance_name, _instance.rotations,
                                                   _instance.bin_width)
            if max_utilization < heu_utilization:
                # 初始化模式
                for i in range(len(sorted_indices) - 1):
                    if i == 0:
                        current_selection_pattern["empty"] = [sorted_indices[i], episode, episode - 1]
                    current_selection_pattern[sorted_indices[i]] = [sorted_indices[i + 1], episode, episode - 1]
                # print(current_selection_pattern)

        # if (episode == 1 and len(_instance.rotations) == 1) or episode == pattern_start:
        #     current_selection_pattern = {}

        graph_data = nx_to_data_0(state_graph).to(device)
        no_placed_graph_data = nx_to_data_0(no_placed_state_graph).to(device)

        gnn_model.train()
        placed_state_encoding = gnn_model(graph_data).mean(dim=0)
        no_placed_state_encoding = gnn_model(no_placed_graph_data).mean(dim=0)

        # 组合成统一的状态表示
        selection_encoding = torch.cat([placed_state_encoding, no_placed_state_encoding]).to(device)
        state_encoding = placed_state_encoding

        # 两个模型分别记录 buffer，记录 (state, action, reward, next_state, done)
        buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': [],
        }

        piece_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': [],
        }

        angle_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': [],
        }

        step = 1
        done = False
        current_action_sequence = []
        episode_placement = {}
        pattern_tenure = 4  # NOTE: key params    lyp modified this (5)
        tabu_tenure = 8  # NOTE: key params    lyp modified this (10)
        # prob_random_prob = 0.1  # NOTE

        last_selection_action = -1

        print("non improve count", non_improve_count)
        if episode == pattern_start or (episode > pattern_start and non_improve_count > max_non_improve):  # Lyp
            # 当前模式不好，需要更新模式
            non_improve_count = 0
            if not pattern_tracker.is_empty():
                selection_sequences, angle_sequences = get_sequence(pattern_tracker.top_k())
                current_selection_pattern = get_selection_pattern(selection_sequences, episode, "selection")  # [(id1, id2)...]
                if len(_instance.rotations) != 1:
                    current_angle_pattern = get_angle_pattern(angle_sequences, episode)

                selection_sequences_tabu, angle_sequences_tabu = get_sequence(
                    pattern_tracker.bottom_k())  # if not use tabu, be worse
                this_current_selection_tabu = get_selection_pattern(selection_sequences_tabu, episode, "tabu")  # 当前轮次获取的tabu模式

                for key in this_current_selection_tabu:  # lyp added this code
                    if key not in current_selection_tabu:  # 不在禁忌表中
                        current_selection_tabu[key] = this_current_selection_tabu[key]  # 新增模式
                    else:
                        # EXP: 模式的存储格式为，后继集合+产生轮次+从产生开始到起作用的轮数
                        current_selection_tabu[key][0].update(this_current_selection_tabu[key][0])  # 新增模式
                        current_selection_tabu[key][1] = episode
                        current_selection_tabu[key][2] = episode

                # current_selection_tabu = get_selection_pattern(selection_sequences_tabu, episode,"tabu")
                # current_angle_tabu = get_angle_pattern(angle_sequences_tabu, episode)
                if len(_instance.rotations) != 1:
                    current_angle_tabu = get_angle_pattern(selection_sequences_tabu, episode)
                # for key in current_angle_tabu_2.keys():  # lyp added this code
                #     if key not in current_angle_tabu:
                #         current_angle_tabu[key] = current_angle_tabu_2[key]  # 新增模式

        else:  # 上一轮更新了tabu  lyp modified
            # else:  #lyp comment this
            # 不需重新求模式，但需将超出tenure的模式删除
            if current_selection_tabu:
                for key, p in list(current_selection_tabu.items()):
                    if episode - p[-1] > tabu_tenure + random.randint(0, tabu_tenure):  # this is wrong;lyp modified this
                        del current_selection_tabu[key]
            if current_angle_tabu is not None:
                for key, p in list(current_angle_tabu.items()):
                    if episode - p[-1] > tabu_tenure + random.randint(0, tabu_tenure):  # this is wrong;lyp modified this
                        del current_angle_tabu[key]
            if current_selection_pattern:
                for key, p in list(current_selection_pattern.items()):
                    if episode - p[-1] > pattern_tenure + random.randint(0, pattern_tenure):
                        if key in current_selection_tabu:
                            current_selection_tabu[key][0].add(current_selection_pattern[key][0])
                            current_selection_tabu[key][1] = episode
                            current_selection_tabu[key][2] = episode - 1
                        else:
                            current_selection_tabu[key] = [{current_selection_pattern[key][0]}, episode, episode - 1]
                        del current_selection_pattern[key]

            if current_angle_pattern and episode % pattern_tenure == 0:
                for key, p in list(current_angle_pattern.items()):
                    if episode - p[-1] > pattern_tenure + random.randint(0, pattern_tenure):
                        if key in current_angle_tabu:
                            current_angle_tabu[key][0] = current_angle_pattern[key][0]
                            current_angle_tabu[key][1] = episode
                            current_angle_tabu[key][2] = episode - 1
                        else:
                            current_angle_tabu[key] = [{current_angle_pattern[key][0]}, episode, episode - 1]
                        del current_angle_pattern[key]

        while not done:
            # 1️⃣ 选择零件（piece）
            mask = torch.ones(len(env.input_pieces), device=device, dtype=torch.int8)
            pattern_mask = torch.ones(len(env.input_pieces), device=device, dtype=torch.int8)
            for placed_piece in env.placed_pieces:
                mask[placed_piece] = 0

            piece_logit = piece_selector(selection_encoding)  # as mu
            # piece_logit = piece_selector(selection_encoding) # sigma                piece_logit~ N(mu,sigma)

            piece_logit = torch.where(torch.isnan(piece_logit), torch.full_like(piece_logit, 1e-4), piece_logit)
            piece_prob = torch.softmax(piece_logit, dim=-1)
            masked_piece_prob = piece_prob * mask

            # if random.random() < prob_random_prob:
            #     # NOTE: 随机选出一个没被mask的索引，重新赋值为0-1之间的随机数
            #     random_id = random.choice(list(torch.where(mask == 1)[0].cpu().numpy()))
            #     masked_piece_prob[random_id] = random.random()

            piece_prob = masked_piece_prob / masked_piece_prob.sum()
            piece_dist = torch.distributions.Categorical(piece_prob)
            sorted_piece_idx = list(torch.argsort(piece_dist.probs, descending=True))  # lyp added this code

            for key in current_selection_pattern:
                if type(key) == int and mask[key] == 1:
                    pattern_mask[current_selection_pattern[key][0]] = 0

            indices_s = []
            if current_selection_pattern:
                indices_s = list(current_selection_pattern.keys())
            if (current_selection_pattern is not None and last_selection_action in
                    indices_s):
                piece_id = current_selection_pattern[last_selection_action][0]

            elif (current_selection_pattern is not None and len(env.placed_pieces) == 0 and "empty" in  # lyp added this code
                  current_selection_pattern):
                piece_id = current_selection_pattern["empty"][0]

            else:
                piece_id = piece_dist.sample().item()

            if current_selection_tabu:
                indices_s = list(current_selection_tabu.keys())

            if (episode >= pattern_start and current_selection_tabu and last_selection_action in
                    indices_s):
                # use_selection_tabu = (True, last_selection_action)   #lyp added this code
                cnt = 0
                while (piece_id in current_selection_tabu[last_selection_action][0]) or mask[
                    piece_id].item() == 0:  # lyp added this code
                    piece_id = sorted_piece_idx[cnt].item()  # 重新选取
                    cnt += 1
                    if cnt >= len(sorted_piece_idx):
                        print("No available piece in the current selection pattern.")
                        piece_id = piece_dist.sample().item()
                        break
            elif (episode >= pattern_start and current_selection_tabu is not None and len(env.placed_pieces) == 0 and (
                    not current_selection_tabu) and "empty" in
                  current_selection_tabu):
                # use_selection_tabu = (True, last_selection_action)   #lyp added this code
                cnt = 0
                while piece_id in current_selection_tabu['empty'][0] or mask[  # lyp added this part of code
                    piece_id].item() == 0:  # lyp added this code
                    piece_id = sorted_piece_idx[cnt].item()  # 重新选取
                    cnt += 1
                    if cnt >= len(sorted_piece_idx):
                        print("No available piece in the current selection pattern.")
                        piece_id = piece_dist.sample().item()
                        break
            else:
                pass
            cnt = 0
            while pattern_mask[piece_id].item() == 0:  # lyp added this code
                piece_id = sorted_piece_idx[cnt].item()  # 重新选取
                cnt += 1
            cnt = 0
            while mask[piece_id].item() == 0:  # lyp added this code
                piece_id = sorted_piece_idx[cnt].item()  # 重新选取
                cnt += 1

            piece_log_prob = piece_dist.log_prob(torch.tensor(piece_id).to(device)).item()

            # 先得到当前step的已放置零件状态编码，用于分别与其他编码拼接
            state_graph = combine_graphs(
                get_pieces_from_id(env.placed_pieces, env.located_pieces),
                boundary=Polygon([(0, 0), (env.bin_width, 0), (env.bin_width, env.bin_height),
                                  (0, env.bin_height)])
            )
            no_placed_state_graph = combine_graphs(
                get_pieces_from_id(env.no_placed_pieces, env.located_pieces)
            )
            graph_data = nx_to_data_0(state_graph).to(device)
            no_placed_graph_data = nx_to_data_0(no_placed_state_graph).to(device)
            placed_state_encoding = gnn_model(graph_data).mean(dim=0)
            no_placed_state_encoding = gnn_model(no_placed_graph_data).mean(dim=0)

            next_state_s, reward_s, done = env.step_s(piece_id, gnn_model, placed_state_encoding, no_placed_state_encoding)
            last_selection_action = piece_id

            next_state_r = torch.zeros(64, device=device)
            if len(env.placed_pieces) > 0:  # 放完第一个零件后才开始获取旋转状态
                next_state_r = env.get_rotation_state(gnn_model, piece_id, placed_state_encoding)
                angle_buffer['next_states'].append(next_state_r)

            # 2️⃣ 选择旋转角度（rotation）
            piece_graph = polygon_to_graph(env.input_pieces[piece_id])
            piece_data = nx_to_data_0(piece_graph).to(device)
            piece_encoding = gnn_model(piece_data).mean(dim=0)
            angle_state_encoding = torch.cat([state_encoding, piece_encoding]).to(device)
            angle_logit = angle_selector(angle_state_encoding)
            angle_logit = torch.where(torch.isnan(angle_logit), torch.full_like(angle_logit, 1e-4), angle_logit)
            angle_prob = torch.softmax(angle_logit, dim=-1)
            angle_dist = torch.distributions.Categorical(angle_prob)
            sorted_angle_idx = list(torch.argsort(angle_dist.probs, descending=True))  # lyp added this code

            indices = []
            if current_angle_pattern:
                indices = list(current_angle_pattern.keys())
            if episode >= pattern_start and current_angle_pattern is not None and indices is not None and piece_id in indices:
                angle = current_angle_pattern[piece_id][0]
                use_angle_pattern = (True, piece_id)
            else:
                angle = angle_dist.sample().item()

            if len(_instance.rotations) != 1:
                if current_angle_tabu is not None:
                    indices = list(current_angle_tabu.keys())
                if episode >= pattern_start and current_angle_tabu is not None and indices is not None and piece_id in indices:
                    # use_angle_tabu = (True, piece_id)
                    cnt = 0
                    while angle == current_angle_tabu[piece_id][0]:  # lyp added this code
                        angle = sorted_angle_idx[cnt].item()  # 重新选取
                        cnt += 1
                else:
                    pass

            angle_log_prob = angle_dist.log_prob(torch.tensor(angle).to(device)).item()
            reward_r, done = env.step_r(piece_id, angle)

            # 3️⃣ 位置选择直接通过 nfp 放置
            next_state, reward, sub_rewards, done, final_piece, placed_flag = \
                env.step(piece_id, angle, None, _instance, nfp_cache, step_id=step, scheme=placement_scheme)

            # 若未成功放置，则跳过记录（或根据需要记录惩罚）
            if not placed_flag:
                continue

            step += 1
            # 记录当前放置信息
            episode_placement[piece_id] = {
                "id": piece_id,
                "angle": angle,  # 注意这里的角度为离散索引
                "coord": list(env.located_pieces[piece_id].exterior.coords),
                "centroid": env.located_pieces[piece_id].centroid.coords[0],
            }
            current_action_sequence.append((piece_id, angle, -1))

            next_state_graph = combine_graphs(
                get_pieces_from_id(env.placed_pieces, env.located_pieces),
                boundary=Polygon([(0, 0), (_instance.bin_width, 0), (_instance.bin_width, _instance.bin_height),
                                  (0, _instance.bin_height)])
            )
            next_graph_data = nx_to_data_0(next_state_graph).to(device)
            next_state_encoding = gnn_model(next_graph_data).mean(dim=0)

            piece_buffer['states'].append(selection_encoding.squeeze())
            piece_buffer['actions'].append(piece_id)
            piece_buffer['rewards'].append(reward_s)
            piece_buffer['next_states'].append(next_state_s)
            piece_buffer['dones'].append(done)
            piece_buffer['log_probs'].append(piece_log_prob)

            angle_buffer['states'].append(angle_state_encoding.squeeze())
            angle_buffer['actions'].append(angle)
            angle_buffer['rewards'].append(reward_r)
            angle_buffer['dones'].append(done)
            angle_buffer['log_probs'].append(angle_log_prob)

            # buffer用于critic
            buffer['states'].append((torch.cat([selection_encoding.squeeze(), angle_state_encoding.squeeze()])))  # 64
            buffer['actions'].append((piece_id, angle))  # 可用tuple记录多个动作
            buffer['rewards'].append(sum(sub_rewards))  # 也可加权组合各个子奖励
            buffer['next_states'].append(torch.stack([next_state_s, next_state_r], dim=0))
            buffer['dones'].append(done)
            buffer['log_probs'].append((piece_log_prob, angle_log_prob))
            # 更新当前状态
            state_encoding = next_state_encoding

            Z.append(
                {
                    'selection_state': selection_encoding.squeeze(),
                    'angle_state': angle_state_encoding.squeeze(),
                    'selection_action': piece_id,
                    'angle_action': angle,
                }
            )

            if visualize:
                piece_in_bins = get_pieces_from_id(env.placed_pieces, env.located_pieces)
                visualize_polygon_list(piece_in_bins, _instance.bin_width, _instance.bin_height)

        # if episode % 10 == 0:
        #     piece_in_bins = get_pieces_from_id(env.placed_pieces, env.located_pieces)
        #     visualize_polygon_list(piece_in_bins, _instance.bin_width, _instance.bin_height)

        # 将本轮episode状态、动作序列等信息保存
        utilization = get_utilization_from_list(_instance, env.located_pieces)
        print(f'Episode {episode + 1} finished with utilization of {utilization}...', 'max_utilization:', max_utilization)
        print([current_action_sequence[i][0] for i in range(len(current_action_sequence))])
        print("num of patterns", len(current_selection_pattern))
        print("num of tabu", len(current_selection_tabu) if current_selection_tabu is not None else 0)
        # print("num of patterns",len(current_selection_pattern),'\n',current_selection_pattern)
        # print("num of tabu", len(current_selection_tabu) if current_selection_tabu is not None else 0,'\n',current_selection_tabu)
        episode_utilization.append(utilization)
        all_episode_records.append((utilization, current_action_sequence))

        this_record = {
            'placement': episode_placement,
            'utilization': utilization,
        }
        all_placements[episode] = this_record

        pattern_tracker.add(this_record)

        # PPO更新：先计算统一经验的 advantage
        piece_advantages = compute_advantages(buffer['rewards'], [critic(s).item() for s in buffer['states']], buffer['dones'],
                                              gamma, gae_lambda)

        total_samples = len(buffer['states'])
        batch_size = 64
        indices = np.random.randint(total_samples, size=batch_size)
        Z_length = 10  # NOTE
        Z_weight = 0.05  # NOTE

        for epoch_idx in range(10):
            np.random.shuffle(indices)
            total_loss = 0
            for start in range(0, total_samples, max(batch_size, total_samples)):
                mb_idx = indices[start:start + batch_size]
                # 构造mini-batch
                mb_states = torch.stack([buffer['states'][i] for i in mb_idx]).to(device)
                mb_selection_states = torch.stack([piece_buffer['states'][i] for i in mb_idx]).to(device)
                mb_angle_states = torch.stack([angle_buffer['states'][i] for i in mb_idx]).to(device)
                mb_actions = torch.tensor([piece_buffer['actions'][i] for i in mb_idx], dtype=torch.long, device=device)
                mb_angles = torch.tensor([angle_buffer['actions'][i] for i in mb_idx], dtype=torch.long, device=device)
                mb_old_piece_log_probs = torch.tensor([piece_buffer['log_probs'][i] for i in mb_idx], dtype=torch.float32,
                                                      device=device)
                mb_old_angle_log_probs = torch.tensor([angle_buffer['log_probs'][i] for i in mb_idx], dtype=torch.float32,
                                                      device=device)
                mb_advantages = torch.tensor([piece_advantages[i] for i in mb_idx], dtype=torch.float32, device=device)
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 对应的critic目标
                mb_returns = torch.tensor([piece_advantages[i] for i in mb_idx], dtype=torch.float32, device=device)

                # 更新零件选择器 (actor1)
                mb_piece_logit = piece_selector(mb_selection_states)
                mb_piece_probs = torch.softmax(mb_piece_logit, dim=-1)
                mb_piece_dist = torch.distributions.Categorical(mb_piece_probs)
                mb_new_piece_log_probs = mb_piece_dist.log_prob(mb_actions)
                ratio = torch.exp(mb_new_piece_log_probs - mb_old_piece_log_probs)
                surrogate1 = ratio * mb_advantages
                surrogate2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * mb_advantages
                piece_policy_loss = -torch.min(surrogate1, surrogate2).mean()
                piece_entropy = mb_piece_dist.entropy().mean()
                loss_piece = piece_policy_loss - entropy_coefficient * piece_entropy

                # 更新角度选择器 (actor2)
                mb_angle_logit = angle_selector(mb_angle_states)
                mb_angle_probs = torch.softmax(mb_angle_logit, dim=-1)
                mb_angle_dist = torch.distributions.Categorical(mb_angle_probs)
                mb_new_angle_log_probs = mb_angle_dist.log_prob(mb_angles)
                ratio_angle = torch.exp(mb_new_angle_log_probs - mb_old_angle_log_probs)
                surrogate1_angle = ratio_angle * mb_advantages
                surrogate2_angle = torch.clamp(ratio_angle, 1 - clip_epsilon, 1 + clip_epsilon) * mb_advantages
                angle_policy_loss = -torch.min(surrogate1_angle, surrogate2_angle).mean()
                angle_entropy = mb_angle_dist.entropy().mean()
                loss_angle = angle_policy_loss - entropy_coefficient * angle_entropy

                # 更新Critic网络
                mb_values = critic(mb_states).view(-1)
                critic_loss = (mb_returns - mb_values).pow(2).mean()

                # diversity loss
                sample_Z = random.sample(Z, min(len(Z), Z_length))  # lyp
                sum_D = 0
                # for i in range(len(sample_Z)):
                #     piece_logit = piece_selector(sample_Z[i]['selection_state'])
                #     piece_prob = torch.softmax(piece_logit, dim=-1)
                #     piece_dist = torch.distributions.Categorical(piece_prob)
                #     piece_id = piece_dist.sample().item()
                #
                #     angle_logit = angle_selector(sample_Z[i]['angle_state'])
                #     angle_prob = torch.softmax(angle_logit, dim=-1)
                #     angle_dist = torch.distributions.Categorical(angle_prob)
                #     angle_id = angle_dist.sample().item()
                #
                #     mse_D = (((piece_id - sample_Z[i]['selection_action']) ** 2 + (angle_id - sample_Z[i]['angle_action']) ** 2)
                #              / 2)
                #     sum_D += mse_D

                piece_logit_list = [sample_Z[i]['selection_state'] for i in range(Z_length)]
                piece_logit_tensor = torch.stack(piece_logit_list).to(device)
                piece_prob_tensor = torch.softmax(piece_logit_tensor, dim=-1)
                piece_dist_tensor = torch.distributions.Categorical(piece_prob_tensor)
                piece_id_tensor = piece_dist_tensor.sample()
                piece_id_list = piece_id_tensor.tolist()

                angle_logit_list = [sample_Z[i]['angle_state'] for i in range(Z_length)]
                angle_logit_tensor = torch.stack(angle_logit_list).to(device)
                angle_prob_tensor = torch.softmax(angle_logit_tensor, dim=-1)
                angle_dist_tensor = torch.distributions.Categorical(angle_prob_tensor)
                angle_id_tensor = angle_dist_tensor.sample()
                angle_id_list = angle_id_tensor.tolist()

                for i in range(Z_length):
                    mse_D = (((piece_id_list[i] - sample_Z[i]['selection_action']) ** 2 + (
                            angle_id_list[i] - sample_Z[i]['angle_action']) ** 2) / 2)
                    sum_D += mse_D

                ED = sum_D / Z_length

                if len(_instance.rotations) == 1:
                    total_loss += (loss_piece + critic_loss - ED * Z_weight)
                else:
                    total_loss += (loss_piece + loss_angle + critic_loss - ED * Z_weight)

            optimizer_ac.zero_grad()
            total_loss.backward()
            optimizer_ac.step()

        # 记录利用率与保存 the best placement 信息
        with open(utilization_log_path, 'a') as log_file:
            log_file.write(f"{episode + 1},{utilization}\n")

        if resume and (episode + 1) % 5 == 0:
            checkpoint_state = {
                'piece_selector': piece_selector.state_dict(),
                'angle_selector': angle_selector.state_dict(),
                'critic': critic.state_dict(),
                'optimizer': optimizer_ac.state_dict(),
                'gnn_model': gnn_model.state_dict(),
                'episode': episode + 1
            }
            save_checkpoint(checkpoint_state, checkpoint_path)
            save_trained_model(models, save_model_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        if utilization > best_placement[1]:
            pieces_info = {}
            for i in range(len(env.located_pieces)):
                pieces_info[i] = list(env.located_pieces[i].exterior.coords)
            best_placement = (pieces_info, utilization)

        if utilization > max_utilization and episode > 0:  # 有效，更新

            non_improve_count = 0  # lyp added this
            # if use_selection_pattern[0]:
            #     current_selection_pattern[use_selection_pattern[-1]][-1] = episode-1
            #     current_selection_pattern[use_selection_pattern[-1]][1] = episode
            # if use_angle_pattern[0]:
            #     current_angle_pattern[use_angle_pattern[-1]][-1] = episode-1
            #     current_angle_pattern[use_angle_pattern[-1]][1] = episode
            # if use_selection_tabu[0]:
            #     current_selection_tabu[use_selection_tabu[-1]][-1] = episode
            # if use_angle_tabu[0]:
            #     current_angle_tabu[use_angle_tabu[-1]][-1] = episode
            for key in current_selection_pattern:
                current_selection_pattern[key][-1] = episode  # 更新episode
                current_selection_pattern[key][1] = episode  # 更新episode
            for key in current_angle_pattern:
                current_angle_pattern[key][-1] = episode  # 更新episode
                current_angle_pattern[key][1] = episode  # 更新episode
        # else:
        # lyp added this
        # if utilization>=max_utilization:
        #     non_improve_count=0  # lyp added this
        non_improve_count += 1
        max_utilization = max(max_utilization, utilization)  # NOTE：若在tenure内，pattern或tabu没有使max_utilization改善则移除模式

        # NOTE：每隔swap interval轮重放一次，提取当前top k sequence进行局部搜索
        # 若得到了best_placement则更新
        if episode > 0 and episode % swap_interval == 0:
            info = {
                'name': instance_name,
                'polygons': env.input_pieces,
                'bin': None,
                'width': _instance.bin_width,
                'height': _instance.bin_height,
                'rotations': CONFIG_DICT_ALL[instance_name]['rotations'],
                'nfp_cache': load_nfp_cache(f"nfp/{instance_name}.json")
            }
            selection_sequences, angle_sequences = get_sequence(pattern_tracker.top_k())
            # print(CONFIG_DICT_ALL[instance_name]['rotations'])
            used, record = perform_swap(info, selection_sequences, angle_sequences, instance_name, utilization,
                                        env.located_pieces, scheme=placement_scheme)
            swapped_utilization = record['utilization']
            print(f'swap finished. Best utilization: {swapped_utilization}')

            pattern_tracker.add(record)
            if swapped_utilization > max_utilization:
                max_utilization = swapped_utilization
                swapped_info = [record['placement'][k]['coord'] for k in range(len(record['placement']))]
                best_placement = (swapped_info, swapped_utilization)

        print(f'finish episode {episode + 1} with utilization {utilization}', 'max_utilization:', max_utilization)
    return total_episode_num


def compute_advantages(rewards, values, dones, gamma, gae_lambda):
    advantages = []
    last_advantage = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            delta = rewards[t] - values[t]
        else:
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        last_advantage = delta + gamma * gae_lambda * (1 - dones[t]) * last_advantage
        advantages.insert(0, last_advantage)
    return advantages


def load_checkpoint(models, optimizer, filename):
    checkpoint = torch.load(filename, map_location=device)

    # Load model state dictionaries
    models['piece_selector'].load_state_dict(checkpoint['piece_selector'])
    models['angle_selector'].load_state_dict(checkpoint['angle_selector'])
    models['critic'].load_state_dict(checkpoint['critic'])
    models['gnn_model'].load_state_dict(checkpoint['gnn_model'])

    # Load optimizer state dictionary
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Get the episode number from the checkpoint
    episode = checkpoint['episode']
    print(f"Checkpoint loaded from {filename}, resuming from episode {episode}")

    return episode


def save_checkpoint(state, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the checkpoint as a dictionary
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def save_best_placement(best_placement, results_filename):
    pieces_info, utilization = best_placement
    data = {
        "pieces_info": pieces_info,
        "utilization": utilization
    }
    with open(results_filename, 'w') as f:
        # noinspection PyTypeChecker
        json.dump(data, f, indent=4)
    print(f"Best placement saved to {results_filename}")


def save_trained_model(models, path_prefix):
    """
    保存训练好的模型
    """
    torch.save(models['piece_selector'].state_dict(), f"{path_prefix}_piece_selector.pt")
    torch.save(models['angle_selector'].state_dict(), f"{path_prefix}_angle_selector.pt")
    torch.save(models['critic'].state_dict(), f"{path_prefix}_critic.pt")
    torch.save(models['gnn_model'].state_dict(), f"{path_prefix}_gnn_model.pt")
    print(f"Trained models saved with prefix {path_prefix}")


def launch(instance_name, _episode, _rotation, threshold, resume=True, visualize=False):
    instance_path = f"Nesting/{instance_name}.xml"
    result_path_30 = f"NFP_records_LS_2/30/results_nomatch/{instance_name}_result.json"
    result_path_60 = f"NFP_records_LS_2/60/results_nomatch/{instance_name}_result.json"
    result_path_120 = f"NFP_records_LS_2/120/results_nomatch/{instance_name}_result.json"
    result_path_240 = f"NFP_records_LS_2/240/results_nomatch/{instance_name}_result.json"
    result_path = [result_path_30, result_path_60, result_path_120, result_path_240]

    save_model_path = f"NFP_records_LS_2/trained_models_nomatch/{instance_name}/{instance_name}"

    checkpoint_path = f"NFP_records_LS_2/checkpoint_nomatch/{instance_name}.pth"

    # 日志统一记录
    utilization_log_path = f"NFP_records_LS_2/logs_nomatch/{instance_name}_utilization.csv"

    save_placement_path_30 = f"NFP_records_LS_2/30/placements/{instance_name}_placement.json"
    save_placement_path_60 = f"NFP_records_LS_2/60/placements/{instance_name}_placement.json"
    save_placement_path_120 = f"NFP_records_LS_2/120/placements/{instance_name}_placement.json"
    save_placement_path_240 = f"NFP_records_LS_2/240/placements/{instance_name}_placement.json"
    save_placement_path = [save_placement_path_30, save_placement_path_60, save_placement_path_120, save_placement_path_240]

    all_record_path = f"NFP_records_LS_2/all_records/{instance_name}_record.json"
    nfp_cache_path = f"nfp/{instance_name}.json"

    _rotations = _rotation
    _polygons, _bin = parse_xml(instance_path)
    _instance = Instance(_polygons, _bin[2][1] - _bin[1][1] + 0.05, _bin[1][0] - _bin[0][0] + 20, _rotations)
    training_cfg = {
        "episodes": _episode,
        "epochs": 50,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "resume": resume,
        "visualize": visualize,
        "checkpoint_path": checkpoint_path,
        "results_filename": result_path,
        "visualization_dir": '',
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "entropy_coefficient": 0.01,
        "clip_epsilon": 0.1,
        "matched_path": '',
        "save_model_path": save_model_path,
        "utilization_log_path": utilization_log_path,
        "save_placement_path": save_placement_path,
        "all_record_path": all_record_path,
        "nfp_cache_path": nfp_cache_path,
    }
    _instance.set_training_config(training_cfg)
    print("##############################")
    print(f"Start training for {instance_name}")
    print("##############################")

    start_time = time.time()
    eps = train(_instance, instance_name, threshold)
    end_time = time.time()

    return eps, int((end_time - start_time) / 60)


def get_sequence(record):
    selection_sequences = []
    angle_sequences = []
    for _dict in record:
        s_seq = []
        a_seq = []
        d = _dict['placement']
        for key, value in d.items():
            s_seq.append(value['id'])
            a_seq.append(value['angle'])
        selection_sequences.append(s_seq)
        angle_sequences.append(a_seq)
    return selection_sequences, angle_sequences


def get_selection_pattern(sequences, born_eps, mode='selection', replacement=False, double_sample_start=200):  # NOTE: lyp
    """
    获取选择模式
    :param sequences: top k sequences obtained from the record
    :param born_eps: current episode
    :param mode: 选择模式 or 禁忌模式
    :param replacement:
    :param double_sample_start: 之前采样一次，之后在两个序列中采样
    :return: patterns
    """
    if replacement:
        alpha = random.uniform(0.7, 0.9)
    else:
        alpha = 0.6

    if not (mode == 'selection' or mode == 'tabu'):
        raise ValueError('mode should be "selection" or "tabu"')

    if mode == 'selection':
        if born_eps < double_sample_start:
            alpha = 0.4
            number_of_selections = len(sequences[0]) * alpha
            pattern = {}
            common_pairs = set()
            seq_id_1 = random.randint(0, len(sequences) - 1)  # select one sequence randomly from the top k
            seq_id = seq_id_1
            random_starts = random.sample(list(range(len(sequences[seq_id]))), int(number_of_selections))  # 从序列中随机选择alpha比例的开始位置
            # random_starts=set(range(len(sequences[seq_id])))
            # random_removed=random.sample(list(range(len(sequences[seq_id]))), len(sequences[seq_id])//3)
            for i in random_starts:
                if i + 1 < len(sequences[seq_id]):
                    common_pairs.add((sequences[seq_id][i], sequences[seq_id][i + 1]))
                if i == 0:
                    common_pairs.add(('empty', sequences[seq_id][i]))  # 规定了第一个放置的pattern
            # random_starts=set(range(len(sequences[seq_id])))
            # random_removed=random.sample(list(range(len(sequences[seq_id]))), len(sequences[seq_id])//3)
            # for i in random_starts:
            #     if i + 1 < len(sequences[seq_id]):
            #         common_pairs.add((sequences[seq_id][i], sequences[seq_id][i + 1]))
            #     if i == 0:
            #         common_pairs.add(('empty', sequences[seq_id][i]))

            for a, b in common_pairs:
                if a not in pattern:
                    pattern[a] = [b, born_eps, born_eps - 1]
        else:  # 开始用pattern
            number_of_selections = len(sequences[0]) * alpha // 2  # 选了alpha/2比例个pattern
            pattern = {}
            common_pairs = set()
            seq_id_1 = random.randint(0, len(sequences) - 1)
            seq_id = seq_id_1
            random_starts = random.sample(list(range(len(sequences[seq_id]))), int(number_of_selections))
            # random_starts=set(range(len(sequences[seq_id])))
            # random_removed=random.sample(list(range(len(sequences[seq_id]))), len(sequences[seq_id])//3)
            for i in random_starts:
                if i + 1 < len(sequences[seq_id]):
                    common_pairs.add((sequences[seq_id][i], sequences[seq_id][i + 1]))
                if i == 0:
                    common_pairs.add(('empty', sequences[seq_id][i + 1]))

            number_of_selections = int(len(sequences[0]) * alpha) - int(number_of_selections)  # 又选了alpha/2比例个pattern
            seq_id_2 = random.randint(0, len(sequences) - 1)
            if seq_id_2 == seq_id_1:
                seq_id = (seq_id_2 + 1) % len(sequences)
            else:
                seq_id = seq_id_2  # 选了另一个sequence

            random_starts = random.sample(list(set(range(len(sequences[seq_id]))) - set(random_starts)),
                                          int(number_of_selections))
            # random_starts=set(range(len(sequences[seq_id])))
            # random_removed=random.sample(list(range(len(sequences[seq_id]))), len(sequences[seq_id])//3)
            for i in random_starts:
                if i + 1 < len(sequences[seq_id]):
                    common_pairs.add((sequences[seq_id][i], sequences[seq_id][i + 1]))
                if i == 0:
                    common_pairs.add(('empty', sequences[seq_id][i + 1]))

            for a, b in common_pairs:
                if a not in pattern:
                    pattern[a] = [b, born_eps, born_eps - 1]

    else:  # mode = 'tabu'
        # 如果输入为空，直接返回None
        if not sequences:
            return None

        common_pairs = None
        current_pairs1 = None
        pattern = {}
        # 遍历每个序列，提取相邻的数对
        for _ in range(len(sequences)):
            idx = random.sample(range(len(sequences)), 3)
            seq = sequences[idx[0]]
            seq1 = sequences[idx[1]]
            seq2 = sequences[idx[2]]  # 采样三个不同的序列，后两个用于交叉

            if len(seq) < 2:  # 为空或长度小于2的序列直接认为没有相邻数对
                current_pairs = set()
                current_pairs2 = set()
            else:  # 分别采样出三个序列中所有的相邻数对
                current_pairs = {(seq[i], seq[i + 1]) for i in range(len(seq) - 1)}
                current_pairs1 = {(seq1[i], seq1[i + 1]) for i in range(len(seq1) - 1)}
                current_pairs2 = {(seq2[i], seq2[i + 1]) for i in range(len(seq2) - 1)}

            if common_pairs is None:
                common_pairs = current_pairs
            else:
                common_pairs = current_pairs.intersection(current_pairs2)  # 求交集
                common_pairs = current_pairs1.intersection(common_pairs)

            # 如果交集为空，则无需继续查找，认为不存在禁忌模式
            if not common_pairs:
                return pattern

            for a, b in common_pairs:
                if a not in pattern:
                    pattern[a] = [{b}, born_eps, born_eps - 1]
                else:
                    pattern[a][0].add(b)  # 一个索引的后继可以为多个索引，用集合表示
    return pattern


def get_angle_pattern(sequences, born_eps):
    if not sequences:
        return None
    min_length = min(len(seq) for seq in sequences)
    pattern = {}
    for k in range(min_length):
        common_val = sequences[0][k]
        if all(seq[k] == common_val for seq in sequences):
            pattern[k] = [common_val, born_eps, born_eps - 1]
    return pattern


if __name__ == '__main__':

    for name in CONFIG_DICT_ALL:
        time_rec = {}
        episode_num_rec = {}
        work = CONFIG_DICT_ALL[name]
        rotation = work['rotations']

        # NOTE：为多种旋转设计的旋转角度
        # rotation = [i * 15 for i in range(24)]

        episode = work['episodes']
        threshold = 0
        # threshold = work['threshold']
        eps, used_time = launch(
            instance_name=name, _episode=episode, _rotation=rotation, threshold=threshold, resume=False, visualize=False
        )
        time_rec[name] = used_time
        # with open(f'NFP_records_LS/time/{name}.json', 'w') as f:
        #     # noinspection PyTypeChecker
        #     json.dump(time_rec, f, indent=4)
        with open('NFP_records_LS_2/eps.json', 'w') as f:
            # noinspection PyTypeChecker
            json.dump(episode_num_rec, f, indent=4)

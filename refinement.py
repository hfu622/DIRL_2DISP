import math
import random
import warnings

from tools import *

warnings.filterwarnings("ignore")


def place_one(input_pieces, located_pieces, angle_list, moving_id, moving_angle, placed_indices, nfp_cache, scheme='BL'):
    # moving_angle *= 90
    # print(moving_angle)
    polygon_nfp_cache = nfp_cache[0]
    inner_nfp_cache = nfp_cache[1]
    moving_piece = input_pieces[moving_id]
    moving_piece = rotate(moving_piece, moving_angle)
    inner_nfp_key = get_inner_nfp_key(moving_id, moving_angle)
    try:
        inner_nfp_record = inner_nfp_cache[inner_nfp_key]
    except KeyError:
        raise KeyError(f"Inner NFP not found for key -> {inner_nfp_key}.")
    inner_nfp = standardize_polygon(inner_nfp_record).buffer(0.025)  # no need to calculate offset
    if inner_nfp.is_empty:
        raise ValueError("inner_nfp is empty.")
    polygon_nfp_list = []

    for static_id in placed_indices:
        static_piece = located_pieces[static_id]
        static_angle = angle_list[static_id]
        polygon_nfp_key = get_polygon_nfp_key(static_id, static_angle, moving_id, moving_angle)
        try:
            polygon_nfp_record = polygon_nfp_cache[polygon_nfp_key]
        except KeyError:
            raise KeyError(f"Polygon NFP not found for key -> {polygon_nfp_key}.")
        polygon_nfp = standardize_polygon(polygon_nfp_record)

        placed_centroid = static_piece.centroid
        original_centroid = input_pieces[static_id].centroid
        offset_x = placed_centroid.x - original_centroid.x
        offset_y = placed_centroid.y - original_centroid.y
        polygon_nfp = translate(polygon_nfp, offset_x, offset_y)
        polygon_nfp_list.append(polygon_nfp)

    if polygon_nfp_list is not None:
        union_polygon_nfp = unary_union(polygon_nfp_list).buffer(-0.025)
        nfp = inner_nfp.difference(union_polygon_nfp)
    else:
        nfp = inner_nfp

    if scheme == 'BL':
        located_point = get_BL_point(nfp)
    elif scheme == 'MU':
        located_point = get_MU_point(nfp, located_pieces, moving_piece)
    else:
        raise ValueError(f"Invalid scheme -> {scheme}.")

    ref_point = moving_piece.exterior.coords[0]
    translation_x = located_point[0] - ref_point[0]
    translation_y = located_point[1] - ref_point[1]
    located_piece = translate(moving_piece, translation_x, translation_y)
    angle_list[moving_id] = moving_angle

    located_pieces[moving_id] = located_piece
    placed_indices.append(moving_id)

    return located_pieces, angle_list, placed_indices


def place_by_order(input_pieces, selection_sequence, allowed_rotations, models, _name, _width, _height, nfp_cache, rotation,
                   scheme="BL"):
    located_pieces = input_pieces.copy()
    angle_list = [0] * len(input_pieces)
    placed_pieces = []

    for idx in selection_sequence:
        moving_id = idx
        moving_piece = input_pieces[moving_id]

        if rotation == 'infer':
            # 用预训练模型推理角度
            angle_selector = models['angle_selector'].to(device).eval()
            gnn_model = models['gnn_model'].to(device).eval()

            state_graph = combine_graphs(
                get_pieces_from_id(placed_pieces, located_pieces),
                boundary=Polygon([(0, 0), (_width, 0), (_width, _height),
                                  (0, _height)])
            )

            graph_data = nx_to_data_0(state_graph).to(device)
            gnn_model.train()
            placed_state_encoding = gnn_model(graph_data).mean(dim=0)
            state_encoding = placed_state_encoding

            piece_graph = polygon_to_graph(input_pieces[moving_id])
            piece_data = nx_to_data_0(piece_graph).to(device)
            piece_encoding = gnn_model(piece_data).mean(dim=0)
            angle_state_encoding = torch.cat([state_encoding, piece_encoding]).to(device)
            angle_logit = angle_selector(angle_state_encoding)
            angle_logit = torch.where(torch.isnan(angle_logit), torch.full_like(angle_logit, 1e-4), angle_logit)
            angle_prob = torch.softmax(angle_logit, dim=-1)
            angle_dist = torch.distributions.Categorical(angle_prob)
            moving_angle = angle_dist.sample().item()

        elif rotation == 'random':
            # 随机选择角度
            # print(f'by order:{allowed_rotations}')
            moving_angle = random.choice(allowed_rotations)
        elif rotation == 'greedy':
            # 贪心选择角度，选择放完后最靠下的角度
            moving_angle = 0
            top_y = 1e6
            for a in allowed_rotations:
                located_pieces, angle_list, placed_pieces = place_one(input_pieces, located_pieces, angle_list, moving_id, a,
                                                                      placed_pieces, nfp_cache, scheme)
                cur_top_y = located_pieces[moving_id].bounds[3]
                if cur_top_y < top_y:
                    moving_angle = a
                    top_y = cur_top_y
        else:
            raise ValueError(f"Invalid rotation mode -> {rotation}.")

        located_pieces, angle_list, placed_pieces = place_one(input_pieces, located_pieces, angle_list, moving_id, moving_angle,
                                                              placed_pieces, nfp_cache, scheme)

    # visualize_polygon_list(located_pieces, _width, _height)
    ur = get_utilization(_width, located_pieces)
    return ur, located_pieces, angle_list


def local_search_swap(info, selection_sequence, _instance_name, original_utilization, original_located_pieces, models,
                      accept_prob=0.1, scheme='BL'):
    input_pieces = info['polygons']
    _width = info['width']
    _height = info['height']
    allowed_rotations = info['rotations']
    nfp_cache = info['nfp_cache']

    best_sequence = selection_sequence.copy()
    best_angle_sequence = [0] * len(input_pieces)
    init_sequence = selection_sequence.copy()  # 初始解
    current_sequence = init_sequence.copy()

    best_located_pieces = original_located_pieces.copy()
    best_utilization = original_utilization
    length = len(selection_sequence)
    swap_num = min(int(length * (length - 1) / 2) + 1, 1000)
    # T0 = math.exp(-7.38 / swap_num)

    last_utilization = best_utilization
    last_placed_pieces = best_located_pieces.copy()

    last_i, last_j = -1, -1
    for _ in range(swap_num):
        i, j = random.sample(range(length), 2)
        if i == last_i or j == last_j:  # avoid swapping back
            continue
        current_sequence[i], current_sequence[j] = current_sequence[j], current_sequence[i]  # swap
        last_i, last_j = i, j

        # print(f'local search:{allowed_rotations}')
        new_utilization, new_located_pieces, new_angle_sequence = place_by_order(input_pieces, current_sequence,
                                                                                 allowed_rotations, models, _instance_name,
                                                                                 _width, _height, nfp_cache, rotation='random',
                                                                                 scheme=scheme)
        # 邻域中的解和上一轮的解相比
        if new_better_than_old(new_ur=new_utilization, old_ur=last_utilization, new_located_pieces=new_located_pieces,
                               old_located_pieces=last_placed_pieces):
            # 比上一轮的解好，接受
            last_utilization = new_utilization
            last_placed_pieces = new_located_pieces.copy()
            if new_utilization > best_utilization:  # 维护最优解
                best_utilization = new_utilization
                best_sequence = current_sequence.copy()
                best_located_pieces = new_located_pieces.copy()
                best_angle_sequence = new_angle_sequence.copy()
        else:
            p_accept = random.random()
            if p_accept < accept_prob:
                # 执行跳转，即使是一个次优解；否则这一轮不跳转新解，进入新一轮继续搜索
                last_utilization = new_utilization
                last_placed_pieces = new_located_pieces.copy()

        # print(f'utilization in this iteration: {new_utilization:.4f}')

    # print(f'current iteration ur: {new_utilization:.4f}, best ur: {best_utilization:.4f}')

    return best_sequence, best_angle_sequence, best_located_pieces, best_utilization


def place_by_all_order(info, selection_sequence, angle_seq, allowed_rotations, scheme='BL'):
    input_pieces = info['polygons']
    nfp_cache = info['nfp_cache']
    _width = info['width']
    located_pieces = input_pieces.copy()
    angle_list = [0] * len(input_pieces)
    placed_pieces = []

    # print(angle_seq)

    for idx in range(len(input_pieces)):
        moving_id = selection_sequence[idx]
        moving_angle = allowed_rotations[angle_seq[idx]]

        located_pieces, angle_list, placed_pieces = place_one(input_pieces, located_pieces, angle_list, moving_id,
                                                              moving_angle, placed_pieces, nfp_cache, scheme)
    ur = get_utilization(_width, located_pieces)
    return ur, located_pieces


def new_better_than_old(new_ur, old_ur, new_located_pieces, old_located_pieces):
    if new_ur > old_ur:
        return True
    elif math.fabs(new_ur - old_ur) <= 1e-6:  # 利用率不变但重心降低，也接受
        new_centroid_y = sum([p.centroid.y for p in new_located_pieces]) / len(new_located_pieces)
        old_centroid_y = sum([p.centroid.y for p in old_located_pieces]) / len(old_located_pieces)
        if new_centroid_y < old_centroid_y:
            return True
        else:
            return False
    else:
        return False


def perform_swap(info, top_seq_s, top_seq_a, _name, original_utilization, original_located_pieces, scheme):
    """返回值分别为：是否与初始解不同，搜索到最优解的打包方案"""
    # 不加model，直接random
    # original_located_pieces是这一轮迭代获得的解
    # k组记录中最优的利用率/序列/打包方案
    best_ur_overall = -1.0
    best_located_overall = None
    best_seq_overall_s = None
    best_seq_overall_a = None

    allowed_rotations = info['rotations']

    final_placement = {}

    for i in range(len(top_seq_s)):
        # 每次循环跑一个记录
        cur_seq_selection = top_seq_s[i]
        cur_seq_angle = top_seq_a[i]
        initial_ur, initial_located = place_by_all_order(info, cur_seq_selection, cur_seq_angle, allowed_rotations, scheme=scheme)

        this_best_seq_s, this_best_seq_a, this_best_located, this_best_ur = local_search_swap(info, cur_seq_selection, _name,
                                                                                              initial_ur, initial_located,
                                                                                              models=None, scheme=scheme)
        # 这个initial_located是从k个记录中得到的打包方式

        if this_best_ur > best_ur_overall:
            best_ur_overall = this_best_ur
            best_located_overall = this_best_located.copy()
            best_seq_overall_s = this_best_seq_s.copy()
            best_seq_overall_a = this_best_seq_a.copy()

    for j in range(len(best_seq_overall_s)):
        final_placement[j] = {
            "id": best_seq_overall_s[j],
            # "angle": best_seq_overall_a[j] // 90,
            "angle": allowed_rotations.index(best_seq_overall_a[j]),  # NOTE：注意是索引
            "coord": list(best_located_overall[best_seq_overall_s[j]].exterior.coords),
            "centroid": best_located_overall[best_seq_overall_s[j]].centroid.coords[0],
        }
    rec = {
        'placement': final_placement,
        'utilization': best_ur_overall,
    }

    use = True
    if original_located_pieces == best_located_overall:
        use = False
    return use, rec

# if __name__ == '__main__':
#     for name in CONFIG_DICT_ALL:
#         print(f'Start swapping for instance {name}...')
#         instance_name = name
#         instance_path = f"Nesting/{instance_name}.xml"
#         _polygons, _bin = parse_xml(instance_path)
#
#         instance_info = {
#             'name': instance_name,
#             'polygons': _polygons,
#             'bin': _bin,
#             'width': _bin[2][1] - _bin[1][1] + 0.05,
#             'height': _bin[1][0] - _bin[0][0] + 20,
#             'rotations': CONFIG_DICT_ALL[name]['rotations'],
#             'nfp_cache': load_nfp_cache(f"nfp/{instance_name}.json")
#         }
#
#         with open(f"NFP_records_LS/top_sequences/{instance_name}.json", 'r', encoding='utf-8') as f:
#             # noinspection PyTypeChecker
#             data = json.load(f)
#             initial_selection_sequences = data['top_k_selection_seq']
#             initial_angle_sequences = data['top_k_angle_seq']
#
#         model_path = f"NFP_records_LS/checkpoint_nomatch/{instance_name}.pth"
#         gnn_model = GNNModel(input_dim=2, hidden_dim=64, output_dim=32).to(device)
#         angle_selector = AngleSelector(len(instance_info['rotations'])).to(device)
#
#         checkpoint = torch.load(model_path, map_location=device)
#         angle_selector.load_state_dict(checkpoint['angle_selector'])
#         gnn_model.load_state_dict(checkpoint['gnn_model'])
#         models = {
#             'gnn_model': gnn_model,
#             'angle_selector': angle_selector
#         }
#
#         best_utilization_overall = -1.0
#         best_located_pieces_overall = None
#         best_sequence_overall = None
#
#         for i in range(len(initial_selection_sequences)):
#             # 每次循环跑一个记录
#             cur_seq_s = initial_selection_sequences[i]
#             cur_seq_a = initial_angle_sequences[i]
#             initial_utilization, initial_located_pieces = place_by_all_order(instance_info, cur_seq_s, cur_seq_a)
#
#             this_best_sequence, this_best_located_pieces, this_best_utilization = local_search_swap(instance_info, cur_seq_s,
#                                                                                                     instance_name,
#                                                                                                     initial_utilization,
#                                                                                                     initial_located_pieces,
#                                                                                                     models)
#             # 执行 local search
#             if this_best_utilization > best_utilization_overall:
#                 best_utilization_overall = this_best_utilization
#                 best_located_pieces_overall = this_best_located_pieces.copy()
#                 best_sequence_overall = this_best_sequence.copy()
#
#         print(f"Best utilization: {best_utilization_overall:.4f}")
#         # print(f"Best sequence: {best_sequence_overall}")
#
# # 初始解->swap得到邻域->评估优劣->接受或拒绝->更新当前解，同时维护最优解

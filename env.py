from tools import *


class PipelineEnv:
    def __init__(self, instance: Instance):
        self.bin_width = instance.bin_width
        self.bin_height = instance.bin_height
        self.bin = Polygon([(0, 0), (self.bin_width, 0), (self.bin_width, self.bin_height), (0, self.bin_height)])
        self.input_pieces = instance.polygons  # 用Polygon对象表示，原始输入中的零件
        self.located_pieces = []  # 所有零件实际的位置和姿态
        self.angle_list = []
        self.no_placed_pieces = []  # 初始是所有零件的索引集合
        self.placed_pieces = []  # 初始为空
        self.gaps = []  # 空隙列表，初始为空；存储的是空隙的Polygon对象
        self.state = None
        # self.cur_placed_piece = self.update_placed_pieces()  # 维护当前已放置的多边形为一个，减少nfp计算开销？（多边形可能反而变得更复杂）

    def reset(self):
        # 返回的是状态tensor，即已放置的零件信息（具体信息）
        self.no_placed_pieces = [i for i in range(len(self.input_pieces))]
        self.placed_pieces = []
        self.gaps = [create_polygon([
            (0, 0), (self.bin_width, 0), (self.bin_width, self.bin_height), (0, self.bin_height)
        ])]  # 初始状态下的gap就是空的底板
        self.located_pieces = self.input_pieces.copy()
        self.angle_list = [0] * len(self.input_pieces)
        self.state = self.update_state()

    def get_utilization(self):
        top_y = self.get_top_y()
        sum_area = 0
        for p in self.located_pieces:
            sum_area += p.area  # polygon对象直接通过area属性获取面积
        return sum_area / (self.bin_width * top_y)

    def get_top_y(self):
        global_max = None

        for polygon in self.located_pieces:
            # 提取多边形外边界的所有坐标，获取其中所有的纵坐标
            max_y = max(coord[1] for coord in polygon.exterior.coords)

            # 更新全局最大值
            if global_max is None or max_y > global_max:
                global_max = max_y

        return global_max

    def get_reward_s(self, selected_action):  # 改为面积相对于最大面积的比值
        """use rectangular-ity as sub-reward"""
        p = self.input_pieces[selected_action]
        # x_min = get_min_x(p)
        # x_max = get_max_x(p)
        # y_min = get_min_y(p)
        # y_max = get_max_y(p)
        # return p.area / (x_max - x_min) * (y_max - y_min)
        return p.area / self.get_max_area()


    def step_s(self, piece_idx, cur_gae, placed_state_encoding, no_placed_state_encoding):
        # self.no_placed_pieces.remove(piece_idx)
        # self.placed_pieces.append(piece_idx)
        done = len(self.no_placed_pieces) == 0

        if done:
            next_state = torch.zeros(64).to(device)
            reward = self.get_utilization() * 1000
        else:
            next_state = torch.cat([placed_state_encoding, no_placed_state_encoding]).to(device)
            reward = self.get_reward_s(piece_idx)

        return next_state, reward, done

    def get_reward_r(self):
        placed_polygons = get_pieces_from_id(self.placed_pieces, self.located_pieces)
        total_area = 0
        x_min, y_min, x_max, y_max = 1e9, 1e9, -1e9, -1e9
        for pp in placed_polygons:
            total_area += pp.area
            x_min = min(x_min, get_min_x(pp))
            y_min = min(y_min, get_min_y(pp))
            x_max = max(x_max, get_max_x(pp))
            y_max = max(y_max, get_max_y(pp))
        return total_area / ((x_max - x_min) * (y_max - y_min))

    def get_rotation_state(self, cur_gae, cur_piece_id, placed_state_encoding):
        """placed_encoding concatenated with the selected piece's encoding"""
        p = self.input_pieces[cur_piece_id]
        g = polygon_to_graph(p)
        g_data = nx_to_data_0(g).to(device)
        selected_piece_encoding = cur_gae(g_data).mean(dim=0)
        return torch.cat([placed_state_encoding.to(device), selected_piece_encoding.to(device)]).to(device)

    def step_r(self, piece_idx, rotation):
        """
        旋转模型的step，不调整信息
        :param piece_idx: 被选中的零件索引
        :param rotation: 角度（角度制真实值）
        :return:
        """
        # self.located_pieces[piece_idx] = rotate(self.input_pieces[piece_idx], rotation)
        done = len(self.no_placed_pieces) == 0

        if done:
            reward = self.get_utilization() * 1000
        else:
            reward = self.get_reward_r()

        return reward, done  # next_state在选完下一个零件后再更新

    # def step(self, piece_idx, angle, x_range, _instance, step_id, pos=None):  # 主要就是放置的过程
    #     piece = rotate(self.input_pieces[piece_idx], angle * 90)  # Polygon旋转函数的angle是角度制
    #     if pos is not None:
    #         # 直接指定位置
    #         ref_x, ref_y = piece.centroid.coords[0]
    #         x, y = pos
    #         dx, dy = x - ref_x, y - ref_y
    #         located_piece = translate(piece, dx, dy)
    #     else:
    #         done = False
    #         placed_polys = get_pieces_from_id(self.placed_pieces, self.located_pieces)
    #         # located_piece = place_pieces_into_gap_BL(piece, gap, placed_polys)
    #         # located_piece = NFP_place(piece, self.bin, placed_polys, _instance)
    #         located_piece = svg_place(piece, self.bin, placed_polys)
    #
    #     # 到此零件位置确定，更新状态
    #     if piece_idx in self.no_placed_pieces:
    #         self.no_placed_pieces.remove(piece_idx)
    #     self.placed_pieces.append(piece_idx)
    #     self.located_pieces[piece_idx] = located_piece  # 更新零件信息，因为加入的旋转和移动操作
    #     self.gaps = (find_gaps(
    #         self.bin_width, self.bin_height, get_pieces_from_id(self.placed_pieces, self.located_pieces))
    #     )
    #     self.state = self.update_state()
    #     if len(self.no_placed_pieces) == 0:
    #         done = True
    #     prev_height = max([p.bounds[3] for p in self.located_pieces])
    #
    #     reward = self.get_reward(
    #         _instance=_instance,
    #         piece_idx=piece_idx,
    #         step_id=step_id,
    #         done=done,
    #         prev_height=prev_height
    #     )
    #
    #     sub_rewards = (
    #         self.get_selection_reward(done, step_id),
    #         self.get_angle_reward(piece_idx, done, step_id),
    #     )
    #
    #     print(f"Placed piece {piece_idx} with angle {angle}")
    #     return self.state, reward, sub_rewards, done, located_piece, True  # 返回的最后一个参数是是否成功放置

    def step(self, piece_idx, angle, x_range, _instance, nfp_cache, step_id, scheme, pos=None):  # 主要就是放置的过程
        piece = rotate(self.input_pieces[piece_idx], _instance.rotations[angle], origin="centroid")  # Polygon旋转函数的angle是角度制
        rotation = _instance.rotations[angle]
        if pos is not None:
            # 直接指定位置
            ref_x, ref_y = piece.centroid.coords[0]
            x, y = pos
            dx, dy = x - ref_x, y - ref_y
            located_piece = translate(piece, dx, dy)
        else:
            done = False
            placed_polys = get_pieces_from_id(self.placed_pieces, self.located_pieces)
            # located_piece = place_pieces_into_gap_BL(piece, gap, placed_polys)
            # located_piece = NFP_place(piece, self.bin, placed_polys, _instance)
            # located_piece = svg_place(piece, self.bin, placed_polys)
            located_piece = self.nfp_cache_place_poly(piece_idx, rotation, piece, nfp_cache, scheme=scheme)  # Note
            # located_piece = self.nfp_cache_svg_place(piece_idx, rotation, piece, nfp_cache)

        # 到此零件位置确定，更新状态
        if piece_idx in self.no_placed_pieces:
            self.no_placed_pieces.remove(piece_idx)
        self.placed_pieces.append(piece_idx)
        # print(f'current piece_id: {piece_idx}, rotation: {rotation}')
        self.located_pieces[piece_idx] = located_piece  # 更新零件信息，因为加入的旋转和移动操作
        self.angle_list[piece_idx] = rotation  # 存储实际旋转角度
        self.gaps = (find_gaps(
            self.bin_width, self.bin_height, get_pieces_from_id(self.placed_pieces, self.located_pieces))
        )
        self.state = self.update_state()
        if len(self.no_placed_pieces) == 0:
            done = True
        prev_height = max([p.bounds[3] for p in self.located_pieces])

        reward = self.get_reward(
            _instance=_instance,
            piece_idx=piece_idx,
            step_id=step_id,
            done=done,
            prev_height=prev_height
        )

        sub_rewards = (
            self.get_selection_reward(done, step_id),
            self.get_angle_reward(piece_idx, done, step_id),
        )

        # print(f"Placed piece {piece_idx} with angle {angle}")
        return self.state, reward, sub_rewards, done, located_piece, True  # 返回的最后一个参数是是否成功放置

    def update_state(self):
        return get_vertices_array(get_vertices_list(get_pieces_from_id(self.placed_pieces, self.input_pieces)) +
                                  [[(self.bin_width, self.bin_height)]])  # 状态tensor（已放置零件+底板）

    def get_no_placed_pieces_state(self):
        return get_vertices_array(get_vertices_list(get_pieces_from_id(self.no_placed_pieces, self.input_pieces)))

    def get_piece_state(self, piece: Polygon):
        return get_vertices_array(get_vertices_list([piece]))

    def get_gap_state(self):
        return get_vertices_array(get_vertices_list(self.gaps))

    def get_highest_bound(self):
        max_y = -1
        for poly in self.placed_pieces:
            # 获取多边形的顶点坐标
            poly_vertices = self.input_pieces[poly].exterior.coords
            # 提取所有顶点的 y 坐标
            y_coord = [vertex[1] for vertex in poly_vertices]
            # 找到最高的 y 坐标
            top_y = max(y_coord)
            # 更新最高的 y 坐标
            if top_y > max_y:
                max_y = top_y
        return max_y

    def get_reward(self, _instance, piece_idx, step_id, done, prev_height):
        R = self.get_selection_reward(step_id=step_id, done=done) + \
            self.get_angle_reward(piece_idx=piece_idx, done=done, step_id=step_id) + \
            self.get_gap_reward(prev_height=prev_height, done=done, step_id=step_id)

        if done:
            return 1000 * get_utilization_from_list(_instance, self.located_pieces)
        else:
            return R * pow(0.9999, step_id) * 10

    def get_selection_reward(self, done, step_id):
        # 矩形度
        placed_polys = get_pieces_from_id(self.placed_pieces, self.located_pieces)
        whole_poly = unary_union(placed_polys)
        x_min, y_min, x_max, y_max = whole_poly.bounds

        if done:
            return sum([poly.area for poly in placed_polys]) / (x_max - x_min) / (y_max - y_min) * 1000
        else:
            return sum([poly.area for poly in placed_polys]) / (x_max - x_min) / (y_max - y_min) * \
                pow(0.9999, step_id) * 10

    # def get_angle_reward(self, piece_idx, done, step_id):
    #     # 零件下方利用率
    #     P = self.located_pieces[piece_idx]
    #     x_min, y_min, x_max, y_max = P.bounds
    #     placed_polys = get_pieces_from_id(self.placed_pieces, self.located_pieces)
    #     whole_poly = unary_union(placed_polys)
    #     judge_part = unary_union([Polygon([(x_min, 0), (x_max, 0), (x_max, y_min), (x_min, y_min)]), whole_poly])
    #     if done:
    #         return 1 - judge_part.area / (x_max - x_min) / y_min * 1000
    #     else:
    #         return 1 - judge_part.area / (x_max - x_min) / y_min * pow(0.9999, step_id) * 10

    # 改成邻接度
    def get_angle_reward(self, piece_idx, done, step_id):
        P = self.located_pieces[piece_idx]
        other_pieces = get_pieces_from_id(self.placed_pieces, self.located_pieces)
        adj_P = calculate_adjacent_length(P, other_pieces, self.bin_width, self.bin_height)
        if done:
            return adj_P / P.length * 1000
        else:
            return adj_P / P.length * pow(0.9999, step_id) * 10

    def get_gap_reward(self, prev_height, done, step_id):
        current_height = max([p.bounds[3] for p in self.located_pieces])
        if done:
            return prev_height / current_height * 1000
        else:
            return prev_height / current_height * pow(0.9999, step_id) * 10

    def update_gaps(self):
        pieces_in_bin = get_pieces_from_id(self.placed_pieces, self.located_pieces)
        # visualize_polygon_list(pieces_in_bin, self.bin_width, self.bin_height)  # gaps就是从pieces_in_bin中找出来的
        self.gaps = find_gaps(self.bin_width, self.bin_height, pieces_in_bin)

    def move_placed_pieces(self, _instance: Instance):
        """
        挪动未被选中的零件，贪心地挪动？
        思路一：将零件按从高到低排序，然后依次贪心地放置，即bottomleft，遍历角度，选择最低的位置放置。若放完后的高度大于等于原来的高度，则停止
        针对于placed的located_pieces
        """

        pieces_with_max_y = []
        for idx, poly in enumerate(self.input_pieces):
            if idx in self.placed_pieces:
                max_y = poly.bounds[3]  # 获取最大纵坐标
                pieces_with_max_y.append((idx, max_y))
        pieces_with_max_y.sort(key=lambda x: -x[1])  # 按最大纵坐标降序排列

        for idx, _ in pieces_with_max_y:
            current_poly = self.located_pieces[idx]
            self.update_gaps()

            gap = unary_union(self.gaps)  # 合并 gap
            original_height = current_poly.bounds[3]  # 原来的高度
            max_height = _instance.bin_height * 2  # 更新后，是多边形放置之后的最大高度
            best_located_piece = None
            best_angle = -1

            # 贪心地寻找放置位置和姿态
            for angle in _instance.rotations:
                rotated_poly = rotate(current_poly, angle, origin=current_poly.centroid.coords[0], use_radians=False)
                placed_polys = get_pieces_from_id(self.placed_pieces, self.located_pieces)

                # located_piece = place_pieces_into_gap_BL(rotated_poly, gap, placed_polys)
                located_piece = NFP_place(rotated_poly, self.bin, placed_polys, _instance)

                if located_piece is not None:
                    top_y = located_piece.bounds[3]
                    if top_y < max_height:
                        max_height = top_y
                        best_located_piece = located_piece

            # 找到位置后，判断高度是否有下降
            if max_height <= original_height:
                # 挪动是有效的，更新状态
                if best_located_piece is not None:
                    self.located_pieces[idx] = best_located_piece
                    # self.placed_pieces.append(idx)  # 本来就已经在placed_pieces中了
            else:
                # 挪动不会起作用了，退出挪动循环
                break

    def update_placed_pieces(self):
        pass

    def nfp_cache_place(self, moving_id, moving_angle, moving_piece, nfp_cache, scheme='BL'):
        polygon_nfp_cache = nfp_cache[0]
        inner_nfp_cache = nfp_cache[1]
        inner_nfp_key = get_inner_nfp_key(moving_id, moving_angle)
        try:
            inner_nfp_record = inner_nfp_cache[inner_nfp_key]
        except KeyError:
            raise KeyError(f"Inner NFP not found for key -> {inner_nfp_key}.")
        inner_nfp = standardize_polygon(inner_nfp_record).buffer(0.025)  # no need to calculate offset
        if inner_nfp.is_empty:
            raise ValueError("inner_nfp is empty.")
        polygon_nfp_list = []

        for static_id in self.placed_pieces:
            static_piece = self.located_pieces[static_id]
            static_angle = self.angle_list[static_id]
            polygon_nfp_key = get_polygon_nfp_key(static_id, static_angle, moving_id, moving_angle)
            try:
                polygon_nfp_record = polygon_nfp_cache[polygon_nfp_key]
            except KeyError:
                raise KeyError(f"Polygon NFP not found for key -> {polygon_nfp_key}.")
            polygon_nfp = standardize_polygon(polygon_nfp_record)

            placed_centroid = static_piece.centroid
            original_centroid = self.input_pieces[static_id].centroid
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
            located_point = get_MU_point(nfp, self.located_pieces, moving_piece)
        else:
            raise ValueError(f"Invalid scheme -> {scheme}.")

        ref_point = moving_piece.exterior.coords[0]
        translation_x = located_point[0] - ref_point[0]
        translation_y = located_point[1] - ref_point[1]
        moving_piece = translate(moving_piece, translation_x, translation_y)

        return moving_piece

    def nfp_cache_place_poly(self, moving_id_o, moving_angle, moving_piece, nfp_cache, scheme='BL'):
        polygon_nfp_cache = nfp_cache[0]
        inner_nfp_cache = nfp_cache[1]
        moving_id = moving_id_o % 15
        inner_nfp_key = get_inner_nfp_key(moving_id, moving_angle)
        try:
            inner_nfp_record = inner_nfp_cache[inner_nfp_key]
        except KeyError:
            raise KeyError(f"Inner NFP not found for key -> {inner_nfp_key}.")
        inner_nfp = standardize_polygon(inner_nfp_record).buffer(0.025)  # no need to calculate offset
        if inner_nfp.is_empty:
            raise ValueError("inner_nfp is empty.")
        polygon_nfp_list = []

        for static_id_o in self.placed_pieces:
            static_id = static_id_o % 15
            static_piece = self.located_pieces[static_id_o]
            static_angle = self.angle_list[static_id_o]
            polygon_nfp_key = get_polygon_nfp_key(static_id, static_angle, moving_id, moving_angle)
            try:
                polygon_nfp_record = polygon_nfp_cache[polygon_nfp_key]
            except KeyError:
                raise KeyError(f"Polygon NFP not found for key -> {polygon_nfp_key}.")
            polygon_nfp = standardize_polygon(polygon_nfp_record)

            placed_centroid = static_piece.centroid
            original_centroid = self.input_pieces[static_id_o].centroid
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
            located_point = get_MU_point(nfp, self.located_pieces, moving_piece)
        else:
            raise ValueError(f"Invalid scheme -> {scheme}.")

        ref_point = moving_piece.exterior.coords[0]
        translation_x = located_point[0] - ref_point[0]
        translation_y = located_point[1] - ref_point[1]
        moving_piece = translate(moving_piece, translation_x, translation_y)

        return moving_piece

    def nfp_cache_svg_place(self, moving_id, moving_angle, moving_piece, nfp_cache):
        polygon_nfp_cache = nfp_cache[0]
        inner_nfp_cache = nfp_cache[1]
        inner_nfp_key = get_inner_nfp_key(moving_id, moving_angle)
        try:
            inner_nfp_record = inner_nfp_cache[inner_nfp_key]
        except KeyError:
            raise KeyError(f"Inner NFP not found for key -> {inner_nfp_key}.")
        inner_nfp = standardize_polygon(inner_nfp_record)  # no need to calculate offset
        if inner_nfp.is_empty:
            raise ValueError("inner_nfp is empty.")
        polygon_nfp_list = []

        for static_id in self.placed_pieces:
            static_piece = self.located_pieces[static_id]
            static_angle = self.angle_list[static_id]
            polygon_nfp_key = get_polygon_nfp_key(static_id, static_angle, moving_id, moving_angle)
            try:
                polygon_nfp_record = polygon_nfp_cache[polygon_nfp_key]
            except KeyError:
                raise KeyError(f"Polygon NFP not found for key -> {polygon_nfp_key}.")
            polygon_nfp = standardize_polygon(polygon_nfp_record)

            placed_centroid = static_piece.centroid
            original_centroid = self.input_pieces[static_id].centroid
            offset_x = placed_centroid.x - original_centroid.x
            offset_y = placed_centroid.y - original_centroid.y
            polygon_nfp = translate(polygon_nfp, offset_x, offset_y)
            polygon_nfp_list.append(polygon_nfp)

        if polygon_nfp_list is not None:
            union_polygon_nfp = unary_union(polygon_nfp_list)
            # visualize_polygon(union_polygon_nfp)
            nfp = inner_nfp.difference(union_polygon_nfp)
        else:
            nfp = inner_nfp

        coord = []
        if isinstance(nfp, MultiPolygon):  # 检查是否是 MultiPolygon
            for poly in nfp.geoms:  # 遍历 MultiPolygon 的每个 Polygon
                coord.extend(list(poly.exterior.coords))
        else:
            coord = list(nfp.exterior.coords)  # 单一 Polygon 的情况
        candidate_coord = coord

        x_min, y_min, x_max, y_max = 1e9, 1e9, -1e9, -1e9
        for pp in self.located_pieces:
            x_min = min(x_min, get_min_x(pp))
            y_min = min(y_min, get_min_y(pp))
            x_max = max(x_max, get_max_x(pp))
            y_max = max(y_max, get_max_y(pp))

        min_score = 1e10
        located_point = None
        for p in candidate_coord:
            cur_x_min, cur_y_min, cur_x_max, cur_y_max = x_min, y_min, x_max, y_max
            cur_x_min = min(cur_x_min, get_min_x(moving_piece) + p[0])
            cur_y_min = min(cur_y_min, get_min_y(moving_piece) + p[1])
            cur_x_max = max(cur_x_max, get_max_x(moving_piece) + p[0])
            cur_y_max = max(cur_y_max, get_max_y(moving_piece) + p[1])

            width = cur_x_max - cur_x_min
            height = cur_y_max - cur_y_min

            fitness = 2 * height #+ width  # 乘2的是“重力方向”
            if fitness < min_score:
                min_score = fitness
                located_point = p
            elif fitness == min_score:
                if p[1] < located_point[1]:
                    located_point = p

        ref_point = moving_piece.exterior.coords[0]
        translation_x = located_point[0] - ref_point[0]
        translation_y = located_point[1] - ref_point[1]
        moving_piece = translate(moving_piece, translation_x, translation_y)

        return moving_piece


    def get_max_area(self):
        max_area = -1
        for p in self.located_pieces:
            if p.area > max_area:
                max_area = p.area
        return max_area

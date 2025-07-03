import heapq
import json
import time
import os
import re
import xml.etree.ElementTree as ET

import cv2
import networkx as nx
import torch
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from torch_geometric.data import Data

from entity.poly import *
from utils.geomertryutils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def standardize_ccw(nodes):
    polygon = Polygon(nodes)
    if polygon.exterior.is_ccw:
        return nodes
    else:
        return list(reversed(nodes))


def find_gaps(bin_width, bin_height, polygons: list[Polygon]):
    # 返回空隙的Polygon对象列表
    _bin = Polygon([(0, 0), (bin_width, 0), (bin_width, bin_height), (0, bin_height)])
    union_of_polygons = unary_union(polygons)
    gaps = _bin.difference(union_of_polygons)  # 差集
    res = []
    if isinstance(gaps, Polygon):  # 一个空隙
        res.append(gaps)
    elif isinstance(gaps, MultiPolygon):  # 多个空隙
        for gap in gaps.geoms:
            res.append(gap)

    clipped_gaps = []
    for gap in res:
        clipped_gaps.extend(clip_gap_polygons(gap))

    return clipped_gaps


# def find_gaps(bin_width, bin_height, polygons: list[Polygon]):
#     # 返回空隙的Polygon对象列表
#     _bin = Polygon([(0, 0), (bin_width, 0), (bin_width, bin_height), (0, bin_height)])
#     union_of_polygons = unary_union(polygons)
#     gaps = _bin.difference(union_of_polygons)  # 差集
#     res = []
#     if isinstance(gaps, Polygon):  # 一个空隙
#         res.append(gaps)
#     elif isinstance(gaps, MultiPolygon):  # 多个空隙
#         for gap in gaps.geoms:
#             res.append(gap)
#
#     return res  # 将所有空余区域看成一个空隙


def clip_gap_polygons(gap: Polygon, tolerance=0.05):
    clipped_gap = gap.buffer(tolerance)  # 正值扩展，负值收缩
    if isinstance(clipped_gap, MultiPolygon):
        separated_polygons = list(clipped_gap.geoms)  # 使用 geoms 属性拆分
    else:
        separated_polygons = [clipped_gap]

    return separated_polygons


def get_ref_point(piece_nodes):
    # 选出最靠左下方的点作为参考点（生成IFP）
    leftmost_point = min(piece_nodes, key=lambda point: (point[1], point[2]))
    return leftmost_point[0]


# 返回逆时针坐标列表
def get_bin_vertices(_path):
    _, _bin = parse_xml(_path)
    return _bin


# 返回多边形对象列表
def get_piece_vertices(_path):
    polygons, _ = parse_xml(_path)
    return polygons


def parse_xml(_path):
    tree = ET.parse(_path)
    root = tree.getroot()
    ns_content = str(root.tag.split('{')[1].split('}')[0])
    namespace = {'ns': ns_content}  # 解决了命名空间不一致的问题

    piece_info_0 = root.find("ns:problem/ns:lot", namespace)
    piece_info_1 = root.find("ns:polygons", namespace)

    if piece_info_0 is None or piece_info_1 is None:
        raise ValueError("Failed to find required nodes in the XML file.")

    cnt = []
    for piece in piece_info_0.findall("ns:piece", namespace):
        cnt.append(int(piece.get('quantity')))
    cnt = [-1] + cnt  # 底板占位

    polygons = []
    _bin = []
    piece_menu = []
    for piece in piece_info_1.findall("ns:polygon", namespace):
        idx = piece.attrib['id']
        if not idx.startswith('polygon'):
            continue
        piece_menu.append(piece)  # 将所有零件信息存入piece_menu列表中

    for i in range(len(piece_menu)):
        piece = piece_menu[i]
        vertices = []
        for segment in piece.find("ns:lines", namespace).findall("ns:segment", namespace):
            x0 = float(segment.attrib['x0'])
            y0 = float(segment.attrib['y0'])
            vertices.append((x0, y0))
        x_min = float(piece.find("ns:xMin", namespace).text)
        x_max = float(piece.find("ns:xMax", namespace).text)
        y_min = float(piece.find("ns:yMin", namespace).text)
        y_max = float(piece.find("ns:yMax", namespace).text)

        piece_cnt = cnt[i]
        if i > 0:
            while piece_cnt > 0:  # 同一个零件可能有多个实例
                polygons.append(Polygon(vertices))
                piece_cnt -= 1
        if i == 0:
            _bin = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

    return polygons, _bin  # 返回pieces对象列表，便于记录坐标和旋转角度


def get_snapshot(bin_vertices, piece_vertices):
    base_polygon = Polygon(bin_vertices)
    polygon = Polygon(piece_vertices)

    minx, miny, max_x, maxy = base_polygon.bounds

    width = int(max_x - minx)
    height = int(maxy - miny)
    grid = np.zeros((width, height), dtype=int)

    for i in range(width):
        for j in range(height):
            x = minx + i  # 列索引对应 x 坐标
            y = miny + j  # 行索引对应 y 坐标

            point = Point(x + 0.5, y + 0.5)  # 使用点的中心进行判断

            # 判断这个点是否在底板内，以及多边形是否包含这个点
            if base_polygon.covers(point) and polygon.covers(point):
                grid[i, j] = 1

    # return np.rot90(grid)
    return grid


def get_gap_coord(bin_raster, gap_vertices):
    gap_polygon = Polygon(gap_vertices)
    rows, cols = bin_raster.shape
    gap_coord = []

    for col in range(cols):
        for row in range(rows):
            x, y = col + 0.5, row + 0.5

            point = Point(x, y)
            if gap_polygon.covers(point):
                gap_coord.append((row, col))

    return gap_coord  # raster中坐标从0开始


def standardize_coord(coord: list[tuple[int, int]]):
    return sorted(coord, key=lambda x: (x[1], x[0]))


def find_position(gap, _bin, piece):
    px, py = piece.shape
    # 遍历hole中的所有点，尝试放置piece；这些点一定是没被占用的
    for point in gap:
        i, j = point
        if i + px > _bin.shape[0] or j + py > _bin.shape[1]:
            continue
        if np.all(_bin[i:i + px, j:j + py] + piece <= 1):
            return i, j
    return -1, -1


def get_piece_rasters(_path):
    polygons = get_piece_vertices(_path)
    piece_rasters = []
    for p in polygons:
        pr = p.create_tensor()
        piece_rasters.append(remove_zero(pr))  # 保证rastermap的长和宽就是零件的最大长和宽
    return piece_rasters


def remove_zero(matrix):
    # 找出非零行和列的索引
    non_zero_rows = np.any(matrix != 0, axis=1)
    non_zero_cols = np.any(matrix != 0, axis=0)

    # 根据索引筛选矩阵
    trimmed_matrix = matrix[non_zero_rows][:, non_zero_cols]
    return trimmed_matrix


def rotate_piece(piece: np.ndarray, angle):
    if angle == 0:
        return piece
    elif angle == 90:
        return np.rot90(piece, k=-1)
    elif angle == 180:
        return np.rot90(piece, k=2)
    elif angle == 270:
        return np.rot90(piece, k=1)


def get_poly_area(poly_vertices):
    poly = Polygon(poly_vertices)
    return poly.area


def read_results(filename, echo=False):
    """
    读取并打印存储的 episode 结果。
    """
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return

    with open(filename, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            print(f"File {filename} is not a valid JSON file.")
            return

    # 判断数据是列表还是单一字典
    # if isinstance(data, dict):  # 如果是单一字典，包裹为列表形式
    #     data = [data]

    if echo:
        print("\n--- Stored Episodes ---\n")
        for i, episode in enumerate(data):
            print(f"Episode {i + 1}:")
            print(f"  Utilization: {episode['utilization']}")
            print(f"  Placement Info:")
            for placement in episode.get("placement", []):
                print(
                    f"    - Piece ID: {placement['id']}, Position: {placement['position']},"
                    f"Angle: {placement['angle']}°"
                )
            if episode.get("is_max_utilization", False):
                print("  *** This is the max utilization episode ***")
            print()
    return data


def create_polygon(vertices):
    return Polygon(vertices)


def get_vertices_list(polygons: list[Polygon]):
    return [p.exterior.coords[:-1] for p in polygons]


def get_vertices_array(vertices_list: list[list[tuple[float, float]]]):
    res = []
    for vertices in vertices_list:
        for vertex in vertices:
            res.append(vertex[0])
            res.append(vertex[1])
    return np.array(res)


def get_pieces_from_id(idx_list, input_polygons):
    pieces = []
    for idx in idx_list:
        pieces.append(input_polygons[idx])
    return pieces


def adjust_position_BL(piece: Polygon, gap, placed_pieces: list[Polygon]):
    # 初始移动距离
    dx, dy = 0, 0
    print("adjusted BL")

    cur_piece = piece

    # 用于判断是否可以继续移动
    moved = True

    while moved:
        moved = False  # 假设这次没有任何移动

        # 尝试向下移动
        while True:
            cur_piece = translate(cur_piece, xoff=dx, yoff=-0.01)  # 向下移动
            if not can_placed_in_gap(cur_piece, gap, placed_pieces):
                cur_piece = translate(cur_piece, xoff=dx, yoff=0.01)
                break  # 如果放置失败，停止向下移动
            # dy -= 0.01
            moved = True  # 标记为已经移动过

        # 尝试向左移动
        while True:
            cur_piece = translate(cur_piece, xoff=-0.01, yoff=dy)  # 向左移动
            if not can_placed_in_gap(cur_piece, gap, placed_pieces):
                cur_piece = translate(cur_piece, xoff=0.01, yoff=dy)
                break  # 如果放置失败，停止向左移动
            # dx -= 0.01
            moved = True  # 标记为已经移动过

    return cur_piece


def adjust_position_BL_1(piece: Polygon, placed_pieces: list[Polygon]):
    # 初始移动距离
    dx, dy = 0, 0
    print("adjusted BL")

    cur_piece = piece

    # 用于判断是否可以继续移动
    moved = True

    while moved:
        moved = False  # 假设这次没有任何移动

        # 尝试向下移动
        while True:
            cur_piece = translate(cur_piece, xoff=dx, yoff=-0.01)  # 向下移动
            if not can_placed_in_coord(cur_piece, placed_pieces):
                cur_piece = translate(cur_piece, xoff=dx, yoff=0.01)
                break  # 如果放置失败，停止向下移动
            # dy -= 0.01
            moved = True  # 标记为已经移动过

        # 尝试向左移动
        while True:
            cur_piece = translate(cur_piece, xoff=-0.01, yoff=dy)  # 向左移动
            if not can_placed_in_coord(cur_piece, placed_pieces):
                cur_piece = translate(cur_piece, xoff=0.01, yoff=dy)
                break  # 如果放置失败，停止向左移动
            # dx -= 0.01
            moved = True  # 标记为已经移动过

    return cur_piece


def adjust_position_BR(piece: Polygon, gap: Polygon, placed_pieces: list[Polygon]):
    # 初始移动距离
    dx, dy = 0, 0
    print("adjusted BR")

    cur_piece = piece

    # 用于判断是否可以继续移动
    moved = True

    while moved:
        moved = False  # 假设这次没有任何移动

        # 尝试向下移动
        while True:
            cur_piece = translate(cur_piece, xoff=dx, yoff=-0.01)  # 向下移动
            if not can_placed_in_gap(cur_piece, gap, placed_pieces):
                cur_piece = translate(cur_piece, xoff=dx, yoff=0.01)
                break  # 如果放置失败，停止向下移动
            # dy -= 0.01
            moved = True  # 标记为已经移动过

        # 尝试向左移动
        while True:
            cur_piece = translate(cur_piece, xoff=0.01, yoff=dy)  # 向左移动
            if not can_placed_in_gap(cur_piece, gap, placed_pieces):
                cur_piece = translate(cur_piece, xoff=-0.01, yoff=dy)
                break  # 如果放置失败，停止向左移动
            # dx -= 0.01
            moved = True  # 标记为已经移动过

    return cur_piece


# NOTE: NFP
# TODO：将不正确的情况跳过
def NFP_place(piece: Polygon, _bin: Polygon, placed_pieces: list[Polygon], step_size=1, _rotations=None):
    inner_nfp = inner_fit_polygon_rectangle(_bin, piece)
    if inner_nfp is None:
        return None

    nfp_union = None
    if len(placed_pieces) == 0:
        nfp = inner_nfp
    else:
        nfp = no_fit_polygon(placed_pieces[0], piece)
        polygon_union = placed_pieces[0]
        for i in range(1, len(placed_pieces)):
            polygon_union = polygon_union.union(placed_pieces[i])

        union_start_time = time.time()
        nfp_union = no_fit_polygon(polygon_union, piece)
        union_end_time = time.time()
        print(f"union time: {union_end_time - union_start_time}")

        # if len(placed_pieces) > 1:
        #     for i in range(1, len(placed_pieces)):
        #         p = placed_pieces[i]
        #         n1 = no_fit_polygon(p, piece)
        #         if n1 is None:
        #             print(f'Error nfp with p1={p} and p2={piece}')
        #         nfp = nfp.union(n1)
        # nfp = inner_nfp.difference(nfp)
        # nfp_union = inner_nfp.difference(nfp_union)
        # if nfp_union is None:
        #     print(f'Error nfp during difference')
        #     return None

    # ref = get_BL_point(nfp)
    ref = get_BL_point(inner_nfp)
    coord_list = list(piece.exterior.coords)
    ref_p = coord_list[0]
    trans_x = ref[0] - ref_p[0]
    trans_y = ref[1] - ref_p[1]
    piece = translate(piece, xoff=trans_x, yoff=trans_y)
    return piece


def get_closest_half(vertices):
    """
    对输入的顶点列表，根据每个顶点到原点（0,0）的距离排序，并返回前50%的顶点。

    参数:
        vertices (list of tuple): 顶点坐标列表，如 [(x, y), ...]

    返回:
        list of tuple: 排序后距离原点最近的前50%顶点列表。如果顶点总数为奇数，则返回floor(n/2)个顶点。
    """
    # 按照距离原点的距离排序，这里直接用距离的平方排序，避免计算开方
    sorted_vertices = sorted(vertices, key=lambda pt: pt[0] ** 2 + pt[1] ** 2)

    # 计算需要返回的顶点数量：取前50%
    half_n = len(sorted_vertices) // 2  # 地板除，若需要向上取整可使用 math.ceil(len(sorted_vertices)/2)

    # 返回前半部分顶点列表
    return sorted_vertices[:half_n]


def svg_place(piece: Polygon, _bin: Polygon, placed_pieces: list[Polygon], step_size=1, _rotations=None):
    inner_nfp = inner_fit_polygon_rectangle(_bin, piece)
    if inner_nfp is None:
        return None

    if len(placed_pieces) == 0:
        nfp_final = inner_nfp
    else:
        union_piece = unary_union(placed_pieces)
        nfp = no_fit_polygon(union_piece, piece)
        nfp_final = inner_nfp.difference(nfp)
        if nfp_final is None:
            return None

    # 根据svg的思想使用适应度
    coord = []
    if isinstance(nfp_final, MultiPolygon):  # 检查是否是 MultiPolygon
        for poly in nfp_final.geoms:  # 遍历 MultiPolygon 的每个 Polygon
            coord.extend(list(poly.exterior.coords))
    else:
        coord = list(nfp_final.exterior.coords)  # 单一 Polygon 的情况

    # 遍历每个顶点，按照距离中心的距离排序，取前50%
    candidate_coord = get_closest_half(coord)
    pw = piece.bounds[2] - piece.bounds[0]
    ph = piece.bounds[3] - piece.bounds[1]
    min_score = 1e10
    best_coord = None
    for p in candidate_coord:
        width = p[0] + pw
        height = p[1] + ph
        fitness = 2 * height + width  # 乘2的是“重力方向”
        if fitness < min_score:
            min_score = fitness
            best_coord = p
        elif fitness == min_score:
            if p[1] < best_coord[1]:
                best_coord = p
    ref_p = piece.exterior.coords[0]
    trans_x = best_coord[0] - ref_p[0]
    trans_y = best_coord[1] - ref_p[1]
    piece = translate(piece, xoff=trans_x, yoff=trans_y)
    return piece


# 🌟 放置操作的关键
# 先实现的是BL
def place_pieces_into_gap_BL(piece: Polygon, gap: Polygon, placed_pieces: list[Polygon], step_size=1, _rotations=None):
    # 如果 piece 比 gap 大，直接返回 None
    if piece.area > gap.area:
        return None

    # 如果直接可以放置，直接返回
    if can_placed_in_gap(piece, gap, placed_pieces):
        print("can be placed directly")
        return adjust_position_BL(piece, gap, placed_pieces)

    # 默认旋转角度列表
    if _rotations is None:
        _rotations = [0]

    candidate_coord = get_grids_BL(gap)

    # 获取 gap 的边界范围
    g_min_x, g_min_y, g_max_x, g_max_y = gap.bounds

    # 遍历所有旋转角度
    for angle in _rotations:
        rotated_piece = rotate(piece, angle, origin='centroid')

        # 获取 piece 的边界范围
        p_min_x, p_min_y, _, _ = rotated_piece.bounds

        # 将 piece 平移到 gap 的左下角
        dx = g_min_x - p_min_x
        dy = g_min_y - p_min_y
        translated_piece = translate(rotated_piece, xoff=dx, yoff=dy)

        for coord in candidate_coord:
            xx = coord[0] - p_min_x
            yy = coord[1] - p_min_y
            # 尝试将 piece 放置到当前坐标
            test_piece = translate(translated_piece, xoff=xx, yoff=yy)

            # 如果当前放置可行，返回该放置的 piece
            if can_placed_in_gap(test_piece, gap, placed_pieces):
                print("can be placed in coord")
                return adjust_position_BL(test_piece, gap, placed_pieces)

    # 如果无法找到合适位置，返回 None
    return None


def place_pieces_into_gap_BL_1(piece: Polygon, gap: Polygon, placed_pieces: list[Polygon], step_size=1,
                               _rotations=None):
    """
    将参考点换为零件的质心，减小误差
    :param piece:
    :param gap:
    :param placed_pieces:
    :param step_size:
    :param _rotations:
    :return:
    """
    # 如果 piece 比 gap 大，直接返回 None
    if piece.area > gap.area:
        return None

    # 如果直接可以放置，直接返回
    if can_placed_in_gap(piece, gap, placed_pieces):
        print("can be placed directly")
        return adjust_position_BL(piece, gap, placed_pieces)

    # 默认旋转角度列表
    if _rotations is None:
        _rotations = [0]

    candidate_coord = get_grids_BL(gap)

    # 获取 gap 的边界范围
    g_min_x, g_min_y, g_max_x, g_max_y = gap.bounds

    # 遍历所有旋转角度
    for angle in _rotations:
        rotated_piece = rotate(piece, angle, origin='centroid')

        # # 获取 piece 的边界范围
        # p_min_x, p_min_y, _, _ = rotated_piece.bounds
        #
        # # 将 piece 平移到 gap 的左下角
        # dx = g_min_x - p_min_x
        # dy = g_min_y - p_min_y
        # translated_piece = translate(rotated_piece, xoff=dx, yoff=dy)

        ref_x, ref_y = rotated_piece.centroid.coords[0]

        for coord in candidate_coord:
            xx = coord[0] - ref_x
            yy = coord[1] - ref_y
            # 尝试将 piece 放置到当前坐标
            test_piece = translate(rotated_piece, xoff=xx, yoff=yy)

            # 如果当前放置可行，返回该放置的 piece
            if can_placed_in_gap(test_piece, gap, placed_pieces):
                print("can be placed in coord")
                return adjust_position_BL(test_piece, gap, placed_pieces)

    # 如果无法找到合适位置，返回 None
    return None


def place_pieces_into_gap_BL_2(piece: Polygon, gap: Polygon, placed_pieces: list[Polygon], x_range, _rotations=None):
    # 如果 piece 比 gap 大，直接返回 None
    if piece.area > gap.area:
        return None

    # 如果直接可以放置，直接返回
    if can_placed_in_gap(piece, gap, placed_pieces):
        print("can be placed directly")
        return adjust_position_BL(piece, gap, placed_pieces)

    # 默认旋转角度列表
    if _rotations is None:
        _rotations = [0]

    candidate_coord = get_grids_BL(gap)

    # 获取 gap 的边界范围
    g_min_x, g_min_y, g_max_x, g_max_y = gap.bounds

    # 遍历所有旋转角度
    for angle in _rotations:
        rotated_piece = rotate(piece, angle, origin='centroid')

        # 获取 piece 的边界范围
        p_min_x, p_min_y, _, _ = rotated_piece.bounds

        # 将 piece 平移到 gap 的左下角
        dx = g_min_x - p_min_x
        dy = g_min_y - p_min_y
        translated_piece = translate(rotated_piece, xoff=dx, yoff=dy)

        for coord in candidate_coord:

            if coord[0] < x_range[0] or coord[1] > x_range[1]:
                continue

            xx = coord[0] - p_min_x
            yy = coord[1] - p_min_y
            # 尝试将 piece 放置到当前坐标
            test_piece = translate(translated_piece, xoff=xx, yoff=yy)

            # 如果当前放置可行，返回该放置的 piece
            if can_placed_in_gap(test_piece, gap, placed_pieces):
                print("can be placed in coord")
                return adjust_position_BL(test_piece, gap, placed_pieces)

    # 如果无法找到合适位置，返回 None
    return None


def place_pieces_into_gap_BR(piece: Polygon, gap: Polygon, placed_pieces: list[Polygon], step_size=1, _rotations=None):
    # 如果 piece 比 gap 大，直接返回 None
    if piece.area > gap.area:
        return None

    # 如果直接可以放置，直接返回
    if can_placed_in_gap(piece, gap, placed_pieces):
        return adjust_position_BR(piece, gap, placed_pieces)

    # 默认旋转角度列表
    if _rotations is None:
        _rotations = [0]

    candidate_coord = get_grids_BR(gap)

    # 获取 gap 的边界范围
    g_min_x, g_min_y, g_max_x, g_max_y = gap.bounds

    # 遍历所有旋转角度
    for angle in _rotations:
        rotated_piece = rotate(piece, angle, origin='centroid')

        # 获取 piece 的边界范围
        p_min_x, p_min_y, p_max_x, p_max_y = rotated_piece.bounds

        dx = g_min_x - p_min_x
        dy = g_min_y - p_min_y
        translated_piece = translate(rotated_piece, xoff=dx, yoff=dy)

        for coord in candidate_coord:
            xx = coord[0] - p_min_x
            yy = coord[1] - p_min_y
            # 尝试将 piece 放置到当前坐标
            test_piece = translate(translated_piece, xoff=xx, yoff=yy)

            # 如果当前放置可行，返回该放置的 piece
            if can_placed_in_gap(test_piece, gap, placed_pieces):
                return adjust_position_BR(test_piece, gap, placed_pieces)

    # 如果无法找到合适位置，返回 None
    return None


def can_placed_in_gap(piece: Polygon, gap: Polygon, placed_polygons: list[Polygon], tolerance=0.05):
    if piece.area > gap.area:
        return False

    if not gap.buffer(tolerance).contains(piece):
        return False

    # if any(piece.intersects(p.buffer(-tolerance)) for p in placed_polygons):
    #     return False
    # TODO：用包络矩形来判断？剪枝
    pin_x_min, pin_y_min, pin_x_max, pin_y_max = piece.bounds
    for p in placed_polygons:
        p_x_min, p_y_min, p_x_max, p_y_max = p.bounds
        if pin_x_max <= p_x_min or pin_x_min >= p_x_max or pin_y_max <= p_y_min or pin_y_min >= p_y_max:
            # 两个矩形不相交，则零件肯定不重叠
            continue
        if piece.intersects(p.buffer(-tolerance)):
            # 两个矩形相交，则用多边形本身的形状判断是否重叠
            return False
    return True


def can_placed_in_gap_1(piece: Polygon, gaps: list[Polygon], tolerance=0.05):
    for gap in gaps:
        if piece.area > gap.area:
            continue

        if not gap.buffer(tolerance).contains(piece):
            continue
        return True  # 只用gap判断是否可以放置即可
    return False


def can_placed_in_gap_2(piece: Polygon, gap: Polygon, tolerance=0.05):
    if piece.area > gap.area:
        return False

    if not gap.buffer(tolerance).contains(piece):
        return False

    return True


def can_placed_in_coord(piece: Polygon, placed_polygons: list[Polygon], tolerance=0.05):
    pin_x_min, pin_y_min, pin_x_max, pin_y_max = piece.bounds
    for p in placed_polygons:
        p_x_min, p_y_min, p_x_max, p_y_max = p.bounds
        if pin_x_max <= p_x_min or pin_x_min >= p_x_max or pin_y_max <= p_y_min or pin_y_min >= p_y_max:
            # 两个矩形不相交，则零件肯定不重叠
            continue
        if piece.intersects(p.buffer(-tolerance)):
            # 两个矩形相交，则用多边形本身的形状判断是否重叠
            return False
    return True


def get_utilization_from_list(_instance: Instance, located_pieces: list[Polygon]):
    sum_area = sum([p.area for p in located_pieces])
    top_height = max([p.bounds[3] for p in located_pieces])
    return sum_area / (_instance.bin_width * top_height)


def extract_schema(placement_pool: list[dict]):
    pass


def visualize_polygon(poly: Polygon, color='blue', alpha=0.5):
    plt.figure(figsize=(6, 6))
    x, y = zip(*poly.exterior.coords[:-1])
    plt.fill(x, y, color=color, alpha=alpha)
    plt.show()


# def visualize_polygon_list(poly_list: list[Polygon], lim_x, lim_y):
#     plt.figure(figsize=(6, 6 * lim_y / lim_x))
#
#     for P in poly_list:
#         x, y = zip(*P.exterior.coords[:-1])
#         plt.fill(x, y, alpha=0.5)
#
#     # plt.axhline(y=0, color='blue', linewidth=1)
#     # plt.axvline(x=0, color='blue', linewidth=1)
#     # plt.axvline(x=lim_x, color='blue', linewidth=1)
#     # plt.xlabel('X')
#     # plt.ylabel('Y')
#     plt.xlim(0, lim_x)
#     plt.ylim(0, lim_y)
#     plt.axis('equal')
#     plt.grid(False)
#     plt.show()


def visualize_polygon_list(poly_list: list[Polygon], lim_x, lim_y):
    plt.figure(lim_x, lim_y)  # Scale to actual dimensions

    for P in poly_list:
        x, y = zip(*P.exterior.coords[:-1])
        plt.fill(x, y, alpha=0.5)

    plt.xlim(0, lim_x)
    plt.ylim(0, lim_y)
    plt.axis('equal')
    plt.axis('off')  # Turn off all axes and decorations
    plt.tight_layout()
    plt.show()


def visualize_episode_result(_placement, episode, lim_x, lim_y, top_y=None):
    plt.figure(figsize=(6, 6))

    for idx in range(len(_placement)):
        P = _placement[idx]['polygon']
        x, y = zip(*P.exterior.coords[:-1])
        plt.fill(x, y, alpha=0.5, label=f'Piece {idx}')

    plt.axhline(y=0, color='blue', linewidth=1)
    plt.axvline(x=0, color='blue', linewidth=1)
    plt.axvline(x=lim_x, color='blue', linewidth=1)
    if top_y is not None:
        plt.axhline(y=top_y, color='red', linewidth=1)

    plt.title(f'Episode {episode} Packing Result')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, lim_x)
    plt.ylim(0, lim_y)
    plt.axis('equal')
    plt.grid(False)
    plt.show()


def align_vertex_to_edge(polygons: list[Polygon], threshold=0.1):
    updated_polygons = []

    for i, polygon in enumerate(polygons):
        # 将当前 Polygon 转换为顶点列表
        vertices = list(polygon.exterior.coords)
        new_vertices = []

        for vertex in vertices:
            point = Point(vertex)  # 将顶点转换为 Point 对象
            closest_projection = None
            min_distance = float('inf')

            # 遍历所有其他 Polygon 的边
            for j, other_polygon in enumerate(polygons):
                if i == j:  # 跳过自身
                    continue

                # 获取另一个 Polygon 的边
                edges = list(LineString(other_polygon.exterior.coords).coords)
                for k in range(len(edges) - 1):
                    edge = LineString([edges[k], edges[k + 1]])
                    distance = point.distance(edge)

                    # 如果距离小于阈值并且是最近的
                    if distance < threshold and distance < min_distance:
                        min_distance = distance
                        closest_projection = edge.interpolate(edge.project(point))

            # 如果找到符合条件的投影点，更新顶点坐标
            if closest_projection:
                new_vertices.append((closest_projection.x, closest_projection.y))
            else:
                new_vertices.append(vertex)  # 保持原始坐标

        # 更新当前 Polygon
        updated_polygons.append(Polygon(new_vertices))

    return updated_polygons


def align_vertices_in_placement(current_placement, threshold=0.1):
    updated_placement = {}

    # 获取所有多边形的边集合，用于后续计算
    all_edges = []
    for idx, item in current_placement.items():
        polygon = item['polygon']
        edges = list(LineString(polygon.exterior.coords).coords)
        all_edges.append((idx, edges))  # 保存边的索引和对应的边集

    for idx, item in current_placement.items():
        polygon = item['polygon']
        vertices = list(polygon.exterior.coords)
        new_vertices = []

        for vertex in vertices:
            point = Point(vertex)  # 将顶点转换为 Point 对象
            closest_projection = None
            min_distance = float('inf')

            # 遍历所有其他零件的边
            for other_idx, edges in all_edges:
                if other_idx == idx:  # 跳过自身的边
                    continue

                for k in range(len(edges) - 1):
                    edge = LineString([edges[k], edges[k + 1]])
                    distance = point.distance(edge)

                    # 如果距离小于阈值并且是最近的
                    if distance < threshold and distance < min_distance:
                        min_distance = distance
                        closest_projection = edge.interpolate(edge.project(point))

            # 如果找到符合条件的投影点，更新顶点坐标
            if closest_projection:
                new_vertices.append((closest_projection.x, closest_projection.y))
            else:
                new_vertices.append(vertex)  # 保持原始坐标

        # 生成新的多边形并更新字典
        updated_placement[idx] = {
            'polygon': Polygon(new_vertices),
            'angle': item['angle']
        }

    return updated_placement


def align_vertices_in_list(located_pieces: list[Polygon], threshold=0.1):
    updated_piece_list = located_pieces.copy()

    # 获取所有多边形的边集合，用于后续计算
    all_edges = []
    for idx, polygon in enumerate(located_pieces):
        edges = list(LineString(polygon.exterior.coords).coords)
        all_edges.append((idx, edges))  # 保存边的索引和对应的边集

    for idx, polygon in enumerate(located_pieces):
        vertices = list(polygon.exterior.coords)
        new_vertices = []

        for vertex in vertices:
            point = Point(vertex)  # 将顶点转换为 Point 对象
            closest_projection = None
            min_distance = float('inf')

            # 遍历所有其他零件的边
            for other_idx, edges in all_edges:
                if other_idx == idx:  # 跳过自身的边
                    continue

                for k in range(len(edges) - 1):
                    edge = LineString([edges[k], edges[k + 1]])
                    distance = point.distance(edge)

                    # 如果距离小于阈值并且是最近的
                    if distance < threshold and distance < min_distance:
                        min_distance = distance
                        closest_projection = edge.interpolate(edge.project(point))

            # 如果找到符合条件的投影点，更新顶点坐标
            if closest_projection:
                new_vertices.append((closest_projection.x, closest_projection.y))
            else:
                new_vertices.append(vertex)  # 保持原始坐标

        updated_piece_list[idx] = Polygon(new_vertices)

    return updated_piece_list


def get_grids_BL(gap: Polygon):  # 找到 gap 内部的整数点
    polygon_points = None
    if isinstance(gap, Polygon):
        polygon_points = np.array(gap.exterior.coords)
    elif isinstance(gap, MultiPolygon):
        polygon_points = np.concatenate([np.array(p.exterior.coords) for p in gap.geoms])

    polygon_points = polygon_points.astype(np.int32)  # 注意类型必须是整数

    # 确定多边形的边界矩形
    x_min = np.min(polygon_points[:, 0])
    x_max = np.max(polygon_points[:, 0])
    y_min = np.min(polygon_points[:, 1])
    y_max = np.max(polygon_points[:, 1])

    # 创建一个空的掩码
    mask = np.zeros((int(y_max - y_min) + 1, int(x_max - x_min) + 1), dtype=np.uint8)

    # 将多边形绘制在掩码上
    cv2.fillPoly(mask, [polygon_points - [x_min, y_min]], 1)

    # 找到掩码中值为1的点（即多边形内部的点）
    indices = np.argwhere(mask == 1)

    # 将掩码坐标映射回原始图像坐标
    inside_points = indices + [y_min, x_min]

    # 交换横坐标与纵坐标
    swapped_points = [(point[1], point[0]) for point in inside_points]

    # 按照交换后的坐标排序
    return sorted(swapped_points, key=lambda x: (x[1], x[0]))


def get_grids_BR(gap: Polygon):  # 靠右靠下优先
    polygon_points = None
    if isinstance(gap, Polygon):
        polygon_points = np.array(gap.exterior.coords)
    elif isinstance(gap, MultiPolygon):
        polygon_points = np.concatenate([np.array(p.exterior.coords) for p in gap.geoms])

    polygon_points = polygon_points.astype(np.int32)  # 注意类型必须是整数
    # 确定多边形的边界矩形
    x_min = np.min(polygon_points[:, 0])
    x_max = np.max(polygon_points[:, 0])
    y_min = np.min(polygon_points[:, 1])
    y_max = np.max(polygon_points[:, 1])

    # 创建一个空的掩码
    mask = np.zeros((int(y_max - y_min) + 1, int(x_max - x_min) + 1), dtype=np.uint8)

    # 将多边形绘制在掩码上
    cv2.fillPoly(mask, [polygon_points - [x_min, y_min]], 1)

    # 找到掩码中值为1的点（即多边形内部的点）
    indices = np.argwhere(mask == 1)

    # 将掩码坐标映射回原始图像坐标
    inside_points = indices + [y_min, x_min]

    # 交换横坐标与纵坐标
    swapped_points = [(point[1], point[0]) for point in inside_points]

    return sorted(swapped_points, key=lambda c: (-c[1], c[0]), reverse=True)


def calculate_adjacent_length(target_poly: Polygon, other_polys: list[Polygon],
                              bin_width, bin_height, blur_distance=0.05):
    target = target_poly
    the_bin = Polygon([[0, 0], [bin_width, 0], [bin_width, bin_height], [0, bin_height]])
    the_shell = Polygon([[-0.2, -0.2], [bin_width + 0.2, -0.2],
                         [bin_width + 0.2, bin_height + 0.2], [-0.2, bin_height + 0.2]])
    bin_poly = the_shell.difference(the_bin)

    # 将其他多边形和边框组合在一起
    all_polygons = [Polygon(p) for p in other_polys] + [bin_poly]

    # 模糊探测：扩展目标多边形边界
    target_buffered = target.buffer(blur_distance)

    adj_length = 0.0

    for poly in all_polygons:
        if not poly.is_valid:
            continue

        # 计算与目标多边形的相交部分
        intersection = target_buffered.intersection(poly)
        if not intersection.is_empty:
            # 如果交集是线段或多边形，提取边界长度
            if isinstance(intersection, (LineString, Polygon)):
                adj_length += ((intersection.length - blur_distance) / 2)

    return adj_length


def polygon_to_graph(polygon):
    coord = list(polygon.exterior.coords)
    centroid = polygon.centroid

    G = nx.Graph()
    for i, coord in enumerate(coord):
        G.add_node(i, x=coord[0], y=coord[1])

    # 添加边
    for i in range(len(coord) - 1):
        G.add_edge(i, i + 1)
    G.add_edge(len(coord) - 1, 0)

    # 添加质心节点
    centroid_index = len(coord)
    G.add_node(centroid_index, x=centroid.x, y=centroid.y)

    # 将每个顶点与质心连接
    for i in range(len(coord)):
        G.add_edge(i, centroid_index)

    return G


def combine_graphs(polygon_list, boundary=None):
    """
    将所有多边形和边框组合成一个大图。
    """
    combined_graph = nx.Graph()

    for polygon in polygon_list:
        sub_graph = polygon_to_graph(polygon)
        combined_graph = nx.disjoint_union(combined_graph, sub_graph)

    if boundary is not None:  # 若要加入边界则单独处理
        boundary_graph = polygon_to_graph(boundary)
        combined_graph = nx.disjoint_union(combined_graph, boundary_graph)

    return combined_graph


# def nx_to_data(graph):
#     x = torch.tensor([[graph.nodes[i]['x'], graph.nodes[i]['y']] for i in graph.nodes], dtype=torch.float)
#     edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
#     return Data(x=x, edge_index=edge_index)


# def nx_to_gap_data(graph):
#     # 获取节点的 x, y 坐标
#     coordinates = [[graph.nodes[i]['x'], graph.nodes[i]['y']] for i in graph.nodes]
#     x = torch.tensor(coordinates, dtype=torch.float)
#
#     area = 0
#
#     # 获取包络矩形左下角顶点的坐标
#     min_x = min(graph.nodes[i]['x'] for i in graph.nodes)
#     min_y = min(graph.nodes[i]['y'] for i in graph.nodes)
#
#     # 计算每个顶点到包络矩形左下角顶点的距离
#     distances = []
#     for i in graph.nodes:
#         dx = graph.nodes[i]['x'] - min_x
#         dy = graph.nodes[i]['y'] - min_y
#         distances.append((dx**2 + dy**2)**0.5)
#     distances = torch.tensor(distances, dtype=torch.float)
#
#     # 将面积扩展到与节点数量一致
#     areas = torch.full((len(graph.nodes), 1), area, dtype=torch.float)
#
#     # 将所有信息拼接到 x 中
#     x = torch.cat([x, areas, distances.unsqueeze(1)], dim=1)  # x 形状为 (num_nodes, 4)
#
#     # 构造边信息 edge_index
#     edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
#
#     # 返回图数据
#     return Data(x=x, edge_index=edge_index)


def nx_to_data_0(graph):
    x = torch.tensor([[graph.nodes[i]['x'], graph.nodes[i]['y']] for i in graph.nodes], dtype=torch.float)
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


# 用于生成local search中的初始解
def get_init_placement(results_filename):
    with open(results_filename, 'r') as f:
        data = json.load(f)
    pieces_info = data["pieces_info"]
    utilization = data["utilization"]
    return [Polygon(pieces_info[i]) for i in pieces_info], utilization


def get_matched_pieces(matched_path):
    with open(matched_path, 'r') as file:
        content = file.read()  # 读取整个文件内容
        # 定位每个Piece
        piece_pattern = re.compile(r"Piece\s+(\d+):\n([\d&]+)\n((?:\(-?\d+,-?\d+\)\n?)+)")
        pieces_dependencies = []
        # 提取信息
        pieces = []
        for match in piece_pattern.finditer(content):
            piece_number = int(match.group(1))  # 提取Piece序号
            piece_data_raw = match.group(2)  # 提取piece信息字符串
            piece_data = list(map(int, piece_data_raw.split("&")))  # 提取piece_relative
            vertices_raw = match.group(3)  # 提取所有顶点的字符串
            # 将顶点字符串解析为列表
            vertices = [
                tuple(map(int, coord.strip("()").split(",")))
                for coord in vertices_raw.strip().split("\n")
            ]
            piece = Polygon(vertices)  # 构建Polygon对象
            pieces.append(piece)
            pieces_dependencies.append(piece_data)  # 存储着第i个合并后的piece的依赖，若没被合并则为空
        return pieces, pieces_dependencies


def get_env_pieces(_instance):
    matched_path = _instance.training_config["matched_path"]
    matched_pieces, piece_dependencies = get_matched_pieces(matched_path)
    init_pieces = _instance.polygons
    all_pieces = init_pieces.copy()
    add_sum = 0
    piece_relatives = [[] for _ in range(len(init_pieces))]

    # 扩充零件集合，将合并后的零件加入集合中
    for i in range(len(matched_pieces)):
        if len(piece_dependencies[i]) > 1:
            all_pieces.append(matched_pieces[i])
            piece_relatives.append([])
            add_sum += 1
            for j in piece_dependencies[i]:
                piece_relatives[j].append(len(init_pieces) + add_sum - 1)
                piece_relatives[len(init_pieces) + add_sum - 1].append(j)

    # print(piece_dependencies)
    return piece_relatives, all_pieces


def load_trained_model(path_prefix, models):
    """
    加载训练好的模型
    """
    models['piece_selector'].load_state_dict(torch.load(f"{path_prefix}_piece_selector.pt", map_location=device))
    models['angle_selector'].load_state_dict(torch.load(f"{path_prefix}_angle_selector.pt", map_location=device))
    models['critic'].load_state_dict(torch.load(f"{path_prefix}_critic.pt", map_location=device))
    models['gnn_model'].load_state_dict(torch.load(f"{path_prefix}_gnn_model.pt", map_location=device))
    print(f"Trained models loaded from prefix {path_prefix}")


def get_placement_from_actions(act_seq_path, env):  # 环境要新建一个
    """
    输入一个动作序列文件，保存每个episode的placement信息
    :param act_seq_info:
    :param env:
    :return:
    """
    all_episode_placement = {}

    # all_seq_files = [os.path.join(act_seq_dir, f) for f in os.listdir(act_seq_dir) if
    #                  os.path.isfile(os.path.join(act_seq_dir, f))]

    with open(act_seq_path, 'r') as f:
        all_episodes_seq = json.load(f)

    for k in range(len(all_episodes_seq)):
        episode_seq = all_episodes_seq[k]
        action_sequence = episode_seq[1]
        utilization = episode_seq[0]
        env.reset()  # 重置环境
        angle = -1
        for action in action_sequence:
            env.update_gaps()
            piece_idx = action[0]
            angle = action[1]
            piece = rotate(env.input_polygons[piece_idx], angle * 90)  # Polygon旋转函数的angle是角度制
            gap = unary_union(env.gaps)  # 将所有gap合并成一个
            placed_polys = get_pieces_from_id(env.placed_pieces, env.located_pieces)

            located_piece = place_pieces_into_gap_BL(piece, gap, placed_polys)

            # 到此零件位置确定，更新状态
            env.no_placed_pieces.remove(piece_idx)
            env.placed_pieces.append(piece_idx)
            env.located_pieces[piece_idx] = located_piece  # 更新零件信息，因为加入的旋转和移动操作
            env.gaps = (find_gaps(
                env.bin_width, env.bin_height, get_pieces_from_id(env.placed_pieces, env.located_pieces))
            )
            env.state = env.update_state()
        # 全部零件已经放置完成，根据located_pieces信息生成placement
        P = {}  # 当前轮次的placement信息
        for i in range(len(env.located_pieces)):
            P[i] = {
                'id': i,
                'coord': list(env.located_pieces[i].exterior.coords),
                'centroid': env.located_pieces[i].centroid.coords[0],
                'angle': angle
            }
        all_episode_placement[k] = {
            'placement': P,
            'utilization': utilization
        }
        print(f"Episode {k} placement generated")

    # TODO：将所有episode的placement保存到一个目录下
    return all_episode_placement


def save_all_placement(all_episode_placement, results_path, instance_name):
    with open(results_path, 'w') as f:
        json.dump(all_episode_placement, f, indent=4)
    print(f"All placement of {instance_name} saved to {results_path}")


def get_new_piece_pos(original_piece, centroid_pos, angle):
    """
    将原始零件旋转后放置到新的坐标位置，以质心坐标为基准
    :param original_piece: Shapely Polygon 对象，表示原始零件形状
    :param new_coord_x: 平移后质心的x坐标
    :param new_coord_y: 平移后质心的y坐标
    :param angle: 旋转角度，单位为度（正值为逆时针方向）
    :return: 放置后的零件（Polygon对象）及包络矩形的左下角坐标
    """
    # 获取原始质心坐标
    original_centroid = original_piece.centroid.coords[0]  # (x, y)

    new_coord_x, new_coord_y = centroid_pos

    # 计算平移向量
    translation_vector = (new_coord_x - original_centroid[0],
                          new_coord_y - original_centroid[1])

    # 将多边形平移到新位置
    translated_piece = translate(original_piece,
                                 xoff=translation_vector[0],
                                 yoff=translation_vector[1])

    # 以新质心为基准旋转多边形
    rotated_piece = rotate(translated_piece, angle, origin='centroid')
    return rotated_piece


def get_diversity_from_placement(all_episode_placement):
    """
    遍历所有episode的placement，将放置位置完全相同的episode去重，返回去重后的episode数量
    :param all_episode_placement:
    :return:
    """
    count = 0
    for i in range(len(all_episode_placement)):
        this_episode_placement = all_episode_placement[i]['placement']
        this_episode_utilization = all_episode_placement[i]['utilization']


def get_min_x(p):
    """
    输入：p - shapely.geometry.Polygon 对象
    输出：多边形的最小横坐标（x最小值）
    """
    return p.bounds[0]


def get_min_y(p):
    """
    输入：p - shapely.geometry.Polygon 对象
    输出：多边形的最小纵坐标（y最小值）
    """
    return p.bounds[1]


def get_max_x(p):
    """
    输入：p - shapely.geometry.Polygon 对象
    输出：多边形的最大横坐标（x最大值）
    """
    return p.bounds[2]


def get_max_y(p):
    """
    输入：p - shapely.geometry.Polygon 对象
    输出：多边形的最大纵坐标（y最大值）
    """
    return p.bounds[3]


# NOTE：用于pattern和tabu
class TopBottomTracker:
    def __init__(self, k=20):
        self.k = k
        # 最高 k：min‑heap 存 (utilization, counter, record)
        self._top = []
        # 最低 k：同理存 (-utilization, counter, record)
        self._bottom = []
        self._idx = 0

    def add(self, record):
        u = record['utilization']
        idx = self._idx
        self._idx += 1

        # —— 维护“最高 k” —— #
        heapq.heappush(self._top, (u, idx, record))
        if len(self._top) > self.k:
            heapq.heappop(self._top)

        # —— 维护“最低 k” —— #
        heapq.heappush(self._bottom, (-u, idx, record))
        if len(self._bottom) > self.k:
            heapq.heappop(self._bottom)

    def top_k(self):
        # 堆中是 (util, idx, record)，reverse=True 让 util 最大的在前
        return [rec for _, _, rec in sorted(self._top, reverse=True)]

    def bottom_k(self):
        # 堆中是 (-util, idx, record)，sorted(reverse=True) 先按 -util 降序 => util 升序
        lst = sorted(self._bottom, reverse=True)
        return [rec for _, _, rec in lst]

    def is_empty(self):
        return len(self._top) < self.k or len(self._bottom) < self.k


def jsonify_bin(W, H):
    """Reformat a bin polygon to a JSON-serializable format."""
    return [
        {'x': 0, 'y': 0},
        {'x': W, 'y': 0},
        {'x': W, 'y': H},
        {'x': 0, 'y': H},
    ]


def jsonify_pieces(input_polygons):
    """Reformat input pieces sequence to a JSON-serializable format."""
    paths = []
    for p in input_polygons:
        coord = list(p.exterior.coords)
        vertices = []
        for c in coord:
            vertices.append({'x': c[0], 'y': c[1]})
        paths.append(vertices)
    return paths


def standardize_located(result):
    """Extract located pieces from SVG placement result."""
    pass


def svg_place_direct(bin_w, bin_h, input_polygons, place_order, angle_list):
    b = jsonify_bin(bin_w, bin_h)
    pass


def get_polygon_nfp_key(id_static, rotation_static, id_moving, rotation_moving):
    return '{' + f"id_static: {id_static}, id_moving: {id_moving}, rotation_static: {rotation_static}, rotation_moving: {rotation_moving}" + '}'


def get_inner_nfp_key(id_inner, rotation_inner):
    return '{' + f"id_inner: {id_inner}, rotation_inner: {rotation_inner}" + '}'


def load_nfp_cache(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # store two category of nfp as dicts
    polygon_nfp = data.get('polygon_nfp', {})
    inner_nfp = data.get('inner_nfp', {})

    return polygon_nfp, inner_nfp


def get_utilization(bin_width, located_pieces: list[Polygon]):
    sum_area = 0
    for p in located_pieces:
        sum_area += p.area
    top_height = max([p.bounds[3] for p in located_pieces])
    return sum_area / (bin_width * top_height)


def get_located_pieces(path):
    """Input a placement json file, return a list of located pieces."""
    pass

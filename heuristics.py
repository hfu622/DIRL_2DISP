from entity.config_all import CONFIG_DICT_ALL
from tools import *


def get_place_order(input_pieces):
    """
    将所有 Polygon 按面积降序排列，返回排序后的索引列表。

    参数:
    - input_pieces (list): 包含 Polygon 对象的列表。

    返回:
    - list: 原列表中元素按面积降序排列后的索引列表。
    """
    if not all(isinstance(piece, Polygon) for piece in input_pieces):
        raise ValueError("All elements in input_pieces must be Polygon objects.")

    # 返回按面积降序排列的索引
    sorted_indices = sorted(
        range(len(input_pieces)),
        key=lambda i: input_pieces[i].area,
        reverse=True
    )
    return sorted_indices


def get_centroid_order(input_pieces):
    """
    按照“质心纵坐标减去多边形包络矩形底边纵坐标”递减顺序排列多边形，返回排序后的索引列表。

    参数:
    - input_pieces (list[Polygon]): 一个包含 Polygon 对象的列表。

    返回:
    - list[int]: 原列表中元素按关键值递减排序后的索引列表。
    """
    if not all(isinstance(piece, Polygon) for piece in input_pieces):
        raise ValueError("All elements in input_pieces must be Polygon objects.")

    def get_key(i):
        poly = input_pieces[i]
        centroid_y = poly.centroid.y
        min_y = poly.bounds[1]
        return centroid_y - min_y

    sorted_indices = sorted(
        range(len(input_pieces)),
        key=get_key,
        reverse=True
    )
    return sorted_indices


def get_area_ratio_order(input_pieces):
    """
    按照“多边形面积除以包络矩形面积”递减顺序排列多边形，返回排序后的索引列表。

    参数:
    - input_pieces (list[Polygon]): 一个包含 Polygon 对象的列表。

    返回:
    - list[int]: 原列表中元素按面积比递减排序后的索引列表。
    """
    if not all(isinstance(piece, Polygon) for piece in input_pieces):
        raise ValueError("All elements in input_pieces must be Polygon objects.")

    def get_key(i):
        poly = input_pieces[i]
        area = poly.area
        min_x, min_y, max_x, max_y = poly.bounds
        envelope_area = (max_x - min_x) * (max_y - min_y)
        return area / envelope_area if envelope_area > 0 else 0

    sorted_indices = sorted(
        range(len(input_pieces)),
        key=get_key,
        reverse=True
    )
    return sorted_indices


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


def get_HeuA_utilization(input_polygons, order, instance_name, rotations, _width):
    located_pieces = input_polygons.copy()
    placed_pieces = []
    angle_list = {}
    nfp_cache_path = f"nfp/{instance_name}.json"
    nfp_cache = load_nfp_cache(nfp_cache_path)
    polygon_nfp_cache = nfp_cache[0]
    inner_nfp_cache = nfp_cache[1]

    heu_record = {}
    heu_placement = {}

    for i in range(len(input_polygons)):
        print(order)
        print(i)
        piece_id = order[i]
        selected_angle = -1
        min_y = 1e6
        final_piece = None

        for angle in rotations:
            current_piece = rotate(input_polygons[piece_id], angle, origin="centroid")

            inner_nfp_key = get_inner_nfp_key(piece_id, angle)
            try:
                inner_nfp_record = inner_nfp_cache[inner_nfp_key]
            except KeyError:
                raise KeyError(f"Inner NFP not found for key -> {inner_nfp_key}.")
            inner_nfp = standardize_polygon(inner_nfp_record).buffer(0.025)  # no need to calculate offset
            if inner_nfp.is_empty:
                raise ValueError("inner_nfp is empty.")
            polygon_nfp_list = []

            for static_id in placed_pieces:
                static_piece = located_pieces[static_id]
                static_angle = angle_list[static_id]
                polygon_nfp_key = get_polygon_nfp_key(static_id, static_angle, piece_id, angle)
                try:
                    polygon_nfp_record = polygon_nfp_cache[polygon_nfp_key]
                except KeyError:
                    raise KeyError(f"Polygon NFP not found for key -> {polygon_nfp_key}.")
                polygon_nfp = standardize_polygon(polygon_nfp_record)

                placed_centroid = static_piece.centroid
                original_centroid = input_polygons[static_id].centroid
                offset_x = placed_centroid.x - original_centroid.x
                offset_y = placed_centroid.y - original_centroid.y
                polygon_nfp = translate(polygon_nfp, offset_x, offset_y)
                polygon_nfp_list.append(polygon_nfp)

            if polygon_nfp_list is not None:
                union_polygon_nfp = unary_union(polygon_nfp_list).buffer(-0.025)
                # union_polygon_nfp = unary_union(polygon_nfp_list)
                nfp = inner_nfp.difference(union_polygon_nfp)
            else:
                nfp = inner_nfp

            located_point = get_BL_point(nfp)
            ref_point = current_piece.exterior.coords[0]
            translation_x = located_point[0] - ref_point[0]
            translation_y = located_point[1] - ref_point[1]
            moving_piece = translate(current_piece, translation_x, translation_y)

            # NOTE: bounds -> minx. miny, maxx, maxy
            if moving_piece.bounds[-1] < min_y:
                min_y = moving_piece.bounds[-1]
                selected_angle = angle
                final_piece = moving_piece

            # visualize_polygon(nfp)

        located_pieces[piece_id] = final_piece
        placed_pieces.append(piece_id)
        angle_list[piece_id] = selected_angle

        heu_placement[piece_id] = {
            "id": piece_id,
            "angle": selected_angle,
            "coord": list(final_piece.exterior.coords),
            "centroid": final_piece.centroid.coords[0],
        }
    heu_record["placement"] = heu_placement
    heu_record["utilization"] = get_utilization(_width, located_pieces)

    return heu_record, angle_list

# if __name__ == '__main__':
#     name = 'dighe2'
#     work = CONFIG_DICT_ALL[name]
#     rotations = work['rotations']
#     print(name)
#     instance_name = name
#     instance_path = f"Nesting/{instance_name}.xml"
#     input_polygons, _bin = parse_xml(instance_path)
#     bin_width = _bin[2][1] - _bin[1][1] + 0.05
#     bin_height = _bin[1][0] - _bin[0][0]
#     bin_polygon = Polygon([(0, 0), (bin_width, 0), (bin_width, bin_height), (0, bin_height)])
#
#     order_A = [3, 0, 1, 2, 4, 6, 5, 7, 8, 9]
#     print(get_HeuA_utilization(input_polygons, order_A, instance_name, rotations, bin_width))

if __name__ == '__main__':
    for name in CONFIG_DICT_ALL:
        work = CONFIG_DICT_ALL[name]
        rotations = work['rotations']
        print(name)
        instance_name = name
        instance_path = f"Nesting/{instance_name}.xml"
        input_pieces, _bin = parse_xml(instance_path)
        bin_width = _bin[2][1] - _bin[1][1] + 0.05
        bin_height = _bin[1][0] - _bin[0][0]
        bin_polygon = Polygon([(0, 0), (bin_width, 0), (bin_width, bin_height), (0, bin_height)])

        # NOTE：输入的是副本，确保原始零件不变
        order_A = get_place_order(input_pieces.copy())
        # order_A = [1, 18, 26, 5, 4, 2, 25, 11, 29, 23, 6, 28, 13, 15, 19, 21, 20, 0, 24, 12, 9, 8, 14, 10, 17, 22, 3, 27, 7, 16]

        order_H = get_centroid_order(input_pieces.copy())
        order_U = get_area_ratio_order(input_pieces.copy())

        order_list = [order_A, order_H, order_U]
        path_list = ["A", "H", "U"]

        nfp_cache_path = f"nfp/{instance_name}.json"
        nfp_cache = load_nfp_cache(nfp_cache_path)
        polygon_nfp_cache = nfp_cache[0]
        inner_nfp_cache = nfp_cache[1]

        for k in range(2):
            order = order_list[k]
            result_path = f"heu_records_angle/{path_list[k]}/{instance_name}.json"

            located_pieces = input_pieces.copy()
            placed_pieces = []
            angle_list = {}

            for i in range(len(input_pieces)):
                piece_id = order[i]
                selected_angle = -1
                min_y = 1e6
                final_piece = None

                for angle in rotations:
                    current_piece = rotate(input_pieces[piece_id], angle, origin="centroid")

                    inner_nfp_key = get_inner_nfp_key(piece_id, angle)
                    try:
                        inner_nfp_record = inner_nfp_cache[inner_nfp_key]
                    except KeyError:
                        raise KeyError(f"Inner NFP not found for key -> {inner_nfp_key}.")
                    inner_nfp = standardize_polygon(inner_nfp_record).buffer(0.025)  # no need to calculate offset
                    if inner_nfp.is_empty:
                        raise ValueError("inner_nfp is empty.")
                    polygon_nfp_list = []

                    for static_id in placed_pieces:
                        static_piece = located_pieces[static_id]
                        static_angle = angle_list[static_id]
                        polygon_nfp_key = get_polygon_nfp_key(static_id, static_angle, piece_id, angle)
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
                        # union_polygon_nfp = unary_union(polygon_nfp_list).buffer(-0.025)
                        union_polygon_nfp = unary_union(polygon_nfp_list)
                        nfp = inner_nfp.difference(union_polygon_nfp)
                    else:
                        nfp = inner_nfp

                    located_point = get_BL_point(nfp)
                    ref_point = current_piece.exterior.coords[0]
                    translation_x = located_point[0] - ref_point[0]
                    translation_y = located_point[1] - ref_point[1]
                    moving_piece = translate(current_piece, translation_x, translation_y)

                    # NOTE: bounds -> minx. miny, maxx, maxy
                    if moving_piece.bounds[-1] < min_y:
                        min_y = moving_piece.bounds[-1]
                        selected_angle = angle
                        final_piece = moving_piece

                    # visualize_polygon(nfp)

                located_pieces[piece_id] = final_piece
                placed_pieces.append(piece_id)
                angle_list[piece_id] = selected_angle

                # visualize_polygon_list(located_pieces, bin_width, bin_height)

            final_utilization = get_utilization(bin_width, located_pieces)

            print(f'final utilization: {final_utilization}')

            pieces_info = {}
            for p in range(len(located_pieces)):
                pieces_info[p] = list(located_pieces[p].exterior.coords)
            best_placement = (pieces_info, final_utilization)
            save_best_placement(best_placement, result_path)


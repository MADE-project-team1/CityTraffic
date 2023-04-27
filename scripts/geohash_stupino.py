import folium
import geohash_hilbert as gh

def add_geohash(map, hash, bits_per_char):
    rect = gh.rectangle(hash, bits_per_char=bits_per_char)
    bounds = rect['geometry']['coordinates'][0]
    folium.Polygon(
        bounds,
        color='black',
        fill_color='gray',
        weight=1,
        popup=hash,
        fill_opacity=0.5
    ).add_to(map)


def is_geohash_in_bounding_box(current_geohash, bbox_coordinates,  bits_per_char):
    """Checks if the box of a geohash is inside the bounding box
    :param current_geohash: a geohash
    :param bbox_coordinates: bounding box coordinates
    :return: true if the the center of the geohash is in the bounding box
    """

    coordinates = gh.decode(current_geohash, bits_per_char=bits_per_char)
    geohash_in_bounding_box = (bbox_coordinates[2] < coordinates[0] < bbox_coordinates[0]) and (
            bbox_coordinates[1] < coordinates[1] < bbox_coordinates[3])
    return geohash_in_bounding_box

def build_geohash_box(current_geohash, bits_per_char):
    """
    Returns a GeoJSON Polygon for a given geohash
    :param current_geohash: a geohash
    :return: a list representation of th polygon
    """

    b = gh.rectangle(current_geohash, bits_per_char=bits_per_char)['bbox']
    polygon = [(b['w'], b['s']), (b['w'], b['n']), (b['e'], b['n']), (b['e'], b['s'],), (b['w'], b['s'])]
    return polygon

def compute_geohash_tiles(bbox_coordinates, prec, bits_per_char):
    """Computes all geohash tile in the given bounding box
    :param bbox_coordinates: the bounding box coordinates of the geohashes
    :return: a list of geohashes
    """

    checked_geohashes = set()
    geohash_stack = set()
    geohashes = []
    # get center of bounding box, assuming the earth is flat ;)
    center_latitude = (bbox_coordinates[0] + bbox_coordinates[2]) / 2
    center_longitude = (bbox_coordinates[1] + bbox_coordinates[3]) / 2

    center_geohash = gh.encode(center_latitude, center_longitude, precision=prec, bits_per_char=bits_per_char)
    geohashes.append(center_geohash)
    geohash_stack.add(center_geohash)
    checked_geohashes.add(center_geohash)
    while len(geohash_stack) > 0:
        current_geohash = geohash_stack.pop()
        neighbours = gh.neighbours(current_geohash, bits_per_char=bits_per_char).values()
        for neighbour in neighbours:
            # print(gh.decode(neighbour, bits_per_char=bits_per_char), neighbour)
            # print(bbox_coordinates)
            if neighbour not in checked_geohashes and is_geohash_in_bounding_box(neighbour, bbox_coordinates, bits_per_char):
                # print(gh.decode(neighbour, bits_per_char=bits_per_char))
                geohashes.append(neighbour)
                geohash_stack.add(neighbour)
                checked_geohashes.add(neighbour)

    return geohashes

def add_geohash_grid(layer, bbox, prec, bits_per_char):
    lu_lat, lu_lon = bbox["lu"]
    rl_lat, rl_lon = bbox["rl"]

    NW = gh.encode(lu_lat, lu_lon, precision=prec, bits_per_char=bits_per_char)
    NE = gh.encode(lu_lat, rl_lon, precision=prec, bits_per_char=bits_per_char)
    SW = gh.encode(rl_lat, lu_lon, precision=prec, bits_per_char=bits_per_char)
    SE = gh.encode(rl_lat, rl_lon, precision=prec, bits_per_char=bits_per_char)

    gh_for_bbox = compute_geohash_tiles((lu_lat, lu_lon, rl_lat, rl_lon), prec, bits_per_char)
    for cur_gh in gh_for_bbox:
        add_geohash(layer, cur_gh, bits_per_char)
    
    return map

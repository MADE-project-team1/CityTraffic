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

    coordinates = gh.decode(current_geohash, bits_per_char=bits_per_char)
    geohash_in_bounding_box = (bbox_coordinates[2] < coordinates[0] < bbox_coordinates[0]) and (
            bbox_coordinates[1] < coordinates[1] < bbox_coordinates[3])
    return geohash_in_bounding_box


def build_geohash_box(current_geohash, bits_per_char):
    b = gh.rectangle(current_geohash, bits_per_char=bits_per_char)['bbox']
    polygon = [(b['w'], b['s']), (b['w'], b['n']), (b['e'], b['n']), (b['e'], b['s'],), (b['w'], b['s'])]
    return polygon


def compute_geohash_tiles(bbox_coordinates, prec, bits_per_char):

    checked_geohashes = set()
    geohash_stack = set()
    geohashes = []

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
            if neighbour not in checked_geohashes and is_geohash_in_bounding_box(neighbour, bbox_coordinates, bits_per_char):
                geohashes.append(neighbour)
                geohash_stack.add(neighbour)
                checked_geohashes.add(neighbour)

    return geohashes


def add_geohash_grid(layer, bbox, prec, bits_per_char):
    lu_lat, lu_lon = bbox["lu"]
    rl_lat, rl_lon = bbox["rl"]

    gh_for_bbox = compute_geohash_tiles((lu_lat, lu_lon, rl_lat, rl_lon), prec, bits_per_char)
    for cur_gh in gh_for_bbox:
        add_geohash(layer, cur_gh, bits_per_char)
    

import trimesh
try:
    import pymesh
except ImportError:
    pass


def pymesh2trimesh(m):
    return trimesh.Trimesh(m.vertices, m.faces)


def trimesh2pymesh(m):
    return pymesh.form_mesh(m.vertices, m.faces)


def repair_self_intersection(m):
    m = trimesh2pymesh(m)

    m, _ = pymesh.remove_degenerated_triangles(m)
    mt = pymesh2trimesh(m)
    if mt.is_watertight:
        return mt

    m, _ = pymesh.remove_duplicated_vertices(m)
    mt = pymesh2trimesh(m)
    if mt.is_watertight:
        return mt

    m, _ = pymesh.remove_duplicated_faces(m)
    mt = pymesh2trimesh(m)
    if mt.is_watertight:
        return mt

    m = pymesh.resolve_self_intersection(m)
    return pymesh2trimesh(m)

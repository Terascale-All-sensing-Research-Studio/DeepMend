import os

import vedo
import trimesh
import pyrender
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree as KDTree

os.environ["PYOPENGL_PLATFORM"] = "egl"


def trimesh2vedo(mesh, **kwargs):
    return vedo.Mesh([mesh.vertices, mesh.faces], **kwargs)


def force_trimesh(mesh, remove_texture=False):
    """
    Forces a mesh or list of meshes to be a single trimesh object.
    """

    if isinstance(mesh, list):
        return [force_trimesh(m) for m in mesh]

    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            mesh = trimesh.Trimesh()
        else:
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in mesh.geometry.values()
                )
            )
    if remove_texture:
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    return mesh


def render_mesh(
    mesh,
    objects=None,
    mode="RGB",
    remove_texture=False,
    yfov=(np.pi / 4.0),
    resolution=(1280, 720),
    xtrans=0.0,
    ytrans=0.0,
    ztrans=2.0,
    xrot=-25.0,
    yrot=45.0,
    zrot=0.0,
):
    assert len(resolution) == 2

    mesh = force_trimesh(mesh, remove_texture)

    # Create a pyrender scene with ambient light
    scene = pyrender.Scene(ambient_light=np.ones(3))

    if objects is not None:
        for o in objects:
            o = o.subdivide_to_size(max_edge=0.05)
            n = o.vertices.shape[0]
            o.visual = trimesh.visual.create_visual(
                vertex_colors=np.hstack(
                    (
                        np.ones((n, 1)) * 0,
                        np.ones((n, 1)) * 0,
                        np.ones((n, 1)) * 255,
                        np.ones((n, 1)) * 50,
                    )
                )
            )
            scene.add(pyrender.Mesh.from_trimesh(o, wireframe=True))

    if isinstance(mesh, list):
        for m in mesh:
            scene.add(pyrender.Mesh.from_trimesh(m))
    else:
        scene.add(pyrender.Mesh.from_trimesh(mesh))

    camera = pyrender.PerspectiveCamera(
        yfov=yfov, aspectRatio=resolution[0] / resolution[1]
    )

    # Apply translations
    trans = np.array(
        [
            [1.0, 0.0, 0.0, xtrans],
            [0.0, 1.0, 0.0, ytrans],
            [0.0, 0.0, 1.0, ztrans],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # Apply rotations
    xrotmat = trimesh.transformations.rotation_matrix(
        angle=np.radians(xrot), direction=[1, 0, 0], point=(0, 0, 0)
    )
    camera_pose = np.dot(xrotmat, trans)
    yrotmat = trimesh.transformations.rotation_matrix(
        angle=np.radians(yrot), direction=[0, 1, 0], point=(0, 0, 0)
    )
    camera_pose = np.dot(yrotmat, camera_pose)
    zrotmat = trimesh.transformations.rotation_matrix(
        angle=np.radians(zrot), direction=[0, 0, 1], point=(0, 0, 0)
    )
    camera_pose = np.dot(zrotmat, camera_pose)

    # Insert the camera
    scene.add(camera, pose=camera_pose)

    # Insert a splotlight to give contrast
    spot_light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=8.0,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 6.0,
    )
    scene.add(spot_light, pose=camera_pose)

    # Render!
    r = pyrender.OffscreenRenderer(*resolution)
    color, _ = r.render(scene)
    return np.array(Image.fromarray(color).convert(mode))


def chamfer(gt_shape, pred_shape, num_mesh_samples=30000):
    """
    Compute the chamfer distance for two 3D meshes.
    This function computes a symmetric chamfer distance, i.e. the mean chamfers.
    Based on the code provided by DeepSDF.

    Args:
        gt_shape (trimesh object or points): Ground truth shape.
        pred_shape (trimesh object): Predicted shape.
        num_mesh_samples (points): Number of points to sample from the predicted
            shape. Must be the same number of points as were computed for the
            ground truth shape.
    """

    assert gt_shape.vertices.shape[0] != 0, "gt shape has no vertices"

    try:
        gt_pts = trimesh.sample.sample_surface(gt_shape, num_mesh_samples)[0]
    except AttributeError:
        gt_pts = gt_shape
        assert (
            gt_pts.shape[0] == num_mesh_samples
        ), "Wrong number of gt points, expected {} got {}".format(
            num_mesh_samples, gt_pts.shape[0]
        )
    pred_pts = trimesh.sample.sample_surface(pred_shape, num_mesh_samples)[0]

    # one direction
    one_distances, _ = KDTree(pred_pts).query(gt_pts)
    gt_to_pred_chamfer = np.mean(np.square(one_distances))

    # other direction
    two_distances, _ = KDTree(gt_pts).query(pred_pts)
    pred_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_pred_chamfer + pred_to_gt_chamfer


def connected_components(mesh):
    """
    Return number of connected components.
    """
    return len(trimesh2vedo(mesh).splitByConnectivity())


def component_difference(gt_shape, pred_shape):
    """
    Return difference of connected components.
    """
    return abs(connected_components(pred_shape) - connected_components(gt_shape))

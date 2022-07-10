import argparse
import logging
import os

import trimesh
import pyrender
import numpy as np
from PIL import Image


os.environ["PYOPENGL_PLATFORM"] = "egl"


def force_trimesh(mesh, remove_texture=False):
    """Take a trimesh mesh, scene, or list of meshes and force it to be a mesh"""

    if isinstance(mesh, list):
        return [force_trimesh(m) for m in mesh]

    if isinstance(mesh, trimesh.PointCloud):
        return trimesh.Trimesh(mesh.vertices, vertex_colors=mesh.colors)
    elif isinstance(mesh, trimesh.Scene):
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
    mode="RGB",
    remove_texture=False,
    yfov=(np.pi / 4.0),
    resolution=(1280, 720),
    xtrans=0.0,
    ytrans=0.0,
    ztrans=1.5,
    xrot=-25.0,
    yrot=45.0,
    zrot=0.0,
    depth=False,
):
    """Take a trimesh object and render it, offscreen"""
    assert len(resolution) == 2

    mesh = force_trimesh(mesh, remove_texture)

    # Create a pyrender scene with ambient light
    scene = pyrender.Scene(ambient_light=np.ones(3))

    # Add the meshes
    if not isinstance(mesh, list):
        mesh = [mesh]
    for m in mesh:
        scene.add(pyrender.Mesh.from_trimesh(m))

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
    if depth:
        _, img = r.render(scene)
    else:
        img, _ = r.render(scene)
    return np.array(Image.fromarray(img).convert(mode))


def process(
    f_in,
    f_out,
    save_depth=False,
    normalize_depth=True,
    yfov=(np.pi / 4.0),
    resolution=(1280, 720),
    xtrans=0.0,
    ytrans=0.0,
    ztrans=1.5,
    xrot=-25.0,
    yrot=45.0,
    zrot=0.0,
    overwrite=False
):
    # Load mesh
    mesh = trimesh.load(f_in)

    # Render
    img = render_mesh(
        mesh=mesh,
        yfov=yfov,
        resolution=resolution,
        xtrans=xtrans,
        ytrans=ytrans,
        ztrans=ztrans,
        xrot=xrot,
        yrot=yrot,
        zrot=zrot,
        mode="L",
        remove_texture=True,
    )

    if save_depth:
        if normalize_depth:
            img *= 255.0 / img.max()
        logging.debug("Saving depth image with size {}x{}".format(*resolution))
        img = Image.fromarray(img).convert("L")
    else:
        logging.debug("Saving color image with size {}x{}".format(*resolution))
        img = Image.fromarray(img).convert("RGB")
    
    if overwrite or not os.path.exists(f_out):
        logging.debug("Saving to: {}".format(f_out))
        img.save(f_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a color or depth render of a mesh."
    )
    parser.add_argument(dest="input", type=str, help="Path to the input file.")
    parser.add_argument(dest="output", type=str, help="Path to the output file.")
    parser.add_argument(
        "--depth",
        "-d",
        action="store_true",
        default=False,
        help="If passed, will render the object in depth rather than in color.",
    )
    parser.add_argument(
        "--raw_depth",
        action="store_true",
        default=False,
        help="If passed, will return the true depth values, un-normalized.",
    )
    parser.add_argument(
        "--yfov",
        "-f",
        default=(np.pi / 4.0),
        type=float,
        help="The vertical field-of-view of the camera.",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        nargs=2,
        default=[1280, 720],
        type=int,
        help="The resolution of the resulting render.",
    )
    args = parser.parse_args()

    process(
        f_in=args.input,
        f_out=args.output,
        save_depth=args.depth,
        normalize_depth=(not args.raw_depth),
        yfov=args.yfov,
        resolution=args.resolution,
    )

import os
import logging
import pickle

import trimesh
import numpy as np
import scipy.interpolate as interpolator
from sklearn.decomposition import PCA

import processor.errors as errors


def points_maximal_orient(points):
    """ Return the transformation matrix that orients a point set by its maximal dimensions """
    pca = PCA(n_components=3)
    pca.fit(points)
    matrix = pca.components_
    return np.vstack((
        np.hstack((
            np.expand_dims(matrix[0, :], axis=1), # X corresponds to largest component
            np.expand_dims(matrix[1, :], axis=1), #
            np.expand_dims(matrix[2, :], axis=1), # Z corresponds to smallest component
            np.zeros((3, 1))
        )),
        np.array([0, 0, 0, 1])
    )).T
    

def normalize_transform(v):
    """ Return matrix that centers vertices """
    return trimesh.transformations.translation_matrix(-v.mean(axis=0))


def points_transform_matrix(vs, mat):
    """ Apply a transformation matrix to a set of points """
    return np.dot(
        mat, 
        np.hstack((
            vs, 
            np.ones((vs.shape[0], 1))
        )).T
    ).T[:, :3]


def intersect_mesh(a, b, sig=5):
    """get mask of vertices in a occurring in both a and b, corresponding to a"""
    av = [frozenset(np.round(v, sig)) for v in a]
    bv = set([frozenset(np.round(v, sig)) for v in b])
    return np.asarray(list(map(lambda v: v in bv, av)))


def get_fracture_points(b, r):
    """ Get points on the fracture """
    vb, vr = b.vertices, r.vertices
    logging.debug(
        "Computing fracture points for meshes with size {} and {} ..."
        .format(vb.shape[0], vr.shape[0])
    )
    return intersect_mesh(vb, vr)


def plot_3d(f, inv_orienter, density=128, limit=0.5):
    x, y = np.meshgrid(
        np.linspace(-limit, limit, density),
        np.linspace(-limit, limit, density),
        indexing="xy"
    )
    x = x.flatten()
    y = y.flatten()
    
    z = f(x, y)
    
    pts = np.hstack((
        np.expand_dims(x.flatten(), axis=1), 
        np.expand_dims(y.flatten(), axis=1), 
        np.expand_dims(z.flatten(), axis=1),
    ))

    return inv_orienter(pts)


def fit_3d(b, r, method="thin_plate", smoothing=0):
    # Get the fracture region
    logging.debug("Computing fracture points")
    frac_points = b.vertices[get_fracture_points(b, r), :]
    # assert frac_points.shape[0] > 200, "Too few fracture points"

    # Orient the fracture region and extract the corresponding matrix
    mat1 = normalize_transform(frac_points)
    norm_frac_points = points_transform_matrix(
        frac_points, 
        mat1,
    )
    mat2 = points_maximal_orient(norm_frac_points)
    oriented_norm_frac_points = points_transform_matrix(
        norm_frac_points, 
        mat2,
    )
    mat = mat2 @ mat1
    mat_inv = np.linalg.inv(mat)

    logging.debug("Fitting 3D function on {} points ..."
        .format(oriented_norm_frac_points.shape[0])
    )

    fit_function = interpolator.Rbf(
        oriented_norm_frac_points[:, 0],   
        oriented_norm_frac_points[:, 1],   
        oriented_norm_frac_points[:, 2],
        function=method,
        smoothing=smoothing,
    )

    def orienter(pts):
        return points_transform_matrix(pts, mat)

    def inv_orienter(pts):
        return points_transform_matrix(pts, mat_inv)

    return fit_function, orienter, inv_orienter


def fit_quality(function, orienter, b, r):
    ptsb = orienter(b.vertices)
    zb = function(ptsb[:, 0], ptsb[:, 1])
    ptsr = orienter(r.vertices)
    zr = function(ptsr[:, 0], ptsr[:, 1])
    occupancy = np.hstack((
        zb >= ptsb[:, 2],
        zr <= ptsr[:, 2],
    )).astype(bool)
    if 1-occupancy.mean() > occupancy.mean():
        occupancy = ~occupancy
    return occupancy.astype(int)


def batch_eval(function, pts, batch_size=50000):
    accumulator = []
    for start in range(0, pts.shape[0], batch_size):
        end = min(start + batch_size, pts.shape[0])
        accumulator.append(
            function(pts[start:end, 0], pts[start:end, 1])
        )
    return np.hstack(accumulator).flatten()


def compute_fitted_occupancy(input_pts, function, orienter, b, r, return_accuracy=False):
    logging.debug("Computing sample occupancy from spline ...")
    # Evaluate broken
    oriented_b_pts = orienter(b.vertices)
    zb = batch_eval(function, oriented_b_pts)

    # Evaluate restoration
    oriented_r_pts = orienter(r.vertices)
    zr = batch_eval(function, oriented_r_pts)
    
    # Evaluate sample points
    oriented_input_pts = orienter(input_pts)
    zpts = batch_eval(function, oriented_input_pts)
    
    # Compute if the point is above or below the plane
    occupancy = np.hstack((
        zb >= oriented_b_pts[:, 2],
        zr <= oriented_r_pts[:, 2],
    )).astype(bool)
    pt_occupancy = (zpts < oriented_input_pts[:, 2]).astype(bool)
    
    # We don't know which way the plane is oriented, so we may need to flip it
    if (1-occupancy.mean()) > occupancy.mean():
        occupancy = ~occupancy
    
    if return_accuracy:
        return pt_occupancy, occupancy.mean()
    return pt_occupancy


def process(
    f_in,
    f_sdf,
    f_rest,
    f_rest_sdf,
    f_samp, 
    f_out,
    f_plane,
    method="thin_plate",
    overwrite=False,
):
    # Load meshes
    broken_mesh = trimesh.load(f_in)
    if not broken_mesh.is_watertight:
        raise errors.MeshNotClosedError
    restoration_mesh = trimesh.load(f_rest)
    if not restoration_mesh.is_watertight:
        raise errors.MeshNotClosedError
    pts = np.load(f_samp)["xyz"]

    for _ in range(5):
        # Fit
        f, orienter, inv_orienter = fit_3d(broken_mesh, restoration_mesh, method)

        # This gives you the points corresponding to the fit spline
        plane_points = plot_3d(f, inv_orienter)

        # Compute point occupancy
        mask, accuracy = compute_fitted_occupancy(
            pts, 
            f, 
            orienter, 
            broken_mesh, 
            restoration_mesh, 
            return_accuracy=True
        )

        logging.debug("Computed spline fit with accuracy: {}".format(accuracy))
        # if accuracy < 0.9:
        #     raise errors.SplineFitError

        if accuracy > 0.9:
            break

    logging.info("Computed spline fit with accuracy: {}".format(accuracy))

    b_sdf = np.load(f_sdf)["sdf"]
    r_sdf = np.load(f_rest_sdf)["sdf"]

    inside_b = (b_sdf.squeeze() <= 0.0)
    inside_r = (r_sdf.squeeze() < 0.0)

    correct = (mask[inside_b] == False).sum() + (mask[inside_r] == True).sum()
    correct_flipped = (~mask[inside_b] == False).sum() + (~mask[inside_r] == True).sum()
    if correct_flipped > correct:
        mask = ~mask

    # Mask should include no points in the fracture 
    mask[inside_b] = False
    # Mask should include all points in the restoration
    mask[inside_r] = True

    # Export
    if overwrite or not os.path.exists(f_out):
        logging.debug("Saving to: {}".format(f_out))
        np.savez(f_out, occ=mask)

    if overwrite or not os.path.exists(f_plane):
        logging.debug("Saving to: {}".format(f_plane))
        trimesh.PointCloud(plane_points).export(f_plane)
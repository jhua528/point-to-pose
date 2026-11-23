"""ZeroMQ inference server exposing ModularPipeline to a remote ROS2 client.

Each object's bounding box is fit here and then carried by the tracked
motion. Without a supplied mesh, the fit uses no known bench
height, no gravity, no object shape -- only the mask and the depth:

  1. The surface the object sits on is fit (RANSAC) to a ring of pixels just
     outside the mask. Mask pixels on that surface are the background the mask
     spilled onto at the object's edge, and are dropped.
  2. What is left gets a minimal-volume box. PCA, the usual shortcut, follows
     how the points are spread; one view sees an L-shaped cloud (a top and a
     side), and PCA tilts the box across it, making it too wide and askew.

A box fit to what one view sees is only as good as the view: a part half
hidden behind a hand gets a box too short, or one tilted across the hidden
side and too large. When the client sends a mesh of the object (one estimated
from a demonstration, say), the box is the mesh's own instead, placed by
fitting its pose to the observations. The mesh supplies the part's shape;
its size is fitted too, since a similar mesh need not be the part's size nor
quite its proportions: one scale, then -- where the view shows the part's
length along them -- a scale along each of the mesh's box axes, fitted with
the pose (MODEL_STRETCH_*). A mesh that already is the part's size keeps
exactly scale 1. Carried by the tracked motion, a fitted mesh's pose is then
corrected on each frame's depth (MODEL_CORRECT_*): a small part with little
texture gives the tracker few points to hold it by.
A failed pose fit stays invalid while subsequent frames are retried in the
background. Point-cloud boxes are used only when no mesh was supplied.

An init may also ask for the obstacles in view (meta["discover"]; see
OBSTACLE_MIN_PX): found with no click and no mesh, followed on every frame by
the depth, and replied apart from the parts ("obstacles").

Which way round a fitted mesh lies, where the init view leaves it in doubt,
is settled while the part is tracked: every way round in doubt rides on the
part's one tracked motion, and each frame's view is evidence for and against
each, until one is left or the person names one (BINDING_CONFIDENCE). Each
reply says how far that is ("model_binding").
"""

import argparse
import base64
import atexit
import tempfile
import shutil
import collections
import copy
import functools
import gc
import hashlib
import itertools
import json
import os
import sys
import threading
import time
import zlib
import traceback
import types
import uuid
import warnings
import pickle
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool

# A mesh-fit worker (FIT_WORKERS) is this file imported again in a process of
# its own: it fits with numpy, scipy and cv2 alone. Without torch, SAM2 and the
# pipeline it holds about 110 MB, not 450, and never touches the GPU.
FIT_WORKER_ENV = "POSE_SERVER_FIT_WORKER"
IN_FIT_WORKER = os.environ.get(FIT_WORKER_ENV) == "1"

import cv2
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

if not IN_FIT_WORKER:
    import torch
    import zmq
    from omegaconf import OmegaConf

    from point2pose.data_types.frame import Frame
    from point2pose.modules.segmenter.sam2_real_time_segmenter import Sam2RealTimeSegmenter
    from point2pose.modules.tracker.tapir_tracker import TapirTracker
    from point2pose.pipeline.modular_pipeline import ModularPipeline

    # Every init builds a new pipeline; its SAM2 and TAPIR reuse the models
    # the first one loaded (~3-5 s an init otherwise) and only their tracking
    # state is new. kept_model_mib() is their share of the GPU.
    Sam2RealTimeSegmenter.keep_models = True
    TapirTracker.keep_models = True


def decode_rgb(buf):
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_depth(buf, meta):
    # Both encodings are lossless; float32 metres travel raw because PNG is integer-only.
    if meta.get("depth_enc") == "raw_f32":
        h, w = meta["depth_shape"]
        return np.frombuffer(zlib.decompress(buf), np.float32).reshape(h, w)
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)


def decode_robot_mask(meta, shape):
    """The robot's pixels the client sent with an init, or None: meta
    ["robot_mask"] holds the mask's shape and base64 of zlib of its
    np.packbits(). One that cannot be read or is not the depth's size is
    left out, and the proposals on the robot are fitted like any other."""
    spec = meta.get("robot_mask")
    if not spec:
        return None
    try:
        h, w = (int(n) for n in spec["shape"])
        if (h, w) != tuple(shape):
            raise ValueError(f"{h}x{w} against a {shape[0]}x{shape[1]} depth")
        bits = np.frombuffer(zlib.decompress(base64.b64decode(spec["bits"])), np.uint8)
        return np.unpackbits(bits, count=h * w).reshape(h, w).astype(bool)
    except Exception as exc:
        print(f"[server] unusable robot mask ({exc}); proposals on the robot are fitted too",
              flush=True)
        return None


def anchor_from_points(points, labels, depth, K, depth_factor, radius=3):
    """Object origin on the init frame: the clicked surface point in camera coordinates.

    The pipeline reports each object's motion since the init frame, with the
    object model stored in camera coordinates, so its origin is the camera
    centre. Composing that motion with this anchor turns it into the object's
    pose in the camera frame.
    """
    metres = depth / depth_factor
    xyz = []
    for (u, v), label in zip(points, labels):
        if not label:
            continue
        u, v = int(round(u)), int(round(v))
        patch = metres[max(v - radius, 0):v + radius + 1, max(u - radius, 0):u + radius + 1]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size:
            z = float(np.median(valid))
            xyz.append([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
    if not xyz:
        raise ValueError(f"no valid depth under the picked points {points}")
    anchor = np.eye(4)
    anchor[:3, 3] = np.median(xyz, axis=0)
    return anchor


# Unit-cube corner signs, ordered so bit 0/1/2 of the index flips x/y/z.
_CORNER_SIGNS = np.array([[(i >> k & 1) * 2 - 1 for k in range(3)] for i in range(8)], np.float64)


def as_numpy(value):
    """float64 array from numpy or a (possibly CUDA) torch tensor."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, np.float64)


# Pixel lengths and areas here are those of an image REFERENCE_WIDTH_PX wide;
# on another, they scale with its width (image_px(), image_area_px()): SAM2
# segments every image at its own size, so its edge is off by a share of the
# image, and a depth image's speckle and pieces go with its resolution too.
REFERENCE_WIDTH_PX = 640


def image_px(n, shape):
    """`n` pixels of an image REFERENCE_WIDTH_PX wide, on one of `shape`."""
    return max(1, int(round(n * shape[1] / REFERENCE_WIDTH_PX)))


def image_area_px(n, shape):
    """`n` square pixels of an image REFERENCE_WIDTH_PX wide, on one of `shape`."""
    return max(1, int(round(n * (shape[1] / REFERENCE_WIDTH_PX) ** 2)))


# Box fitting. Camera frame and pixels throughout; nothing about this scene.
# A segmentation's edge is off the part's by a few pixels: SEGMENT_EDGE_PX, the
# one band every pixel length here is of -- a mask grown by it before a point
# counts as outside it (MODEL_MASK_GROW_PX), a located part's region grown by
# it (LOCATE_REGION_GROW_PX), clicks kept it in from an edge (LOCATE_EDGE_PX).
SEGMENT_EDGE_PX = 4
# The ring the support surface is fit to starts clear of the band at the mask
# edge where spill lives, twice it, and is wide enough to hold a plane's worth
# of pixels.
PLANE_RING_PX = (2 * SEGMENT_EDGE_PX, 25)
# RANSAC inlier distance, in TOLs; also how far in front of the surface a point
# must be to count as the object. Parts thinner than this fall back to the raw
# mask. Before the noise is known (it is measured on this plane), the pixel's
# span alone sets the TOL (pixel_tol()).
PLANE_TOLS = 5 / 6
PLANE_RANSAC_ITERS = 300
# Below this share of ring pixels on the best plane there is no clear surface
# (clutter, an object held in the air), and nothing is dropped.
PLANE_MIN_INLIERS = 0.3
MIN_BOX_POINTS = 30
# Boxes within this much of the smallest volume count as ties; see min_obb().
BOX_VOLUME_SLACK = 0.1
# Fewer points than this share of the mask left in front of the surface means
# the part lies flat on it: depth cannot tell it from its support.
FLAT_KEEP_FRACTION = 0.2
# A part rests on its support, so at least this share of its mask's points lie
# on the plane or in front of it, within a TOL (_ModelFit asks the same).
# A plane short of that is an occluder the ring caught: RI-S4-2's ring lies on
# the table behind the box, the box's top filled the ring round its mask, and
# the plane fitted there had the ring 140 mm behind it -- no point in front of
# it, so the ring was taken to lie flat on it, its raw mask fitted (the box's
# edge spilled into it, the table seen through its hole; 35 mm off at the 95th
# percentile), and the box's latch was located as the ring instead. So the
# ring's points on such a plane or in front of it go, and the plane is fitted
# again on what is behind, up to PLANE_OCCLUDER_ROUNDS times.
PLANE_SUPPORT_SHARE = 0.9
PLANE_OCCLUDER_ROUNDS = 2


def backproject(metres, K):
    """Camera-frame point for every pixel, and which pixels have depth."""
    h, w = metres.shape
    v, u = np.mgrid[0:h, 0:w]
    valid = np.isfinite(metres) & (metres > 0)
    z = np.where(valid, metres, 0.0)
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1), valid


def pixel_tol(z, K):
    """TOL as the pixels alone set it: TOL_SIGMAS of PIXEL_SIGMA of the length
    one pixel spans at depth `z` (the median of the values given), metres."""
    z = np.asarray(z, np.float64)
    z = z[np.isfinite(z) & (z > 0)]
    return TOL_SIGMAS * PIXEL_SIGMA * (float(np.median(z)) if len(z) else 1.0) / float(K[0, 0])


def cam_pixel_tol(cam, mask):
    """pixel_tol() from the camera-frame points alone: the span between
    horizontally neighbouring pixels of `mask` (z/f), median."""
    both = mask[:, :-1] & mask[:, 1:] & (cam[:, :-1, 2] > 0) & (cam[:, 1:, 2] > 0)
    if not both.any():
        return TOL_SIGMAS * PIXEL_SIGMA * 0.002
    span = np.abs(cam[:, 1:, 0] - cam[:, :-1, 0])[both] / np.maximum(1e-9, cam[:, 1:, 2][both]) * cam[:, 1:, 2][both]
    return TOL_SIGMAS * PIXEL_SIGMA * float(np.median(span))


def depth_gap_share(K=None, cam=None):
    """The share of their depth two neighbouring pixels' depths may differ by
    and be one surface (MODEL_DEPTH_GAP_DEG): tan(it) / f, f from K, else
    from `cam` (every pixel's camera-frame point: x/z steps 1/f a column),
    else MODEL_DEPTH_GAP_SHARE."""
    inv_f = None
    if K is not None:
        inv_f = 1.0 / float(K[0, 0])
    elif cam is not None:
        z = cam[..., 2]
        both = (z[:, :-1] > 0) & (z[:, 1:] > 0)
        if both.any():
            ratio = np.where(z > 0, cam[..., 0] / np.where(z > 0, z, 1.0), 0.0)
            inv_f = float(np.median((ratio[:, 1:] - ratio[:, :-1])[both]))
    if not inv_f or inv_f <= 0:
        return MODEL_DEPTH_GAP_SHARE
    return inv_f * np.tan(np.radians(MODEL_DEPTH_GAP_DEG))


def depth_quantum(depth, region):
    """The step a depth image's values come in on `region` (metres): the
    smallest difference between two of its distinct values -- a millimetre
    for a camera that reports whole millimetres, next to nothing for floats."""
    v = np.unique(depth[region & np.isfinite(depth) & (depth > 0)])
    d = np.diff(v)
    d = d[d > 0]
    return float(d.min()) if len(d) else 0.0


def image_noise(depth, mask, lags=(1,)):
    """The depth's noise on `mask` (metres): of every three pixels `lag` apart
    in a row or column of it, the second difference d[-lag] - 2 d[0] + d[+lag]
    -- a smooth surface's is near nothing, the noise's spread is sqrt(6)
    sigma -- its robust spread (scaled median absolute deviation) over
    sqrt(6), the largest over `lags` (noise that runs over a few pixels shows
    at the longer ones, on a flat region), with the depth's step's own,
    1/sqrt(12) of it (depth_quantum()); None with too few."""
    d = np.where(mask & np.isfinite(depth) & (depth > 0), depth, np.nan)
    # In inverse depth: a plane's is linear in the pixel's coordinates, so a
    # slanted support's second differences are nothing at any lag (its depth's
    # are not, under perspective); back to metres by the middle pixel's z^2.
    w = 1.0 / d
    best = None
    for lag in lags:
        if 2 * lag >= min(d.shape):
            continue
        second = np.concatenate([((w[:, :-2 * lag] - 2 * w[:, lag:-lag] + w[:, 2 * lag:]) * d[:, lag:-lag] ** 2).ravel(),
                                 ((w[:-2 * lag, :] - 2 * w[lag:-lag, :] + w[2 * lag:, :]) * d[lag:-lag, :] ** 2).ravel()])
        second = second[np.isfinite(second)]
        if len(second) < MIN_BOX_POINTS:
            continue
        sigma = float(1.4826 * np.median(np.abs(second - np.median(second))) / np.sqrt(6.0))
        best = sigma if best is None else max(best, sigma)
    if best is None:
        return None
    step = depth_quantum(depth, mask)
    return float(np.sqrt(best ** 2 + step ** 2 / 12.0))


def noise_tol(depth, region, lags=(1, 2)):
    """TOL_SIGMAS of the depth's noise on `region` (image_noise(), at `lags`,
    for a flat one: the support's ring), at least SUPPORT_CUT_FLOOR_M; None
    where too little of it has depth to tell."""
    sigma = image_noise(depth, region, lags)
    return None if sigma is None else max(TOL_SIGMAS * sigma, SUPPORT_CUT_FLOOR_M)


def plane_ring(mask):
    """The ring round `mask` its support is fitted on (PLANE_RING_PX)."""
    inner, outer = (image_px(n, mask.shape) for n in PLANE_RING_PX)

    def grown(radius):
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel) > 0

    return grown(outer) & ~grown(inner)


# What lies off the support by more than PLANE_TOLS of view_tol() is the part.
# A point of the support is on its plane whichever pixel saw it, so how wide a
# pixel is says nothing of how far off the plane the support's points lie:
# only the depth's noise does, as the ring measures it (depth_noise()); the
# plane's own error, fitted to hundreds of the ring's points, is a small
# share of that. (Until 2026-11-18 at least TOL_PX pixels' span: 4-6.6 mm
# on the scenes, where the bench lies within 0.5 mm of its plane -- a sample's
# top, 5-5.7 mm up, a fifth cut away; at half the resolution twice that, and a
# ring 10 mm thick, too thin for it, fitted with the bench through its hole.)
# Below SUPPORT_CUT_FLOOR_M the depth tells nothing apart: a guard against a
# rendered depth's noise of nothing, no part's or camera's size.
SUPPORT_CUT_FLOOR_M = 1e-4


def view_tol(z, K, sigma=None):
    """The support's TOL (see SUPPORT_CUT_FLOOR_M): TOL_SIGMAS times the depth
    noise `sigma` the ring measured; pixel_tol() where it is not known."""
    if sigma is None:
        return pixel_tol(z, K)
    return max(TOL_SIGMAS * sigma, SUPPORT_CUT_FLOOR_M)


def fit_tol(points, K, sigma=None, diagonal=None, share=None):
    """A fit's TOL on a view of `points` (the part's, camera frame): the
    largest of TOL_SIGMAS of the noise, of the pixel's own spread (pixel_tol())
    and FIT_TOL_SIZE of the part's diagonal: `diagonal` where its size
    is known, at least what the points span (2nd to 98th percentile) -- a
    part half hidden is no smaller, nor is its mesh any truer."""
    points = np.asarray(points, np.float64)
    if len(points) < 2:
        return pixel_tol(points[:, 2] if len(points) else [1.0], K)
    lo, hi = np.percentile(points, [2, 98], axis=0)
    size = max(float(np.linalg.norm(hi - lo)), diagonal or 0.0)
    return max(TOL_SIGMAS * (sigma or 0.0), pixel_tol(points[:, 2], K),
               (FIT_TOL_SIZE if share is None else share) * size)


def support_plane(mask, cam, valid, rng, tol):
    """(normal, point) of the surface around the mask, normal facing away from
    the camera; None when the ring holds no clear plane. A plane the mask lies
    behind is an occluder, not the support (PLANE_SUPPORT_SHARE)."""
    def grown(radius):
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel) > 0

    inner, outer = (image_px(n, mask.shape) for n in PLANE_RING_PX)
    pts = cam[grown(outer) & ~grown(inner) & valid]
    own = cam[mask & valid]
    for _ in range(PLANE_OCCLUDER_ROUNDS + 1):
        plane = _ring_plane(pts, rng, tol)
        if plane is None:
            return None
        n, p = plane
        if not len(own) or np.mean((own - p) @ n <= tol) >= PLANE_SUPPORT_SHARE:
            return plane
        pts = pts[(pts - p) @ n > PLANE_TOLS * tol]
    return None


def depth_noise(mask, cam, plane, tol):
    """The depth's noise about the part (metres), as the support around it
    shows: the robust spread (scaled median absolute deviation) about
    `plane` of support_plane()'s ring, of its points within PLANE_TOLS of
    `tol` of the plane -- so a cluttered ring does not count -- or None with too few."""
    def grown(radius):
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel) > 0

    inner, outer = (image_px(n, mask.shape) for n in PLANE_RING_PX)
    pts = cam[grown(outer) & ~grown(inner) & (cam[..., 2] > 0)]
    n, p = plane
    r = (pts - p) @ n
    r = r[np.abs(r) < PLANE_TOLS * tol]
    if len(r) < MIN_BOX_POINTS:
        return None
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def _ring_plane(pts, rng, tol):
    """support_plane()'s RANSAC on the ring's points: (normal, point), the
    normal facing away from the camera, or None."""
    if len(pts) < MIN_BOX_POINTS:
        return None
    best_count, best = 0, None
    for _ in range(PLANE_RANSAC_ITERS):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        count = int(np.sum(np.abs((pts - a) @ n) < PLANE_TOLS * tol))
        if count > best_count:
            best_count, best = count, (n, a)
    if best is None or best_count < PLANE_MIN_INLIERS * len(pts):
        return None
    n, a = best
    inliers = pts[np.abs((pts - a) @ n) < PLANE_TOLS * tol]
    centre = inliers.mean(0)
    n = np.linalg.svd(inliers - centre, full_matrices=False)[2][2]
    return (n if n @ centre > 0 else -n), centre


def min_obb(points, tol=None):
    """Minimal-volume oriented box: (R, centre, extent), R's columns the box axes.

    Each convex-hull face normal is tried as one axis, the smallest rectangle
    of the points projected across it (rotating calipers) giving the other two:
    the usual approximation of the exact minimum.

    Volume alone is not enough. One view usually shows two faces of a part, an
    L in cross-section, and a box flush with the L's diagonal encloses it in
    the same volume as the box flush with its faces (as for a right triangle);
    sampling noise then picks one, and the diagonal is 30 degrees out. So among
    near-smallest boxes the one whose faces carry the most points wins: the
    surfaces the camera saw are the part's faces, the diagonal carries none.
    """
    try:
        hull = points[ConvexHull(points).vertices]
        normals = ConvexHull(hull).equations[:, :3]
    except Exception:
        # Coplanar points (a flat part seen face on): qhull cannot build a 3D
        # hull, and the one axis worth trying is the plane's normal.
        hull = points
        normals = np.linalg.eigh(np.cov((points - points.mean(0)).T))[1][:, :1].T
    boxes = []
    for n in np.unique(np.round(normals, 4), axis=0):
        n = n / np.linalg.norm(n)
        a = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        _, _, angle = cv2.minAreaRect((hull @ np.column_stack([a, b])).astype(np.float32))
        u = np.cos(np.radians(angle)) * a + np.sin(np.radians(angle)) * b
        R = np.column_stack([u, np.cross(n, u), n])
        proj = hull @ R
        boxes.append((np.prod(proj.max(0) - proj.min(0)), R))
    smallest = min(volume for volume, _ in boxes)

    if tol is None:
        # A depth image's points lie about a pixel's span apart: TOL as pixel_tol() makes it.
        some = points[:: max(1, len(points) // 2000)]
        spacing = cKDTree(some).query(some, k=2)[0][:, 1] if len(some) > 2 else np.array([0.002])
        tol = BOX_FACE_SPACINGS * float(np.median(spacing)) * np.sqrt(len(some) / max(1, len(points)))

    def on_faces(R):
        proj = points @ R
        lo, hi = proj.min(0), proj.max(0)
        return np.mean(np.minimum(proj - lo, hi - proj).min(1) < PLANE_TOLS * tol)

    R = max((R for volume, R in boxes if volume <= smallest * (1 + BOX_VOLUME_SLACK)),
            key=on_faces)
    proj = points @ R
    lo, hi = proj.min(0), proj.max(0)
    return R, R @ ((lo + hi) / 2), hi - lo


def observe(mask, metres, K, rng=None):
    """What one frame shows of one object, or None if too little.

    Returns (cam, mask, keep, plane, note): every pixel's camera-frame point,
    the mask cut to pixels with depth, the mask pixels that are the object
    rather than its support, the support plane (normal, point) or None, and
    how keep was chosen.
    """
    cam, valid = backproject(metres, K)
    mask = mask & valid
    if mask.sum() < MIN_BOX_POINTS:
        return None
    keep, note = mask, "no clear support surface; raw mask"
    # The support is found within the depth's noise, measured on the ring round
    # the part, and at least the pixel's own (pixel_tol()): a rendered depth's
    # noise of nothing found no plane under a tester half behind a box, its
    # bench within 0.5 mm of flat (2026-11-18). How far its points really lie
    # off it is then measured about it (depth_noise()), and that cuts it.
    tol = max(noise_tol(metres, plane_ring(mask) & valid) or 0.0, pixel_tol(metres[mask], K))
    plane = support_plane(mask, cam, valid, rng or np.random.default_rng(0), tol)
    if plane is not None:
        tol = view_tol(metres[mask], K, depth_noise(mask, cam, plane, tol))
        n, p = plane
        in_front = mask & ((cam - p) @ n < -PLANE_TOLS * tol)
        if in_front.sum() >= max(MIN_BOX_POINTS, FLAT_KEEP_FRACTION * mask.sum()):
            keep, note = in_front, "support surface removed"
        else:
            note = "flat on its support; raw mask"
    return cam, mask, keep, plane, note


def fit_box(mask, metres, K, rng=None, view=None):
    """Box of one object on one frame: (T_cam_box, extent, note), or None.

    T_cam_box's rotation is the box axes and its translation the box centre.
    """
    view = view or observe(mask, metres, K, rng)
    if view is None:
        return None
    cam, mask, keep, _, note = view
    R, centre, extent = min_obb(cam[keep])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, centre
    return T, extent, f"{int(keep.sum())} of {int(mask.sum())} mask px, {note}"


# Model fitting. A mesh point is judged by what the camera sees along its line
# of sight, so an occluded part costs nothing while a wrong pose still does:
#
#   * hidden: the camera sees something nearer (a hand, the gripper, another
#     part). Nothing was seen there, so nothing speaks against it.
#   * open: the camera sees that far or further. It must see the object, in
#     the mask and at the mesh point's depth. Seeing past it -- the bench
#     behind where the mesh says the part is -- counts against the pose.
#     Outside the mask, a surface at the mesh point's depth and off the
#     support does not: it is the part where the mask stops short of an
#     occluder touching it (a gripper finger pressing on it), or that
#     occluder. It counts as hidden. The bare support there still counts
#     against the pose: it is how a thin part's too-big mesh shows.
#   * below the support surface counts against it too. A box tilted over the
#     part's hidden side shows itself that way, or as open points in the air.
#
# The observed points, for their part, must lie on the mesh's visible surface.
MODEL_SAMPLES = 3000
# Mesh points and observed points per candidate pose in the coarse search, and
# its ICP rounds; the best MODEL_KEEP are then refined with every point.
MODEL_COARSE = (1000, 600, 8)
MODEL_FINE_POINTS = 3000
MODEL_FINE_ROUNDS = 30
MODEL_KEEP = 5
# Candidate headings about the support normal, one per this many degrees.
MODEL_YAW_STEP_DEG = 30
# Lengths here are not millimetres but multiples of each view's own unit, its
# tolerance TOL: the residual a point may have and still be the part's, where
# every cost saturates. The view sets it, measured on it: TOL_SIGMAS times the
# depth noise the support around the part shows (depth_noise()), or the part
# itself where there is no support (image_noise()), and at least the pixel's
# own spread at the part's depth (pixel_tol()); a fit's adds the mesh's own
# error (fit_tol()). So no camera needs calibrating, and no part's size is
# assumed. (On BOP's test images, Kinect and Primesense, the noise came out
# 1.3-2.5 mm (median), up to 3.9.)
# TOL_SIGMAS: the two-sided 99 % quantile of a normal spread -- a point of the
# part off by noise alone lies within it but once in a hundred. A confidence,
# not a length: the same for any part, camera or scene. (Until 2026-11-18 2.6,
# with TOL_PX = 3.2 pixels' span: set so the first scenes kept the 6 mm they
# had.)
TOL_SIGMAS = 2.576
# Where the depth's own noise is not to be measured, the pixel's: a point
# stands for its pixel's footprint, z/f across, and where on it the surface was
# seen is uniform over it -- sqrt(1/12) of it along each image axis,
# PIXEL_SIGMA across both. TOL_SIGMAS of that (1.05 px) is pixel_tol().
PIXEL_SIGMA = float(np.sqrt(2 / 12))
# min_obb()'s faces carry the points within this many of the points' own
# spacing of them: a sampling's band, not a sensor's.
BOX_FACE_SPACINGS = 3.2
# A fit's TOL also holds for the mesh's own error, which grows with the part
# (a mesh is off its part by a share of its size, not by millimetres), and for
# the mask's registration, a pixel or two. The mesh's error is that of where
# the meshes come from: a demonstration's reconstruction, scaled as one -- the
# fit stretches it only where the view measures it -- lies off its part by a
# median of MESH_ERROR_P95 of the part's diagonal at the 95th percentile
# (tester 4.6 %, ring 2.8, bar 1.7, sample 1.5: the four supplied meshes against
# the simulated parts' shapes at their best single scale, 2026-11-17). A TOL
# is three sigmas where a 95th percentile is two: TOL_PER_P95 of it. So a fit's
# TOL is the largest of TOL_SIGMAS of the noise, the pixel's own spread
# (pixel_tol()) and MESH_ERROR_P95 * TOL_PER_P95 of the part's diagonal as the
# view shows it (fit_tol()).
# (2026-11-20: fixed at 2.0 %, the meshes' quality as agreed with upstream --
# its refined models measure 1.1-2.1 % -- not a median of the meshes at hand.)
MESH_ERROR_P95 = 0.02
TOL_PER_P95 = 1.5
FIT_TOL_SIZE = MESH_ERROR_P95 * TOL_PER_P95
# Where the client does not say, the mesh's own fits do: of each fine fit the
# (fit_model_box()), the 95th percentile of the observed points'
# distances to its surface, less what the depth's noise accounts for, as a
# share of its diagonal (model.fitted_mesh_error), kept by the mesh's id
# (LOCATE_KNOWN_SCALES of them), and said where their median is more than
# MESH_ERROR_P95 -- but not fitted at. Fitted at it (2026-11-18: each mesh
# deformed 4.5 % of its diagonal past what scale and stretch take out, the
# tester's learnt at 4.2-4.9 %, its truth 4.7), the parts silently wrong went
# from 12 of 44 to 20, those refused from 11 to 7: a wider TOL does not let a
# rough mesh tell poses or ways round apart, it only lets the wrong ones in.
# A refusal, a coarse fit or a doubt says so; the client, told, can send a
# truer mesh or its error (mesh_error, which it then answers for).
# That is the default for a demonstration's reconstruction; how far a mesh is
# off its part is the client's to know -- a CAD model is exact, another
# reconstruction rougher -- and it says so per mesh, meta["models"][i]
# ["mesh_error"] (its 95th percentile as a share of the mesh's diagonal), which
# the mesh's own TOL then takes (ObjectModel.tol_share). Nothing in the mesh
# itself tells it: its surface's roughness, 0.2-0.6 % of the diagonal on the
# four supplied meshes, says nothing of their 1.5-4.6 % off their parts.
# That mesh part is what a mesh may be off by, not what it is: once the pose
# search has a pose, the fit's TOL narrows to TOL_SIGMAS of its residuals' own
# spread where that is under NARROW_TOL_MIN of it (narrow_tol()) -- an exact
# mesh (a CAD model) is then judged at the sensor's TOL, a rough one keeps
# what it needs. Only narrowed, never widened by a fit off the part.
NARROW_TOL_MIN = 0.9
# Observed depth this far in front of a mesh point hides it.
MODEL_HIDDEN_TOLS = 4 / 3
# The first ICP round pairs points this far apart; later rounds narrow it.
MODEL_SEARCH_TOLS = 5.0
# The mask is grown this much before a mesh point counts as outside it.
MODEL_MASK_GROW_PX = SEGMENT_EDGE_PX
# Mask spill can also include a hand in front of the part. Only select a
# depth-connected component when it clearly dominates the foreground. Two
# neighbouring pixels are of one surface where the surface between them would
# be turned no more than MODEL_DEPTH_GAP_DEG from facing the camera: a pixel
# spans z/f, so their depths differ by at most z/f * tan(it) (depth_gap_share()).
# Turned further, the depth camera no longer sees a surface but its edge. (3 cm
# at 0.6 m and f = 348, as the scenes it was set on had; with no focal length
# known, MODEL_DEPTH_GAP_SHARE of the depth, the same there.)
MODEL_DEPTH_GAP_DEG = 86.7
MODEL_DEPTH_GAP_SHARE = 0.05
MODEL_COMPONENT_MIN_SHARE = 0.7
# A piece of the mask not touching the main one is kept where what lies
# between them in the image is nearer than both (MODEL_BEHIND_SHARE of it, by
# MODEL_HIDDEN_TOLS) -- one part seen round an occluder -- and their facing edges
# could be one part's: apart in depth by no more than MODEL_BEHIND_DEPTH_RATIO
# times their gap across the view, plus MODEL_DEPTH_GAP_SHARE of their depth. A bar seen 36 deg
# off end-on with a ring on it showed its far end as a separate 110 px piece,
# dropped, and was fitted 221 mm long of 340. A piece touching the main one,
# a depth jump between them, is still dropped: a hand.
MODEL_BEHIND_MIN_PX = 10
MODEL_BEHIND_SHARE = 0.7
MODEL_BEHIND_DEPTH_RATIO = 3.0
# Share of point-to-point in the ICP metric; the rest is point-to-plane, which
# alone lets a flat face slide along itself.
MODEL_POINT_WEIGHT = 0.1
# The final descent on the cost: step sizes, coarse to fine, and at most this
# many moves at each.
MODEL_POLISH_STEPS_TOLS = (2 / 3, 1 / 3, 1 / 6, 1 / 12)
MODEL_POLISH_STEPS_DEG = (4.0, 2.0, 1.0, 0.5)
# With a measured scale, the descent also scales the mesh about its centre by
# these fractions: re-fit from the rigid pose, a scale search alone stops a
# few percent short of the cost's own minimum.
MODEL_POLISH_STEPS_SCALE = (0.02, 0.01, 0.005, 0.0025)
MODEL_POLISH_ROUNDS = 6
# A move must lower the cost this much: less is the depth noise talking.
MODEL_POLISH_MIN_GAIN = 0.002
# The mesh gives the part's shape, not its size: a demonstration's
# reconstruction comes out at whatever size it does -- ring_insertion's bar
# at about twice the simulated bar, pick_placement's tester and sample at
# 0.41 times theirs. One uniform scale is searched over this range, first on
# a grid this coarse (ratio between neighbours), then about its best in steps
# this fine. A best scale at either end of the range means the shape does not
# match, and the fit is refused.
# (A fallback only: each mesh's own range is its scale_range, the scales that
# make its longest side a part's size, MODEL_EXTENT_RANGE_M -- a factor of a
# mesh's size says nothing of the part's, whatever units it came in.)
MODEL_SCALE_RANGE = (0.25, 4.0)
# A mesh sent with its scale known (meta["models"][i]["known_scale"]: a CAD
# model, a measured reconstruction) is fitted at that scale alone, neither
# measured nor stretched: the size is the strongest evidence of which proposal
# is the part, and a shape fitted at a free size also fits a patch of a larger
# part or a smaller look-alike.
MODEL_SCALE_COARSE_RATIO = 1.08
MODEL_SCALE_FINE_RATIO = 1.015
# The coarse grid first spans this factor either way of a start's size.
MODEL_SCALE_BRACKET = 2.0
MODEL_SCALE_ROUNDS = 10
# Nor are its proportions exact: the simulated tester is 1.44 times as wide as
# it is deep, its reconstruction 1.09. So the mesh may also be scaled along
# each of its own box axes, by up to this factor either way of its overall
# scale, where that lowers the cost by MODEL_STRETCH_MIN_GAIN. A pose search
# also starts from the proportions of the observed box, when they differ from
# the mesh's by more than MODEL_SCALE_START_RATIO.
# (Measured: the four supplied meshes' proportions are off their simulated
# parts' by up to 1.39 -- the tester, 2.80 along its length against 2.01
# across, sim_alignment's calibration, 2026-11-17. A mesh's proportions may be
# off by that much, and no more is taken.)
# (2026-11-19: fixed at 1.5 for every mesh, as agreed with upstream -- its
# refined models' proportions are off by 1.11-1.40 -- and no longer learnt
# from a mesh's fits: that took an axis the support set, not the view, for a
# measured one, and widened the sample's to 1.80; and on the 21 scenes which
# way round a near-equal pair went out first moved with the bound one way and
# the other (1.29, 1.4, 1.5), safety alike. A client that knows better says so
# per mesh, stretch_max.)
MODEL_STRETCH_MAX = 1.5
# That is the default for a demonstration's reconstruction; the client says
# so per mesh where it knows better, meta["models"][i]["stretch_max"] -- 1 for
# a CAD model, whose proportions are the part's -- and the mesh's own
# (ObjectModel.stretch_max) takes its place wherever a mesh is stretched,
# turned a quarter round or located by its proportions.
# Where the client does not say, the mesh's fits do: a fine fit whose
# proportions the view measured within a step (MODEL_STRETCH_STEP) of the
# bound -- not one that only wanted past it, a stretch the bound stopped
# (MODEL_STRETCH_EDGE): that is not taken, so no measured proportion ever
# shows more than the bound, so it is not that -- widens that mesh's bound a
# step past what it measured for
# its later fits (the server's learnt_stretch, by the mesh's id), up to 4.
# Never narrower than MODEL_STRETCH_MAX: a reconstruction's proportions are
# off by what its own fits show, at least.
MODEL_STRETCH_MIN_GAIN = 0.03
MODEL_STRETCH_START_RATIO = 1.15
# The proportions are fitted together with the pose too, by Gauss-Newton over
# both (_ModelFit.stretch): the descent, one axis a step at a time, moves a
# scale only a few percent from where it starts. The simulated bar, 340 by
# 27 mm, has its reconstruction's length (386 mm) scaled 1.27 times its
# width's (39 mm): at one scale it came out 268 mm long on a view of 334 mm,
# the points past its ends 32 mm off it, and was refused; the descent took
# its length from 0.696 to 0.703 of the mesh's. A round changes a scale by at
# most MODEL_STRETCH_STEP. Its pairs narrow to MODEL_STRETCH_RADIUS_TOLS, not to
# one TOL: past that, the points beyond a short mesh's ends no longer
# draw it out, and the bar stopped 6 mm short at each.
MODEL_STRETCH_ROUNDS = 30
MODEL_STRETCH_STEP = 1.15
MODEL_STRETCH_RADIUS_TOLS = 2.0
# Within this share of its bound, a stretched scale is at it.
MODEL_STRETCH_EDGE = 0.99
# Fitted so, a scale only grows -- what the fit at one scale leaves wrong is
# a mesh too short where the view shows more: the bar's length, the ring's
# thickness; the cost leans to a mesh a little small (MODEL_SCALE_MIN_GAIN)
# -- and by MODEL_STRETCH_MAX at most: this fits the mesh's proportions, its
# size is measure_scale()'s. Unbounded, a pose at scale 1 on a patch of the
# tester grew 1.9 times along its length and 3.1 across, 152 mm long where
# the part is 194, and fitted better than the right pose. A scale that ends
# at that bound wants more, as one at the end of MODEL_SCALE_RANGE does: the
# shape does not match, and the stretch is not taken -- the tester's mesh, on
# a sample, pressed both its long sides there, flattened onto it. (The bar
# grew 1.30 times, the ring's thickness 1.15.)
# An axis is stretched only where the view spans it: the observed points
# cover at least this share of the mesh's extent along it, from their
# MODEL_SEEN_PERCENTILE-th to their (100 - it)-th percentile. A bar seen from
# one side shows half its round, and its depth stretched to fit that half:
# 0.62-0.76 of it. Between the 5th and 95th percentiles an upright tester's
# depth, which only its top face spans, came to 0.61 -- 0.88 between the 2nd
# and 98th -- and the proportions its fit had started from, 189 x 62 x 41 mm
# of the part's 194 x 62 x 42, were dropped for one scale: coarse, 48 mm wide.
# (0.8: between what a round body's far half and a face seen alone leave,
# 0.62-0.76, and what an axis seen end to end shows, 0.88 and over.)
MODEL_STRETCH_SEEN = 0.8
MODEL_SEEN_PERCENTILE = 2
# A length is measured only where the view shows both its ends: of the mesh's
# camera-facing points in the last MODEL_END_SHARE of it at either end, no
# more than MODEL_END_HIDDEN may be hidden -- something nearer in front, or
# off the image. (An end facing away, with few such points, is not hidden.)
# A hidden end costs nothing, so a mesh shortened into it fits as well: a
# tester half behind a board came out 164 mm long, not 182, and a sample's
# scale 1.69, not 2.44. So the mesh is stretched only along lengths whose
# ends show; and a size whose two longest lengths do not both show is still
# used -- it is the best there is, and a ring's arc gets its size right half
# hidden -- but not counted as measured, so a later, clearer view measures it
# again (MODEL_UPGRADE_INTERVAL_SEC).
MODEL_END_SHARE = 0.1
MODEL_END_HIDDEN = 0.3
# And a length is measured only where the part is seen to stop at its ends:
# MODEL_END_PROBE_TOLS past the end's camera-facing points, no more than
# MODEL_END_HIDDEN of those in the image lie behind something nearer --
# outside the part's own mask, off the support. A mesh fitted short into an
# occluder ends just in front of it, its own end in view: the bar seen nearly
# end-on, a ring on it, came out 221 mm of 340 and measured (ends_open()).
# Past the image's edge is no evidence either way -- the RI bar's end 2 cm in
# from it was measured right -- unless all but a few probes are.
MODEL_END_PROBE_TOLS = 5.0
# A size the view could not measure is still at least what was seen: along
# each of its axes the mesh spans the observed points (2nd to 98th
# percentile), less this for depth noise. A hidden end costs nothing, so half
# behind a board a sample came out 48 mm wide where 60 mm of it showed; its
# scale is raised until it covers what showed. (A measured size stands: a
# reconstruction's probe a little short is no reason to grow the whole part.)
MODEL_COVER_NOISE_TOLS = 2 / 3
# Nor can a view measure how far a part resting on its support stands off it
# where it sees only its top: a plate's thickness from above, a tester lying
# on its side seen with neither end of its height. The fit keeps the mesh's
# proportion there, and a reconstruction's is not the part's: PP-S4-1's
# sample, 3 mm thick, came out 10.8 mm, 5.8 mm into the bench and 1.4 mm over
# its top; PP-S1-3's tester, 62 mm high on its side, 56 mm, its bottom 6.3 mm
# clear of the bench (2026-11-05: placed 15 mm off where it was planned). So
# along the box axis that stands within MODEL_DOUBT_APART_DEG of the
# support's normal, where the view does not measure it (axes_seen()), a part
# resting there -- its mesh's bottom off the support by no more than the axis
# could be wrong by, MODEL_STRETCH_MAX of it, and the tolerance -- is as high
# as the view sees its top: its observed points' height over the support at
# their 98th percentile, less what depth noise adds there, 2.054 times the
# noise depth_noise() measures (the normal distribution's 98th percentile);
# its bottom on the support. Where the support is not the part's (held in
# the air, on another part), the bottom is far off it and nothing changes.
# A fit refused for going into the support is tried so too: a plate's mesh 4
# times as thick as the part went 9 mm into the bench, and was refused.
MODEL_SUPPORT_TOP_PERCENTILE = 98
MODEL_SUPPORT_TOP_SIGMAS = 2.054
# Where a part lies along its length only its ends tell: a bar's round body
# fits anywhere along it. A hand round one end of RR-S1-2's bar and the ring
# on it hid that end, the length went unmeasured (255 mm of 340), and the bar
# was placed 41-59 mm along its length off the part; RI-S1-1's ring went in
# 40 mm too deep (2026-11-05). So an end counts only where the view shows it:
# not hidden (hidden_ends()) and seen to stop (ends_stop()), or resting on
# the support. Where one end is seen and its mesh's end is further from the
# observed points' end there (the tenth point from it: as few as tell where an
# end is, hidden_ends()) than the tolerance, the mesh is slid along its
# length onto it; and what the ends tell is replied (model_axial): which are
# seen, where the mesh's ends are and how far the part was seen to reach,
# and where none is seen, how far along its length the mesh could slide and
# still span what was seen. A part round about its length -- a bar -- is
# slid so on every frame its carried fit is checked on too, and the reply
# says what that frame's view tells (carried_axial()): neither the tracker
# nor the correction holds a round body along its length, and the live bars
# went 40-59 mm off where the init's view had put them within 3 mm.
# Rotations refined from each start of the pose search, beside the best
# MODEL_KEEP of all: at its own size a mesh can sit snugly on a patch of a
# part twice as big, cheaper than any start at the part's size.
MODEL_KEEP_PER_START = 2
# And about which axis a mesh is round: a turn about it that moves the mesh's
# surface less than the mesh's own error (FIT_TOL_SIZE of its diagonal, the
# part of a fit's TOL the mesh sets) is one no view can tell from the pose.
# Turned about a box axis through its box's centre, by every
# MODEL_SYMMETRY_STEP_DEG up to a half turn, nine in ten of its points
# (MODEL_SYMMETRY_PERCENTILE) must lie within that of its surface, and 99 in
# 100 (MODEL_SYMMETRY_TAIL_PERCENTILE) within twice it -- a feature on one side
# few points fall on, a mug's handle, tells the turn. A share of the mesh's
# size, so it is the shape's, at any size: in millimetres, a box 31 mm long
# came out round. Stretched unevenly across the axis, a mesh is round about it
# no longer where the stretch moves its surface by more than that; a block
# with rounded corners, 1.21 times as wide one way as the other, is not round,
# nor is a cross-section square (its corners turned 45 deg). (Rings, bars and
# cans come out round; the tester, its probe 3.6 mm off under a turn, is not.)
# The fit cannot tell such a turn from the pose, and the tracked turn about
# the axis drifted: 9.1 deg over ten frames of one still image, where nothing
# else moved. About such an axis the published turn is held (bboxes()): each
# reply turns the pose about it as little as lies between it and the last one
# sent, the first with another of the mesh's box axes as near the image's up as
# it goes. Neither is a turn about it any doubt (model_spin_doubt).
MODEL_SYMMETRY_STEP_DEG = 20
MODEL_SYMMETRY_PERCENTILE = 90
MODEL_SYMMETRY_TAIL_PERCENTILE = 99
# Which way round a part lies. Turned onto its own box -- half round about a
# box axis, or a quarter round about one whose other two sides are within
# MODEL_DOUBT_SIDES of each other -- a mesh not the same so (ObjectModel.box_turns,
# MODEL_DOUBT_SAME_SHARE), nor turned about the axis it is round about, is
# refitted there (settle_turns()), and so is the pose itself, alike. The best
# fit of them, then the cheapest, is taken; any as good a fit within the
# margin below either way is sent in doubt with it (model_doubts), for the
# person to say which. (Until 2026-11-17 the pose the search found was kept
# unless another was cheaper by the margin: having had the whole search, it
# was the cheaper one of near-equals as a rule, and which one that was hung on
# where the search started -- a pixel band 0.2 px wider turned 10 of the 44
# parts. The view's own evidence is no different for it: of the 21 scenes'
# parts, those right or with the right way round in doubt stayed 32 and 33.) What tells one way round from
# another is often a feature few points fall on -- a probe, a groove, a notch --
# which the view need not show, and tests of the image or the mask's outline
# made for one part's features told others the wrong way round; the cost alone
# decides, and where it cannot, says so. Two ways round it cannot tell apart by
# less than what the two leave unexplained together -- each its share of
# observed points off the mesh and of the mesh's points the view contradicts --
# for that much a mesh off the part can be off either way: the tester's
# reconstruction, its probe off the real one, cost 0.054 more turned end for end
# seen whole and 0.068 less turned the wrong way in PP-S1-3. Nor by less than
# MODEL_TURN_COST_SLACK, the depth's noise: half behind PP-S4-1's occluder,
# 0.4779 against 0.4780.
MODEL_TURN_COST_SLACK = 0.02     # only where a cost carries no cost_se


def turn_margin(a, b):
    """How much cheaper (cost) one way round must be than another for the view
    to tell them apart, `a` and `b` their terms (see MODEL_TURN_COST_SLACK):
    what the two leave unexplained together -- each its share of observed
    points off the mesh and of the mesh's points the view contradicts -- and
    what the better of them still costs within the TOL. A point within the TOL
    is as the mesh's error and the noise leave it: the TOL is that wide so
    the right pose fits a mesh off its part, and a way round turned on a mesh
    so off can cost that much less or more, the error falling elsewhere. On a
    clean view (2026-11-18: the support cut by the depth's noise, not a
    pixel's span), the sample turned over fitted its top face's bulk 0.3 mm
    better at the median and its notch and edges far worse -- 4.6 % of its
    points over 3 mm off against 0.6 % -- all within its 6.5 mm TOL, and
    nothing unexplained either way: told apart by 0.04 of cost, it went out
    turned over with no doubt, on 8 of the 11 scenes."""
    left = lambda w: 1 - w["fitness"] + w["contradiction"]
    within = lambda w: max(0.0, w["cost"] - left(w) - w.get("penetration", 0.0))
    # Nor by less than another draw of points could move the two costs apart:
    # TOL_SIGMAS of their difference's standard error (each cost's, cost_se) --
    # about 0.02 on 3000 points, the floor this was before (2026-11-18).
    if "cost_se" in a and "cost_se" in b:
        floor = TOL_SIGMAS * float(np.hypot(a["cost_se"], b["cost_se"]))
    else:
        floor = MODEL_TURN_COST_SLACK
    return max(floor, left(a) + left(b) + min(within(a), within(b)))
# Refitted to within MODEL_DOUBT_APART_DEG of the pose or of one already in
# doubt -- but for a turn that leaves the part as it was -- it is that one.
# A quarter turn is a way round to try where the two sides it swaps are within
# what a mesh's proportions may be off by (MODEL_STRETCH_MAX): further apart,
# no stretch the fit allows makes one fit where the other was.
MODEL_DOUBT_SIDES = MODEL_STRETCH_MAX
MODEL_DOUBT_APART_DEG = 30.0
# The same so: turned, its samples no further from its surface -- the
# MODEL_DOUBT_SAME_PERCENTILE-th percentile of their distances, so a feature on
# a thousandth of it counts -- than as many samples drawn afresh from it come out
# on any of MODEL_DOUBT_SAME_DRAWS draws. A ring turned over or a bottle about
# its length is so; a plate's notched end, 1.4 times as far, is not. And so is
# one turn after another that is so: each measured alone, a turn just past the
# floor can be what two just under it make -- RR-S1-1's ring came out the same
# turned over about one diameter and turned half round about its hole, not
# turned over about the diameter square to that one, which those two are, and
# that went out in doubt (2026-11-04). The turns that are so are taken as all
# they make, one after the other (_box_turns()).
MODEL_DOUBT_SAME_PERCENTILE = 99.9
MODEL_DOUBT_SAME_DRAWS = 8
# Each half turn about a box axis -- the three, any two making the third --
# is refitted and polished, its own translation found, whatever its first
# refit costs: on PP-S4-2's tester, 41 % behind the box, the right way round,
# half round about the box's short axis from the fit, cost 0.336 polished
# against the fit's 0.314 -- in doubt by the margin, 0.10 -- but was dropped
# before polishing for costing more than 0.354, and the fit went out 180 deg
# and 23 mm off with no way round in doubt the right one (2026-11-05). A
# quarter turn, where one is tried, still is only where its first refit is
# near. And every way round that fits at all, in doubt or not, is a
# hypothesis of the binding (Binding): beyond the margin it is not sent in
# doubt (model_doubts), but is in model_binding, "in_doubt": false, for the
# views' evidence to bring into doubt and the person to choose.
MODEL_DOUBT_HALF_TURNS = True
# The pose search runs at scale 1, and also at the observed size when that
# is this far from the mesh's: a mesh twice a bar's length fits anywhere
# along it, and scaled about its centre afterwards it leaves the part.
MODEL_SCALE_START_RATIO = 1.3
# The scale is measured only on a view that hides no more than this: a smaller
# mesh slid into the hidden part can fit as well as the right one. On rendered
# scenes up to half hidden, the ring's scale came out within 1%; for the
# pick_placement parts it came out nearer 1 than the truth, and was then kept
# at 1 by MODEL_SCALE_MIN_GAIN, never applied wrongly.
MODEL_SCALE_MAX_HIDDEN = 0.5
# Hidden more, a part takes the other parts' scale where the init gives one
# (fit_model_box()'s scale_hint; see LOCATE_SCALE_AGREE): PP-S4-1's tester,
# 51% behind the box, kept the start it was searched at, 2.94, and came out
# 235 mm long of 182; the sample beside it measured 2.40.
# A measured scale replaces 1 only if it lowers the cost by this much. Too big
# a mesh shows at once, as open points off the part; too small a one costs
# only the few points past its edges, so noise alone can favour a slightly
# smaller one. A mesh that is the part's size so keeps exactly 1, which the
# agent's demonstration grasps require of a native mesh pose.
MODEL_SCALE_MIN_GAIN = 0.03
# A part's shape -- its scale along each of its mesh's box axes -- measured on
# an earlier view (LOCATE_KNOWN_SCALES) is fitted on each later one too, once
# that view's own fit has settled which way round the part lies, about its
# centre, and kept unless the view's own shape costs this much less: a rigid
# part's size is its own, whichever side it shows. Turned over in
# PP-S3-2 (2026-11-02), the tester showed the camera its underside, which the
# reconstruction has only roughly; its fit came out 154-158 mm long of 182,
# measured, both ends seen, its centre 12 mm along its length off the part's,
# and the placement carried that: four runs 14-15 mm off. On that view the
# shape measured upright cost less (0.473 against 0.490; p95 6.0 mm against
# 9.5) and was 1 mm off along its length, but no start of the pose search was
# near it. Not before the ends are settled: the probe under the body, at the
# full length no test saw it, and the colour and the mask turned the part
# end for end, 20 mm off. Nor with MODEL_SCALE_MIN_GAIN's lean to the shape
# held, the lower cost deciding, as between the pose search's starts: first
# seen turned over, the tester held its 154 mm on the upright view after,
# 0.467 against its own 0.441, and would have held it from then on.
MODEL_KNOWN_SHAPE_MIN_GAIN = 0.0
# A part hidden more than MODEL_SCALE_MAX_HIDDEN whose fit passes the coarse
# acceptance neither at its start nor at the other parts' scale is fitted once
# more at the scale, along its box axes, it was located at this init: the
# mesh was fitted to this view then. RR-S1-2's bar, half behind the fist
# holding it, its end showing below it: the reconstruction's 386 x 39 mm,
# thicker for its length than the part, at the ring's 0.80 reached neither
# end, at 1 was too thick (contradiction 0.39); it was located at 0.83 along
# its length and 0.66-0.92 across.
MODEL_LOCATED_FALLBACK = True
# Retry failed localization on fresh frames, with one background fit at a time.
MODEL_RETRY_INTERVAL_SEC = 3.0
# The init's fit of a part takes up its refined starts one after another,
# each measured and stretched in full, until one is a fit; where none is, it
# went through them all: RI-S4-1's bar, a box in its mask, 26.8 s to be refused;
# the init waited 31 s for it (2026-11-20). So at the init no further start is
# taken up past MODEL_INIT_FIT_BUDGET_SEC from the fit's start: the best so far
# goes out -- coarse, or refused, and so retried on later frames
# (MODEL_RETRY_INTERVAL_SEC), where the search runs in full; or fine, but then
# not certain (model_uncertain_why "search cut short") and retried as well
# (fit_rank()): the starts not tried may hold a better pose, however fast or
# slow the machine and however many parts. A latency the
# Agent waits on, the same for any part; the fits of a scene's parts run side
# by side (FIT_SIDE_BY_SIDE), most within 3-8 s.
MODEL_INIT_FIT_BUDGET_SEC = 10.0
# A coarse fit, or one whose size the view hid too much to measure, is fitted
# again on later views this often, and replaced only by a better one -- fine
# over coarse, measured over not. A fine, measured fit is kept: the part's
# size is the part's, and a later view half hidden cannot measure it better.
MODEL_UPGRADE_INTERVAL_SEC = 6.0
# A better one turned MODEL_REFIT_TURN_DEG or more from the fit published --
# half round about its length, or end for end -- is no refinement but the
# other way round, where views a little apart did not agree: the fit is kept,
# and that pose sent in doubt with it (model_spin_doubt, model_ends_doubt)
# for the person to say which. Taken, it would turn the part under the Agent
# between two replies: PP-S1-4's tester came out end for end either way
# round on SAM2's mask and on one cut by its colour, its retries settling it
# on a profile 0.12 against 0.13, and a client running the scene saw its z
# reversed against the replay's (2026-10-30).
# A fine refit over a coarse one may improve size and alignment, but may not
# publish a half turn just because its geometric fit ranks higher. First
# check the refined shape in the currently published direction on this
# frame. Upgrade only if that direction itself passes the fine-fit checks;
# keep the other direction in doubt (preserve_refit_direction()).
MODEL_REFIT_TURN_DEG = 90.0
# Kept, a fit is carried by the tracked motion, and the motion can go wrong
# with the track not lost: a ring lifted face-on in the gripper was tracked
# through its hole, on the camera's ray to it but 147 mm too deep (the far
# background seen through the hole pulled its depth), the mesh with it, and
# the Agent saw the gripper holding nothing. So every frame the carried mesh
# is scored on what the camera sees there, on MODEL_CHECK_POINTS (mesh,
# observed) points, and a frame where the observed points are off it
# (carried_fit_rejection()) is replied without it: held up, the ring's track
# then jumped 217-240 mm off on one frame in three to five, the frames
# between them right, and every one of those frames went out. Off it on
# MODEL_CHECK_FAILS frames running, the fit is no longer the part's where
# the tracker carries it, and it is fitted again at once. The points are
# as many as the fit's acceptance had: at a third of the mesh's, seen ones
# lie sparse on a thin part, and RR-S1-2's bar, fitted at p95 8.7 mm on
# all, came out 17.8 mm on 1000 and 600 -- off the part on the frame it was
# fitted on. All of them cost 2 ms a frame more.
MODEL_CHECK_POINTS = (MODEL_SAMPLES, MODEL_FINE_POINTS)
MODEL_CHECK_FAILS = 2
# The tracked motion is not the part's once its points are gone: the robot
# turning a part over leaves the tracker's points on the side now underneath,
# and on the frames after it set the part down the motion moved a part lying
# still. On PP-S3-1 (2026-11-05, 08:44:49-08:45:40) a refit came back right
# on the frame it was fitted on, was carried to the next by that motion and
# was off the part there (fitness 0.04-0.09, 95th percentile 56-59 mm), was
# turned another way round to fit, withheld, dropped and fitted again, every
# 13 s for 30 s and more, the pose sent 108 deg off between. So the carried
# fit is not all a frame is checked against: off the part where its motion
# carries it, it is tried where it was last seen on the part (a part lying
# still: still_motions), and where the robot says it set it down (a pose
# prior, meta["priors"]), each corrected on the frame as the carried pose is;
# the first of them on the part is kept, the pose held there. Where the
# motion's pose and the held one are further apart than the correction moves
# a pose (correct_max_m(), MODEL_CORRECT_MAX_DEG), or once it is held,
# both are corrected and the cheaper kept -- the held one while it costs no
# more than the motion's
# by MODEL_CORRECT_MIN_GAIN: on the same test, the pose the motion carried,
# corrected, passed the check 8-16 deg off a tester lying still, and was
# taken over the held one on as little. The motion carries it again once the
# held pose is off the part: the robot moved it. And a refit is not sent before
# it is on the part on the frame it is sent with: carried by the motion from
# the frame it was fitted on, where it was fitted, or at the prior, the
# cheapest on the part (adopt_refit()); on none, it is not taken.

# Before it is scored, the carried mesh pose is corrected on the frame's
# depth (correct_carried_pose()): MODEL_CORRECT_ROUNDS ICP rounds from it,
# pairs within MODEL_CORRECT_RADIUS_TOLS, kept where they lower the fit's cost
# by MODEL_CORRECT_MIN_GAIN. A small part with little texture gives the
# tracker few points to hold it: of a still scene's frames, RR-S1-1's ring
# turned by up to 9 degrees, RI-S4-2's by 5.7, where its depth had not
# moved. Around the pose the fit found, the cost is flat to a degree or a
# few millimetres, and where ICP settles in that flat depends on where it
# starts; so the correction is kept from frame to frame, in the mesh's frame
# so it rides along with the tracked motion, and each frame starts from it,
# not from the tracked pose: a still part then stays still. Where that puts
# the mesh more than correct_max_m() / MODEL_CORRECT_MAX_DEG from the
# tracked pose, the tracked pose is corrected too, and kept unless the
# carried correction fits by MODEL_CORRECT_MIN_GAIN better: RR-S1-1's ring's
# tracked pose jumped 14 degrees on one frame in three, its depth not at all.
MODEL_CORRECT_ROUNDS = 5
MODEL_CORRECT_RADIUS_TOLS = 2.0
MODEL_CORRECT_MIN_GAIN = 0.002
# The move that matches that turn: a part turned MODEL_CORRECT_MAX_DEG about its
# centre moves its far points by sin(it) of half its diagonal (correct_max_m()),
# so a correction is no refinement by either measure alike, at any size.
MODEL_CORRECT_MAX_DEG = 10.0


# The surface seen of each object, kept in its frame at the init frame: every
# SURFACE_EVERY-th frame its mask's points, support removed, are carried back
# there by its tracked motion and binned into voxels. A part's size is
# measured once and kept; so is what was seen of it, so a part half hidden
# now still shows what earlier views saw. Voxels seen on fewer than
# SURFACE_MIN_SEEN frames are left out once there have been that many: depth
# noise, or a hand that passed in front once.
# A voxel is SURFACE_VOXELS_ACROSS-th of the part's diagonal (as first seen;
# the mesh's, in its archive): a box's surface is at most twice its diagonal
# squared, so at that size the whole of it fits SURFACE_MAX_VOXELS, a part of
# any size kept as finely. (Until 2026-11-18 3 mm: four voxels across a
# 12 mm part, 6000 of them not a third of a 0.4 m one.)
SURFACE_EVERY = 2
# The tracking front end's gates on a registration's mean residual -- the
# only way an object is no longer lost, and whether its map grows -- and on a
# new map point's spread of observations, in metres, were one number for any
# part at any depth (6 and 8 mm, set on a RealSense held in the hand). Each
# object's are set (frontend_gates()) on every frame its surface is recorded:
# TOL_SIGMAS of what a tracked point's position is off by there -- the depth's
# noise on the part (image_noise()) with the tracker's own error across,
# TRACK_ERROR_PX at the part's depth (z/f). 6.2 mm on the 21 scenes' parts.
TRACK_ERROR_PX = 1.0
SURFACE_PIXELS = 1500
SURFACE_MAX_VOXELS = 6000
SURFACE_VOXELS_ACROSS = int(np.sqrt(SURFACE_MAX_VOXELS / 2))
SURFACE_MIN_SEEN = 2
# Replies carry each object's surface this often (in frames), and on the first.
SURFACE_PUBLISH_EVERY = 5

# Which way round a fitted mesh lies on its part -- its binding -- is one fact
# for as long as the part is tracked: the tracker carries the part's motion,
# and every way round rides on that one motion. So the ways round the init
# view could not tell apart (model_doubts) are kept as hypotheses (Binding),
# and every BINDING_EVERY-th frame each is scored on the view until one is
# the part's: no new pose search, only new evidence, gathered while the part
# turns in the hand or the camera moves.
#
# What tells two ways round apart is where they differ: the samples of one
# farther than a TOL from the other's surface -- a probe at one end, a
# handle on one side. Such a sample seen by the camera at its depth on the
# part speaks for its way round; seen through -- the camera sees past it, or
# the bare support there -- against it (sample_evidence()). Seen through
# counts however hidden the feature is on later frames: a frame that looked
# through the place where one way round has its probe rules that out for
# good (the free space, carved). A sample keeps the verdict of most of the
# frames it was seen on, and a frame is looked at only where the part has
# moved against the camera, since the last one looked at, by more than the
# carried correction may move it on its own (correct_max_m(),
# MODEL_CORRECT_MAX_DEG): one still image seen again, its pose re-corrected
# within the cost's flat, re-counted the samples at the edges, and PP-S1-3's
# tester, the right way round left in doubt by the init's view, was ruled
# out after five frames of it.
#
# The evidence is counted in what is independent of the rest, not in
# samples: a feature -- the samples where one way round has what the other
# has not, joined where within MODEL_FRAGMENT_GAP spacings of each other, as
# a mesh's pieces are; under MIN_BOX_POINTS samples, below what the fit
# tells -- gives one vote on each frame looked at, the verdict of most of
# its samples seen there (seen on: one tolerance's area of it or more).
# Counted in samples, a reconstruction a little off the part -- its scale,
# its probe -- had hundreds of them wrong together, and ruled the right way
# round out at odds of 1e6 to 1e21: PP-S3-2's tester, its probe in view, and
# PP-S1-4's (2026-11-05). A vote is wrong as often as a sample on what the
# ways round agree on is seen through (Laplace's rule of succession over
# them: the mesh's error, the tracking's, the mask's). And beside the two
# ways round there is a third: that the mesh does not match the part where
# they differ, every vote then a coin's toss. One way round beats another
# when the odds of the votes for it against the other and that third reach
# BINDING_CONFIDENCE, on votes from SURFACE_MIN_SEEN frames looked at or
# more: one view tells nothing a mesh's error might not, and a vote is worth
# at most twice a toss against the third, so several independent votes are
# needed. A reconstruction whose probe is off the part's has
# its probe seen through the right way round too, and then the third wins.
#
# A way round beaten is dropped; the one left is the binding, published from
# then on without doubts -- turned to it where it was not the one published,
# with a new fit id, so the Agent sees the change. A part at rest on its
# support, no robot pixels on it, also rules out a way round it could not
# rest in (rest_margin()): its samples' centre outside the polygon of those
# within the tolerance of the support, by more than the tolerance, where
# another way round has its centre inside its own -- on SURFACE_MIN_SEEN
# views, as the features' evidence, not on one. Leaning on something
# else, every way round is outside, and nothing is ruled out. The person can
# settle it too, naming a way round (meta["bind"]) -- or the Agent, on its own
# look at the camera's view, or the robot, of a part it set down: each bind
# says whose it is ("by", "person" where it does not), and so does the
# binding's "by" from then on. Each reply says how far
# the binding is (model_binding), and where each way round in doubt has the
# features that would tell it, up to BINDING_FEATURE_PX image points of them.
BINDING_EVERY = SURFACE_EVERY
BINDING_CONFIDENCE = 0.95
BINDING_FEATURE_PX = 60
# Two ways round turned end for end ("ends", turn_kind()) differ where the
# part's two ends differ -- a notch at one end, a probe -- and that can be
# under what the evidence above counts. PP-S1-1's sample has a notch through
# it at one end, 8.6 x 35.7 mm as fitted, on a plate 1.5 mm thick: 6-7 of
# its 3000 samples lie farther than a TOL from the other way round's
# surface (a feature takes MIN_BOX_POINTS), and in the image the notch is
# 3.9 px deep, inside MODEL_MASK_GROW_PX's band. The ways round cost about
# the same, and the published one had the notch at the wrong end
# (2026-11-07). So every frame looked at also compares the two silhouettes
# (the meshes' triangles, each way round corrected on the frame as for the
# evidence above) with the mask, on the pixels only one of the two covers:
# such a pixel in the mask, at that one's depth, is for it; one off the
# mask, or seen past it, is against it -- unless something nearer hides it,
# or there is no depth there (ends_evidence()). Off the mask is against it
# however near the depth: the bench seen through a notch in a thin plate is
# within the tolerance of the plate's face. Each way round is corrected on its
# own, and one a pixel off along a long side puts a strip a pixel wide all one
# way -- RI-S1-1's bar, 36 px along its top edge outvoted its ends 120 to 47
# for the way round with the shorter cone at the tip (the view shows the
# longer), where without the strip the ends split 51 to 44: a strip along an
# outline is what the view's own mismatch makes, and is counted as such.
# The pixels told one way more than the other must be more than the view's
# own mismatch could make them, for a vote, counted with the features'
# (BINDING_CONFIDENCE): where the two silhouettes agree, the one way round's
# silhouette and the mask still differ by some pixels along every pixel of its
# outline -- the mesh's own error, the mask's, the pose's; along the outline of
# where the two differ, as many again could fall either way by those alone.
# So a vote takes a margin over that, plus ENDS_SILHOUETTE_SIGMAS of the
# count's own spread, sqrt(told) (ends_evidence()). Nothing in it is the
# part's or the scene's; it replaced a share of the silhouettes' area, a share
# for one way round and a band along the mask's edge, each set on these parts. On the
# init's view, a way round the silhouettes vote for over the published one
# is published in its place, the published one kept in doubt: one view
# decides nothing, but of near-equal fits the one the view favours goes
# first. PP-S1-1's sample: 26 to 9 and 24 to 8 px (2.6% and 2.3% of its
# 1371) for the two ways round with the notch where the image has it, the
# published one turned to one of them; its tester and RI-S1-1's bar kept as
# fitted. (Those counts, judged against the view's own mismatch since
# 2026-11-17, mostly fall within it: of 23 votes on the 21 scenes, 4 stand.)
ENDS_SILHOUETTE_SIGMAS = 2.0
# A feature too thin for the silhouettes still moves where the part ends:
# PP-S1-3's tester lies on its side, its probe 1-2 px thick and about 10 px
# long in the image. The silhouettes split 48 to 23 and 43 to 24 for the
# published way round, its probe at the wrong end (2026-11-07): the probe
# pixels are its mask's edge, and a strip along a long side outvoted the
# rest. Where the part ends along the mesh's long box axis does not: the
# published mesh's probe stuck 22 mm past the part seen at one end, and the
# part seen, its probe, ran 27 mm past the mesh at the other. So each pair of
# ways round turned end for end is also judged by its ends (ends_extent()):
# each way round, at its own corrected pose, how far each of its two ends
# along its long axis lies from where the part seen ends there -- the tenth
# point in from the end, as _ModelFit.axial() measures it -- summed over the
# ends neither of the two has hidden (hidden_ends()); one way round's sum
# less than the other's by ENDS_EXTENT_SHARE of the part's length or more is
# a vote for it, counted as the silhouettes' are, and on the init's view it
# goes first as they do. A share, not a length: the same for a part near or
# far, large or small.
# (Twice the mesh's own error, MESH_ERROR_P95, the two ends each off by theirs;
# a mesh's declared error in its place, ObjectModel.tol_share.)
ENDS_EXTENT_SHARE = 2 * MESH_ERROR_P95
# Where the server keeps an archive (--archive), a binding settled by the
# evidence or the person files the surface seen of the part under it --
# SurfaceMemory's, carried into the mesh's frame, within the mesh's box grown
# by two TOLs -- under the mesh's id, with its measured shapes
# (LOCATE_KNOWN_SCALES), when the session ends; a voxel's runs are counted.
# On later inits, in later runs, the shapes are known from the start. The
# surface is the part's as seen, in the mesh's frame, for whoever reads the
# archive; it is not evidence for a binding: each frame's tracking error
# spreads it a few millimetres either side of the part's, past a TOL
# at its edges, and as the mesh's samples on a rendered tester turning (the
# points farther than the tolerance from the mesh, seen on two runs) it stood
# out past the part's outline the right way round, was seen through, and
# settled the next run the wrong way.

# Published motions are smoothed: on a still scene the pipeline's pose hops
# among a few near-equal registration solutions, ~1 cm and ~1 deg apart. Each
# frame moves the published pose a fraction of the way to the new one, that
# fraction growing with the step -- SMOOTH_MIN_ALPHA for a step well inside
# SMOOTH_FULL_M / SMOOTH_FULL_DEG, all the way (no lag) for one past them, so
# real motion is followed at once. Only the reply is smoothed; the meshes,
# surfaces and the pipeline itself keep the raw motion. 1.0 turns it off.
# A step's length is judged against the part's size, SMOOTH_FULL_SHARE of its
# box's diagonal: a registration's hops go with the part -- the ~1 cm seen
# on parts about 0.18 m across, three of them the 3 cm this was -- and a 1 cm
# part moving a centimetre a frame is not jitter. SMOOTH_FULL_M only where
# there is no box. An angle is the same for a part of any size.
# Those are only where a part has no history yet: once it has
# SMOOTH_HISTORY_MIN raw steps of its own (of the last SMOOTH_HISTORY), a
# step is judged against its own jitter -- the scatter of one raw step from
# the one before it (steady motion cancels out), per axis as a robust sigma --
# and one past the 99 % radius of that (chi, three axes) is motion, followed
# at once; within it, jitter, smoothed. A part whose pose never hops is not
# smoothed at all; one that hops by a centimetre is, by that centimetre,
# whatever its size.
SMOOTH_HISTORY = 10
SMOOTH_HISTORY_MIN = 4
SMOOTH_RADIUS_99 = 3.368                # scipy.stats.chi(3).ppf(0.99)
SMOOTH_MIN_ALPHA = 0.3
SMOOTH_FULL_SHARE = 1 / 6
SMOOTH_FULL_M = 0.03
SMOOTH_FULL_DEG = 3.0

# A track request is answered in time even when its frame is not: on
# PP-S1-4 (2026-11-05, from 07:57:52) a frame took longer than the client's
# 15 s, every request after it timed out, and the trial ended 35 s on with no
# reply; RI-S1-2 (2026-11-01) and RI-S4-1 (2026-11-02) stalled so too. What
# took the time is not known: no refit runs on the request's thread (a
# background fit in mesh_fit_executor slowed the request's own checks from 83
# to 151 ms a frame on a rendered tester, 2026-11-05, not to seconds), and
# nothing waits on a lock. So a client that says how long it waits
# (meta["timeout_ms"]) has its track requests worked on another thread, and
# is answered by SERVER_REPLY_SHARE of that whether the frame is done or not:
# not done, with the last frame's reply, marked "stale" -- its frame_id that
# frame's -- and a request that comes while a frame is still worked on is
# answered so at once, its frame dropped. The share leaves the rest of the
# allowance to the transfer and the client's own work. A frame worked on
# longer than the whole allowance has the stack it is in printed, once per
# allowance, for what holds it. A client that says nothing is answered as
# before, when the frame is done.
SERVER_REPLY_SHARE = 0.5

# GPU memory still allocated after a release is held by something an earlier
# session left behind. A fresh process frees all but 8-16 MiB, through inits,
# tracks on the track thread, losses and inits again (PP-S1-1, 2026-11-06).
# The Gazebo trial of 2026-11-06 had 6519 MiB allocated after every release
# on this 8 GB card. Its init then reached 8415 MiB allocated and 10960 MiB
# reserved, and the excess spilled into system memory: the locate took
# 226-276 s (25 s fresh), the init 694 s (55 s fresh), and frames 1.4-3.6 s
# (0.78 s fresh), and the client timed out on the init. The trace found it:
# the autograd graphs of TAPIR's keyframe query features, made on the track
# thread with grad on (no_grad_thread()).
# Over GPU_HELD_WARN_MIB so is said at every release. With --trace-gpu-memory
# the allocator also keeps the Python stack of each live block, and the
# warning lists where the held blocks were allocated (GPU_TRACE_SITES of
# them), how much of the held memory Python objects reach and through what
# (GPU_TRACE_HOLDERS of the paths to it from the server, a module's globals
# or a thread's stack) -- memory Python does not reach is held from C++, such
# as a thread's autocast cache. Every
# GPU_TRACE_EVERY_FRAMES track frames the memory is logged too, and the
# allocation sites are listed again each time the memory grows by another
# GPU_HELD_WARN_MIB over the init's.
GPU_HELD_WARN_MIB = 256
GPU_TRACE_EVERY_FRAMES = 20
GPU_TRACE_SITES = 12
GPU_TRACE_HOLDERS = 12
# How far gpu_python_holders() walks, and how much of each path it names.
GPU_TRACE_VISIT = 3_000_000
GPU_TRACE_PATH = 3
# A frame of the stack is "ours", and names the site, in these.
GPU_TRACE_OURS = (os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                  f"{os.sep}sam2{os.sep}", f"{os.sep}tapnet{os.sep}")


def gpu_alloc_sites(limit=GPU_TRACE_SITES):
    """(MiB in live blocks, [(MiB, site)] largest first): torch's snapshot
    of the caching allocator's live blocks, each by where it was allocated --
    the innermost frames of ours (GPU_TRACE_OURS), else of any code."""
    sites = collections.Counter()
    for segment in torch.cuda.memory._snapshot().get("segments", []):
        for block in segment.get("blocks", []):
            if block.get("state") != "active_allocated":
                continue
            frames = block.get("frames") or []
            ours = [f for f in frames if any(part in f["filename"] for part in GPU_TRACE_OURS)]
            site = " <- ".join(f"{os.path.basename(f['filename'])}:{f['line']} {f['name']}"
                               for f in (ours or frames)[:4]) or "(allocated before the trace began)"
            sites[site] += block["size"]
    return sum(sites.values()) / 2**20, [(size / 2**20, site) for site, size in sites.most_common(limit)]


def kept_model_mib():
    """GPU memory of the models kept across pipelines (keep_models): held
    on purpose, not by an earlier session (GPU_HELD_WARN_MIB)."""
    models = [*Sam2RealTimeSegmenter._kept.values(), *TapirTracker._kept.values()]
    tensors = [t for m in models for t in (*m.parameters(), *m.buffers()) if t.is_cuda]
    return sum(t.numel() * t.element_size() for t in tensors) / 2**20


def no_grad_thread():
    """A worker thread's start: grad mode off, as on the main thread. Grad
    mode is per thread, and the pipeline's modules turn it off only on the
    thread that builds them (TapirTracker's constructor); on the track
    thread every keyframe's TAPIR query features kept their autograd graph,
    ~720 MiB each, and the 7.9 GB of them outlived every release (2026-11-06,
    found by --trace-gpu-memory)."""
    torch.set_grad_enabled(False)


def gpu_children(obj):
    """(name, child) of what `obj` refers to, for gpu_python_holders()'s
    walk: items, indices, attributes; other referents by their type."""
    if isinstance(obj, (str, bytes, bytearray, int, float, complex, bool, type(None), np.ndarray)):
        return []
    if isinstance(obj, dict):
        return [(f"[{str(k)[:40]!r}]" if isinstance(k, str) else f"[{type(k).__name__}]", v)
                for k, v in list(obj.items())]
    if isinstance(obj, (list, tuple, collections.deque)):
        return [(f"[{i}]", v) for i, v in enumerate(list(obj))]
    if isinstance(obj, (set, frozenset)):
        return [("{}", v) for v in list(obj)]
    if isinstance(obj, types.FrameType):
        return [(f" {obj.f_code.co_name}().{k}", v) for k, v in list(obj.f_locals.items())] + \
               ([(" caller", obj.f_back)] if obj.f_back is not None else [])
    named = []
    try:
        named = [(f".{k}", v) for k, v in list(vars(obj).items())]
    except TypeError:
        pass
    slots = getattr(type(obj), "__slots__", ())
    for slot in (slots,) if isinstance(slots, str) else slots:
        if hasattr(obj, slot):
            named.append((f".{slot}", getattr(obj, slot)))
    if named:
        return named
    return [(f"<{type(r).__name__}>", r) for r in gc.get_referents(obj)]


def gpu_python_holders(server, limit=GPU_TRACE_HOLDERS):
    """(MiB of CUDA storage Python objects reach, MiB of it in tensors made
    with grad on (a grad_fn), [(MiB, path)] largest first, up to `limit`): each
    storage under its shortest path from the
    server, a module's globals or a thread's stack (breadth first, up to
    GPU_TRACE_VISIT objects), cut to its first GPU_TRACE_PATH steps."""
    gc.collect()
    with warnings.catch_warnings():
        # Touching deprecated module attributes (torch.distributed.reduce_op) warns.
        warnings.simplefilter("ignore")
        return _gpu_python_holders(server, limit)


def _gpu_python_holders(server, limit):
    storages, graphed = {}, set()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                ptr = obj.untyped_storage().data_ptr()
                storages.setdefault(ptr, obj.untyped_storage().nbytes())
                if obj.grad_fn is not None:        # made with grad on; parameters are not
                    graphed.add(ptr)
        except Exception:
            continue
    names = {threading.get_ident(): None}
    names.update({t.ident: t.name for t in threading.enumerate()})
    roots = [("server", server)]
    roots += [(f"thread {names.get(ident, ident)}:", frame) for ident, frame in sys._current_frames().items()
              if ident != threading.get_ident()]
    roots += [(name, module) for name, module in list(sys.modules.items()) if module is not None]
    found, seen = {}, {id(storages)}
    queue = collections.deque((obj, (name,)) for name, obj in roots)
    while queue and len(seen) < GPU_TRACE_VISIT:
        obj, path = queue.popleft()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, types.ModuleType):
            if len(path) > 1:
                continue                    # walked from sys.modules itself
            seen.add(id(vars(obj)))
        if torch.is_tensor(obj):
            try:
                if obj.is_cuda:
                    found.setdefault(obj.untyped_storage().data_ptr(), path)
            except Exception:
                pass
            continue
        try:
            queue.extend((child, path + (name,)) for name, child in gpu_children(obj) if id(child) not in seen)
        except Exception:
            continue
    held = collections.Counter()
    for ptr, size in storages.items():
        path = found.get(ptr)
        held["".join(path[:GPU_TRACE_PATH + 1]) if path else
             "(not from the server, a module or a thread's stack: through an autograd graph, or a cycle)"] += size
    return (sum(storages.values()) / 2**20, sum(storages[p] for p in graphed) / 2**20,
            [(size / 2**20, path) for path, size in held.most_common(limit)])


class PoseSmoother:
    def __init__(self):
        self.last = None
        self.raw = None
        self.steps = {}

    def reset(self, motions=None):
        self.last = None if motions is None else np.array(motions, dtype=np.float64)
        self.raw = None if motions is None else np.array(motions, dtype=np.float64)
        self.steps = {}

    def jitter(self, index):
        """(m, deg): the 99 % radius of part `index`'s jitter (see
        SMOOTH_HISTORY), or None with too little history."""
        steps = self.steps.get(index) or []
        if len(steps) < SMOOTH_HISTORY_MIN:
            return None
        t = np.array([s[0] for s in steps])
        w = np.array([s[1] for s in steps])
        dt, dw = np.diff(t, axis=0), np.diff(w, axis=0)
        sigma = lambda d: 1.4826 * float(np.median(np.abs(d - np.median(d, axis=0)))) / np.sqrt(2.0)
        return SMOOTH_RADIUS_99 * sigma(dt), SMOOTH_RADIUS_99 * np.degrees(sigma(dw))

    def __call__(self, motions, sizes=None):
        """`sizes`: each part's box diagonal (m), or None (SMOOTH_FULL_SHARE)."""
        motions = np.asarray(motions, dtype=np.float64)
        if (self.last is None or self.last.shape != motions.shape
                or not np.all(np.isfinite(self.last))):
            self.reset(motions)
            return motions.copy()
        out = motions.copy()
        raw = self.raw if self.raw is not None and self.raw.shape == motions.shape else motions
        for index, (prev, cur) in enumerate(zip(self.last, motions)):
            if not np.all(np.isfinite(cur)):
                continue
            # The raw step from the last raw pose, for the part's jitter.
            if np.all(np.isfinite(raw[index])):
                step_w = cv2.Rodrigues(raw[index][:3, :3].T @ cur[:3, :3])[0].ravel()
                history = self.steps.setdefault(index, [])
                history.append((cur[:3, 3] - raw[index][:3, 3], step_w))
                del history[:-SMOOTH_HISTORY]
            dR = prev[:3, :3].T @ cur[:3, :3]
            w = cv2.Rodrigues(dR)[0].ravel()
            dt = cur[:3, 3] - prev[:3, 3]
            measured = self.jitter(index)
            if measured is not None:
                full_m, full_deg = measured
            else:
                size = sizes[index] if sizes is not None and index < len(sizes) else None
                full_m, full_deg = (SMOOTH_FULL_SHARE * size if size else SMOOTH_FULL_M), SMOOTH_FULL_DEG
            step = max(np.linalg.norm(dt) / max(full_m, 1e-9),
                       np.degrees(np.linalg.norm(w)) / max(full_deg, 1e-9))
            alpha = float(np.clip(step, SMOOTH_MIN_ALPHA, 1.0))
            out[index, :3, :3] = prev[:3, :3] @ cv2.Rodrigues(alpha * w)[0]
            out[index, :3, 3] = prev[:3, 3] + alpha * dt
        self.last = out
        self.raw = motions.copy()
        return out.copy()


class SurfaceMemory:
    """One object's seen surface: a voxel's mean point, on how many frames,
    and the mean direction it was seen from (toward the camera). The
    direction says which side of the part a point is on: the top of a thin
    plate is nearer its reconstruction's bottom than its top."""

    def __init__(self):
        self.voxels = {}
        self.frames = 0
        self.voxel = None                    # SURFACE_VOXELS_ACROSS; set on the first points

    def add(self, points, frame_id, camera=None):
        """Points of one frame, in the object's init-frame coordinates, seen
        from `camera` there (the camera's origin; None leaves no direction)."""
        if not len(points):
            return
        self.frames += 1
        if self.voxel is None:
            across = float(np.linalg.norm(np.ptp(points, axis=0)))
            self.voxel = max(across, 1e-4) / SURFACE_VOXELS_ACROSS
        keys, inverse = np.unique(np.floor(points / self.voxel).astype(np.int64),
                                  axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        sums = np.zeros((len(keys), 3))
        np.add.at(sums, inverse, points)
        means = sums / np.bincount(inverse, minlength=len(keys))[:, None]
        if camera is None:
            views = np.zeros_like(means)
        else:
            views = np.asarray(camera, dtype=float) - means
            views /= np.maximum(np.linalg.norm(views, axis=1, keepdims=True), 1e-9)
        for key, mean, view in zip(map(tuple, keys), means, views):
            seen = self.voxels.get(key)
            if seen is None:
                self.voxels[key] = [mean, 1, frame_id, view]
            else:
                seen[0] = (seen[0] * seen[1] + mean) / (seen[1] + 1)
                seen[3] = seen[3] * seen[1] + view
                seen[1] += 1
                seen[2] = frame_id
                seen[3] = seen[3] / max(np.linalg.norm(seen[3]), 1e-9)
        if len(self.voxels) > SURFACE_MAX_VOXELS:
            # The least seen, then the longest unseen, go first.
            order = sorted(self.voxels, key=lambda k: (self.voxels[k][1], self.voxels[k][2]))
            for key in order[:len(self.voxels) - SURFACE_MAX_VOXELS]:
                del self.voxels[key]

    def points(self, views=False):
        """The kept surface, (N, 3) in init-frame coordinates; with `views`,
        also each point's direction toward the camera, (N, 3)."""
        least = min(SURFACE_MIN_SEEN, self.frames)
        kept = [(mean, view) for mean, count, _, view in self.voxels.values() if count >= least]
        points = np.array([mean for mean, _ in kept]).reshape(-1, 3)
        if not views:
            return points
        return points, np.array([view for _, view in kept]).reshape(-1, 3)


def encode_points(points):
    """Points as base64 of little-endian float16 (x, y, z)*: a fraction of a
    millimetre over a part's size, a sixth of JSON's bytes."""
    return base64.b64encode(np.asarray(points, "<f2").tobytes()).decode("ascii")


def fit_rank(model):
    """How settled a fitted model is: (fine, size measured, its search run
    in full -- not cut short at the init, MODEL_INIT_FIT_BUDGET_SEC)."""
    return (getattr(model, "fitted_quality", None) == "fit", bool(getattr(model, "fitted_measured", False)),
            not getattr(model, "fitted_search_cut", False))
# Acceptance: share of observed points on the mesh, of open mesh points the
# camera disagrees with, and of mesh points under the support surface. On
# rendered scenes with 2 mm depth noise, up to 65% hidden, the right mesh
# scored fitness 0.84-0.98 and contradiction 0.00-0.02; the other
# demonstration part's mesh mostly scored below 0.8 or above 0.1 -- though
# with most of a part hidden, what is left can fit both.
MODEL_MIN_FITNESS = 0.8
MODEL_MAX_CONTRADICTION = 0.1
MODEL_MAX_PENETRATION = 0.1
# A pose short of those is not thrown away: an imperfect reconstruction can
# fit no better and still say which way the part lies and where it is. Within
# these it is published as a coarse fit, marked so, for the Agent to use as a
# prior and measure the rest; below them the mesh does not fit at all.
MODEL_COARSE_MIN_FITNESS = 0.6
MODEL_COARSE_MAX_CONTRADICTION = 0.25
MODEL_COARSE_MAX_PENETRATION = 0.2
# A candidate must also explain the observation's extent: a good fit to a
# small surface patch must not hide a long tail of distant object points.
MODEL_MAX_P95_DISTANCE_TOLS = 2.0
MODEL_COARSE_MAX_P95_DISTANCE_TOLS = 4.0
# And within a share of the part's own width (its box's middle side), down
# to a floor that depth noise alone stays under: millimetres say little about
# a small part. Stretched along its axes, pick_placement's tester mesh lay on
# its 24 mm sample within 8.4 mm at the 95th percentile -- a third of the
# sample's width -- where the right meshes, imperfect as they are, keep
# within a fifth of theirs.
MODEL_MAX_P95_SHARE = 0.3
MODEL_P95_FLOOR_TOLS = 0.75
MODEL_COARSE_MAX_P95_SHARE = 0.5
MODEL_COARSE_P95_FLOOR_TOLS = 1.0
# A mesh outside this size (largest box side, metres) is in other units.
MODEL_EXTENT_RANGE_M = (0.005, 1.0)
# A reconstruction can leave loose fragments round the part: a sloth's specks,
# the furthest 102 mm off its body, made its box 242 mm long of 198, and no
# proposal was the part's size. Surface samples are joined where within MODEL_FRAGMENT_GAP
# spacings of each other; what that leaves under MODEL_FRAGMENT_SHARE of the
# area goes from the fit and the box (not from the mesh's id). Meshes in many
# pieces that touch, as reconstructions' are, are one part so.
MODEL_FRAGMENT_GAP = 3.0
MODEL_FRAGMENT_SHARE = 0.02


def _twist_spread(points, box):
    """Per box axis, how far the mesh comes from itself turned about it
    through the box's centre: (the MODEL_SYMMETRY_PERCENTILE-th percentile of
    its points' distances from its surface, the
    MODEL_SYMMETRY_TAIL_PERCENTILE-th), the most over the turns
    MODEL_SYMMETRY_STEP_DEG apart up to a half turn; at the mesh's own size."""
    tree = cKDTree(points)
    B, c = box[:3, :3], box[:3, 3]
    spread, tail = np.zeros(3), np.zeros(3)
    for k in range(3):
        for deg in range(MODEL_SYMMETRY_STEP_DEG, 181, MODEL_SYMMETRY_STEP_DEG):
            turned = (points - c) @ _rotation(np.radians(deg) * B[:, k]).T + c
            near, far = np.percentile(tree.query(turned)[0],
                                      [MODEL_SYMMETRY_PERCENTILE, MODEL_SYMMETRY_TAIL_PERCENTILE])
            spread[k], tail[k] = max(spread[k], float(near)), max(tail[k], float(far))
    return spread, tail


def _box_turns(points, box, extent, tri, area, sides=MODEL_DOUBT_SIDES):
    """The turns of the 24 onto a box's own axes (in the box's frame), but
    the identity, that take the box onto itself (MODEL_DOUBT_SIDES): (those
    that do not take the mesh (`tri`, of `area`) onto itself, those that do;
    MODEL_DOUBT_SAME_PERCENTILE) -- those that do with every one they make,
    one after another."""
    tree = cKDTree(points)
    B, c = box[:3, :3], box[:3, 3]
    floor = 0.0
    for seed in range(1, MODEL_DOUBT_SAME_DRAWS + 1):
        rng = np.random.default_rng(seed)
        face = rng.choice(len(tri), len(points), p=area / area.sum())
        r1, r2 = np.sqrt(rng.random(len(points))), rng.random(len(points))
        drawn = ((1 - r1)[:, None] * tri[face, 0] + (r1 * (1 - r2))[:, None] * tri[face, 1]
                 + (r1 * r2)[:, None] * tri[face, 2])
        floor = max(floor, float(np.percentile(tree.query(drawn)[0], MODEL_DOUBT_SAME_PERCENTILE)))
    out, same = [], []
    for P in _signed_permutations():
        if np.allclose(P, np.eye(3)):
            continue
        to = np.argmax(np.abs(P), axis=0)
        if np.any(np.maximum(extent[to], extent) > sides * np.minimum(extent[to], extent)):
            continue
        turned = (points - c) @ (B @ P @ B.T).T + c
        if np.percentile(tree.query(turned)[0], MODEL_DOUBT_SAME_PERCENTILE) > floor:
            out.append(P)
        else:
            same.append(P)
    # What one turn that leaves it as it was makes after another leaves it so.
    grew = True
    while grew and out:
        grew = False
        made = [np.eye(3)] + same
        for k, P in enumerate(out):
            if any(np.allclose(P, A @ C) for A in made for C in made):
                same.append(out.pop(k))
                grew = True
                break
    return out, same


def _without_fragments(v, f):
    """`f` less the faces of loose fragments (MODEL_FRAGMENT_GAP), itself where
    there are none. Its own draw: a mesh with none samples as it would."""
    rng = np.random.default_rng(0)
    tri = v[f]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if area.sum() <= 0:
        return f
    face = rng.choice(len(f), MODEL_SAMPLES, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(MODEL_SAMPLES)), rng.random(MODEL_SAMPLES)
    points = ((1 - r1)[:, None] * tri[face, 0] + (r1 * (1 - r2))[:, None] * tri[face, 1]
              + (r1 * r2)[:, None] * tri[face, 2])
    tree = cKDTree(points)
    pairs = tree.query_pairs(MODEL_FRAGMENT_GAP * np.sqrt(area.sum() / 2 / MODEL_SAMPLES), output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), np.uint8), (pairs[:, 0], pairs[:, 1])), shape=(MODEL_SAMPLES,) * 2)
    count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=count)
    keep = sizes >= MODEL_FRAGMENT_SHARE * MODEL_SAMPLES
    if count == 1 or keep.all():
        return f
    keep[np.argmax(sizes)] = True
    return f[keep[labels[tree.query(tri.mean(1))[1]]]]


class ObjectModel:
    """A mesh reduced to what the fit needs, in the mesh's frame and metres:
    surface samples with their normals, and the mesh's own minimal box."""

    def set_stretch_max(self, stretch_max):
        """A wider proportion bound (MODEL_STRETCH_MIN_GAIN's comment), and the
        quarter turns onto the box it allows."""
        self.stretch_max = float(min(4.0, max(1.0, stretch_max)))
        tri = self.mesh_v[self.mesh_f]          # as __init__ had them, not kept: shipped to the fit workers
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        self.box_turns, self.box_same = _box_turns(self.points, self.box, self.extent, tri, area, self.stretch_max)
        long_k = int(np.argmax(self.extent))
        self.ends_alike = any(P[long_k, long_k] < 0 for P in self.box_same)

    def __init__(self, vertices, faces, rng=None, colours=None, unit=1.0, stretch_max=MODEL_STRETCH_MAX):
        # Metres a unit of the mesh's (meta["models"][i]["unit_m"]): a mesh at
        # its part's size is at scale `unit` -- 1 for one in metres, 0.001 in
        # millimetres; nothing else here is in the mesh's units but the
        # scales and poses sent back, which are the mesh's as given.
        self.unit = float(unit)
        if not (np.isfinite(self.unit) and self.unit > 0):
            raise ValueError(f"unit_m {unit}: metres a mesh unit, over 0")
        # How far its proportions may be off its part's (MODEL_STRETCH_MAX).
        self.stretch_max = float(stretch_max)
        if not (np.isfinite(self.stretch_max) and 1.0 <= self.stretch_max <= 4.0):
            raise ValueError(f"stretch_max {stretch_max}: a factor, 1 to 4")
        rng = rng or np.random.default_rng(0)
        v = np.asarray(vertices, np.float64).reshape(-1, 3)
        f = np.asarray(faces, np.int64).reshape(-1, 3)
        if not len(v) or not len(f) or not np.all(np.isfinite(v)):
            raise ValueError("empty or non-finite mesh")
        if f.min() < 0 or f.max() >= len(v):
            raise ValueError("face index out of range")
        whole = f
        f = _without_fragments(v, f)
        # Only for what the box is said to be: None where the mesh's own vertices are.
        self.kept_vertices = None if len(f) == len(whole) else v[np.unique(f)]
        tri = v[f]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        double_area = np.linalg.norm(cross, axis=1)
        if double_area.sum() <= 0:
            raise ValueError("mesh has no area")
        # The faces kept, for the silhouette the mesh casts (mesh_depth()).
        self.mesh_v, self.mesh_f = v, f
        R, centre, extent = min_obb(v[np.unique(f)])
        lo, hi = MODEL_EXTENT_RANGE_M
        if not lo <= extent.max() * self.unit <= hi:
            raise ValueError(f"mesh is {extent.max() * self.unit:.3g} m across at {self.unit:g} m a unit; "
                             f"expected a part {lo * 1000:g} mm to {hi:g} m (unit_m: its units)")
        # The scales that make its longest side a part's size (MODEL_SCALE_RANGE).
        self.scale_range = (lo / float(extent.max()), hi / float(extent.max()))
        # Area-weighted samples; the order is random, so any prefix is too.
        face = rng.choice(len(f), MODEL_SAMPLES, p=double_area / double_area.sum())
        r1, r2 = np.sqrt(rng.random(MODEL_SAMPLES)), rng.random(MODEL_SAMPLES)
        a, b, c = tri[face, 0], tri[face, 1], tri[face, 2]
        self.points = ((1 - r1)[:, None] * a + (r1 * (1 - r2))[:, None] * b
                       + (r1 * r2)[:, None] * c)
        # The samples' colours (CIELAB), where the mesh's vertices have them (LOCATE_COLOUR_DELTA).
        self.lab = None
        if colours is not None:
            rgb = np.asarray(colours, np.float64).reshape(-1, 3)[f[face]] / 255.0
            rgb = ((1 - r1)[:, None] * rgb[:, 0] + (r1 * (1 - r2))[:, None] * rgb[:, 1]
                   + (r1 * r2)[:, None] * rgb[:, 2])
            self.lab = cv2.cvtColor(rgb.astype(np.float32)[:, None], cv2.COLOR_RGB2LAB)[:, 0].astype(np.float64)
            self.chroma_tree = cKDTree(self.lab[:, 1:])
        # Only the line a normal lies on is used, never which way it points: a
        # reconstructed mesh's faces are not reliably wound one way.
        self.normals = cross[face] / np.maximum(double_area[face, None], 1e-15)
        self.spacing = float(np.sqrt(double_area.sum() / 2 / MODEL_SAMPLES))
        self.box = np.eye(4)
        self.box[:3, :3], self.box[:3, 3] = R, centre
        self.extent = extent
        # How round the mesh is about each box axis (MODEL_SYMMETRY_STEP_DEG),
        # the turns onto its own box that leave it as it was and those that
        # do not (MODEL_DOUBT_SAME_PERCENTILE), and so whether its ends are alike.
        self.twist_spread, self.twist_tail = _twist_spread(self.points, self.box)
        self.box_turns, self.box_same = _box_turns(self.points, self.box, self.extent, tri, double_area,
                                                   self.stretch_max)
        long_k = int(np.argmax(self.extent))
        self.ends_alike = any(P[long_k, long_k] < 0 for P in self.box_same)
        used, inverse = np.unique(whole, return_inverse=True)
        self.model_id = hashlib.sha256(v[used].astype("<f4").tobytes()
                                      + inverse.reshape(-1, 3).astype("<i4").tobytes()).hexdigest()
        self.known_scale = None              # see MODEL_SCALE_RANGE; set by decode_models()
        self.tol_share = FIT_TOL_SIZE         # see FIT_TOL_SIZE; set by decode_models()
        self.fitted_pose = None
        self.fitted_scale = None
        self.fitted_linear = None
        self.fitted_quality = None
        self.fitted_measured = False
        self.fitted_search_cut = False        # its init search cut short (MODEL_INIT_FIT_BUDGET_SEC)
        # The fit turned half round about its length, relative to it, where
        # the view fits that about as well (MODEL_TURN_COST_SLACK); and an id
        # of the fit, new with every fit, for what is said of it to hold to.
        self.fitted_spin_doubt = None
        self.fitted_ends_doubt = None
        self.fitted_doubts = []               # see MODEL_TURN_COST_SLACK
        self.fitted_id = None


def _pixels(P, K):
    """Pixel coordinates of camera-frame points; points behind the camera map
    far outside any image."""
    z = P[:, 2]
    zs = np.where(z > 1e-6, z, 1e-6)
    uv = P[:, :2] / zs[:, None] * K[[0, 1], [0, 1]] + K[:2, 2]
    uv[z <= 1e-6] = -1e9
    return uv


def _seen_from_camera(P, K, spacing, shape):
    """Which mesh points are in the image and its nearest surface there.

    Decided by depth rather than normals, which need not point outwards. The
    image is cut into cells two samples wide, so every cell holds some of the
    near surface; a point more than a cell's width behind the nearest point in
    its cell is on a far face.
    """
    h, w = shape
    z = P[:, 2]
    uv = _pixels(P, K)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    seen = np.zeros(len(P), bool)
    if not inside.any():
        return seen
    cell_m = 2.0 * spacing
    cell_px = max(1.0, cell_m * K[0, 0] / float(np.median(z[inside])))
    cols = int(w / cell_px) + 1
    cell = ((uv[inside, 1] / cell_px).astype(np.int64) * cols
            + (uv[inside, 0] / cell_px).astype(np.int64))
    nearest = np.full(cell.max() + 1, np.inf)
    np.minimum.at(nearest, cell, z[inside])
    seen[inside] = z[inside] <= nearest[cell] + cell_m
    return seen


def _rotation(w):
    return cv2.Rodrigues(np.asarray(w, np.float64).reshape(3, 1))[0]


def _skew(a):
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def round_axis(model, scale):
    """The box axis (index) a mesh at `scale` -- one, or three along its box
    axes -- is round about, or None; see MODEL_SYMMETRY_STEP_DEG."""
    s = np.broadcast_to(np.asarray(scale, np.float64), (3,))
    best = None
    for k in range(3):
        sides = [j for j in range(3) if j != k]
        # The mesh's own error at this scale, and how far an uneven stretch
        # across the axis moves its surface under a quarter turn.
        error = getattr(model, "tol_share", FIT_TOL_SIZE) * float(np.linalg.norm(model.extent)) * _overall(s)
        across = s[sides].mean()
        uneven = (s[sides].max() / s[sides].min() - 1) * float(model.extent[sides].max()) * across / 2
        spread = model.twist_spread[k] * across
        if (uneven <= error and spread <= error and model.twist_tail[k] * across <= 2 * error
                and (best is None or spread / error < best[0])):
            best = (spread / error, k)
    return None if best is None else best[1]


def _overall(scale):
    """The overall size of a scale: itself, or the geometric mean of three."""
    return float(np.exp(np.mean(np.log(np.asarray(scale, np.float64)))))


def _describe(scale):
    """A scale for a log line."""
    if np.ndim(scale) == 0:
        return f"{float(scale):.3f}"
    return f"{_overall(scale):.3f} (along the mesh's box axes {' x '.join(f'{v:.3f}' for v in scale)})"


def _stretch_allowed(scale, scale_range=MODEL_SCALE_RANGE, stretch_max=MODEL_STRETCH_MAX):
    """Whether three per-axis scales keep to `scale_range` (the mesh's,
    ObjectModel.scale_range) and to within `stretch_max` (the mesh's,
    ObjectModel.stretch_max) of their overall size."""
    f = np.asarray(scale, np.float64)
    lo, hi = scale_range
    ratio = f / _overall(f)
    limit = max(float(stretch_max), 1.0) * (1 + 1e-9)
    return bool(np.all((f >= lo) & (f <= hi))
                and np.all((ratio >= 1 / limit) & (ratio <= limit)))


def _within_stretch(scale, scale_range=MODEL_SCALE_RANGE, stretch_max=MODEL_STRETCH_MAX):
    """Three per-axis scales brought within `scale_range` (the mesh's) and within
    `stretch_max` (the mesh's) of their overall size, their order kept; None
    if that does not settle."""
    f = np.asarray(scale, np.float64)
    lo, hi = scale_range
    limit = max(float(stretch_max), 1.0)
    if limit <= 1 + 1e-9:
        # Proportions exact: one size along every axis.
        f = np.full(3, float(np.clip(_overall(f), lo, hi)))
        return f if _stretch_allowed(f, scale_range, limit) else None
    for _ in range(30):
        f = np.clip(f, lo, hi)
        if _stretch_allowed(f, scale_range, limit):
            return f
        overall = _overall(f)
        f = overall * np.clip(f / overall, 1 / (limit * (1 - 1e-9)), limit * (1 - 1e-9))
    return None


def _signed_permutations():
    """The 24 rotations that map the coordinate axes onto themselves."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            P = np.zeros((3, 3))
            P[list(perm), range(3)] = signs
            if np.linalg.det(P) > 0:
                out.append(P)
    return out


def connected_depth_foreground(mask, depth, cam=None, through=None, tol=None):
    """Remove small mask islands separated by a depth jump from the main part.

    A 2D connected mask can cross an occluder's boundary. Use adjacent valid
    pixels whose depth differs by at most 3 cm as graph edges. If no component
    dominates, retain the evidence for the pose checks instead of choosing one.
    With `cam` (every pixel's camera-frame point), an island the main part
    continues into behind an occluder is kept (MODEL_BEHIND_SHARE).

    `through`, other objects' pixels cut from the mask, joins what it links
    as the mask's own pixels would: RR-S1-2's bar, cut in two by the ring
    round it, had its far piece dropped -- the ring hugs the bar, no
    MODEL_HIDDEN_TOLS nearer -- and was fitted 213 mm long of 330.
    """
    region = mask if through is None else mask | through
    gap_share = depth_gap_share(cam=cam)
    # Two neighbours' depths differ by their noise as well, sqrt(2) of it; the
    # TOL given covers TOL_SIGMAS of the noise.
    gap_noise = np.sqrt(2.0) * (tol or 0.0)
    count = int(region.sum())
    indices = np.full(mask.shape, -1, dtype=np.int32)
    indices[region] = np.arange(count)
    sources, targets = [], []
    for a, b in ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])):
        linked = region[a] & region[b] & (np.abs(depth[a] - depth[b])
                                          <= gap_share * np.maximum(depth[a], depth[b]) + gap_noise)
        sources.append(indices[a][linked])
        targets.append(indices[b][linked])
    sources, targets = np.concatenate(sources), np.concatenate(targets)
    graph = coo_matrix((np.ones(len(sources), dtype=np.uint8), (sources, targets)), shape=(count, count))
    _, labels = connected_components(graph, directed=False)
    own = mask[region]                            # the order of `indices`
    sizes = np.bincount(labels[own], minlength=labels.max() + 1 if count else 0)
    if not len(sizes) or sizes.max() < max(MIN_BOX_POINTS, MODEL_COMPONENT_MIN_SHARE * int(mask.sum())):
        return mask
    main = sizes.argmax()
    keep = labels == main
    if cam is not None:
        ys, xs = np.nonzero(region)
        main_px = np.column_stack([xs[keep & own], ys[keep & own]])
        tree = cKDTree(main_px)
        for label in np.flatnonzero(sizes >= image_area_px(MODEL_BEHIND_MIN_PX, mask.shape)):
            if tol is None:
                tol = max(noise_tol(cam[..., 2], region, (1,)) or 0.0, cam_pixel_tol(cam, region))
            if label != main and _behind_occluder(np.column_stack([xs, ys])[(labels == label) & own],
                                                  main_px, tree, depth, cam, tol, gap_share):
                keep |= labels == label
    result = np.zeros_like(mask)
    result[region] = keep
    return result & mask


def _behind_occluder(piece, main_px, tree, depth, cam, tol, gap_share=MODEL_DEPTH_GAP_SHARE):
    """Whether a mask piece (pixels, x y) is the main part's continuation
    behind something nearer; see MODEL_BEHIND_SHARE."""
    dist, nearest = tree.query(piece)
    i = int(np.argmin(dist))
    if dist[i] <= 1.5:
        return False                             # touching: a depth jump between them
    a, b = piece[i], main_px[nearest[i]]
    n = int(np.ceil(dist[i]))
    share = np.arange(1, n) / n
    between = np.round(a[None] + (b - a)[None] * share[:, None]).astype(np.int64)
    da, db = depth[a[1], a[0]], depth[b[1], b[0]]
    seen = depth[between[:, 1], between[:, 0]]
    hidden = MODEL_HIDDEN_TOLS * tol
    if not len(seen) or np.mean((seen > 0) & (seen < da + (db - da) * share - hidden)) < MODEL_BEHIND_SHARE:
        return False
    gap = cam[a[1], a[0]] - cam[b[1], b[0]]
    return abs(gap[2]) <= (MODEL_BEHIND_DEPTH_RATIO * np.hypot(gap[0], gap[1])
                           + gap_share * max(da, db))


class _ModelFit:
    """Fits one mesh to one object's view; see fit_model_box()."""

    def __init__(self, model, view, K, rng, through=None, scale_hint=None, located=None, image=None,
                 known_shape=None):
        cam, mask, keep, plane, _ = view
        self.lab = (None if image is None or model.lab is None else
                    cv2.cvtColor(np.asarray(image, np.float32) / 255.0, cv2.COLOR_RGB2LAB).astype(np.float64))
        self.scale_hint = scale_hint
        self.stretch_wanted = False           # see MODEL_STRETCH_MIN_GAIN        # The part's shape measured on earlier views (MODEL_KNOWN_SHAPE_MIN_GAIN).
        self.known_shape = None if known_shape is None else np.asarray(known_shape, np.float64).reshape(3)
        # The scale, along the mesh's box axes, it was located at this init
        # (MODEL_LOCATED_FALLBACK).
        self.located = located
        self.model, self.K, self.cam = model, K, cam
        self.depth = cam[..., 2]
        kernel = np.ones((2 * image_px(MODEL_MASK_GROW_PX, mask.shape) + 1,) * 2, np.uint8)
        self.grown = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        self.mask = mask
        self.plane = None
        ring_tol = max(noise_tol(self.depth, plane_ring(mask) & (self.depth > 0)) or 0.0,
                       pixel_tol(self.depth[mask], K))           # as observe() finds the support
        if plane is not None:
            n, p = plane
            # A plane the object is behind is an occluder the ring caught, not
            # its support; it would push the mesh out of the right pose.
            if np.mean((cam[mask] - p) @ n <= ring_tol) >= PLANE_SUPPORT_SHARE:
                self.plane = (n, p)
        # The depth's noise here, and the view's TOL it sets with the pixel's.
        sigma = None if self.plane is None else depth_noise(mask, cam, self.plane, ring_tol)
        if sigma is None:
            # No support to show it (a part held up, a box round it): on the part.
            sigma = image_noise(self.depth, mask)
        self.sigma = sigma
        # The part's size where it is known: the shape an earlier view measured,
        # else the scale it was located at, else the scale the other parts give
        # (scale_hint).
        diagonal = None
        if self.known_shape is not None:
            diagonal = float(np.linalg.norm(model.extent * self.known_shape))
        elif located is not None:
            # The scale, along its box axes, the part was located at on this view.
            diagonal = float(np.linalg.norm(model.extent * np.broadcast_to(np.asarray(located, np.float64), (3,))))
        elif scale_hint is not None:
            diagonal = float(np.linalg.norm(model.extent)) * float(np.mean(scale_hint))
        # The sensor's part of the TOL first: what is the part's, and so its
        # size, is cut with it -- the mask's spill, far off, is no size of it.
        self.tol = max(TOL_SIGMAS * (sigma or 0.0), pixel_tol(self.depth[mask], K))
        self.sensor_tol = self.tol
        self.plane_tol = PLANE_TOLS * self.tol
        # A ring's hole and mask spill can include a large patch of table.
        # For a thick enough mesh, use the foreground already isolated by
        # observe(). For very thin parts, that cut selects positive depth
        # noise and biases the fit upwards, so retain the full mask there.
        # Thin by the part's size: the mesh's thinnest side at the part's scale
        # -- known (diagonal), else as the view's points span the mesh's middle
        # side -- not in the mesh's own units, which could be anything.
        if diagonal is not None:
            part_scale = diagonal / max(float(np.linalg.norm(model.extent)), 1e-12)
        else:
            seen = cam[keep]
            if len(seen) >= MIN_BOX_POINTS:
                spread = np.sort(np.ptp((seen - seen.mean(0)) @ np.linalg.svd(seen - seen.mean(0),
                                                                              full_matrices=False)[2].T, axis=0))[::-1]
                part_scale = float(spread[1] / max(np.sort(model.extent)[::-1][1], 1e-12))
            else:
                part_scale = 1.0
        # ... against the height observe() cut the support at (view_tol()):
        # less than twice that thick, the cut would take most of it.
        cut = PLANE_TOLS * view_tol(self.depth[mask], K, sigma)
        use_foreground = (self.plane is not None
                          and model.extent.min() * part_scale > 2 * cut)
        obs_mask = keep if use_foreground else mask
        # The part is one surface in depth whether or not it rests on a support:
        # held up, its mask's spill onto the room behind, metres off and as
        # noisy as a depth camera is there, made a box 7 m long (2026-11-18).
        obs_mask = connected_depth_foreground(obs_mask, self.depth, cam, through, self.tol)
        # Where the part's size is known, nothing further from its middle than
        # its diagonal is the part's: a strip of the box behind RI-S4-1's ring,
        # 370 mm long, joined it in depth and was fitted as its (2026-11-17).
        # The middle is the kept points' median, inside the part where most
        # of them are its.
        if diagonal is not None and obs_mask.sum() >= MIN_BOX_POINTS:
            middle = np.median(cam[obs_mask], axis=0)
            near = np.linalg.norm(cam - middle, axis=-1) <= diagonal
            if (obs_mask & near).sum() >= MIN_BOX_POINTS and np.any(obs_mask & ~near):
                obs_mask = obs_mask & near
        obs = cam[obs_mask]
        # Then the mesh's part, of the part's size as what was kept shows it:
        # at most that, until the fit shows how far off its part the mesh is
        # (narrow_tol()).
        self.set_tol(fit_tol(obs, K, sigma, diagonal, getattr(model, "tol_share", FIT_TOL_SIZE)))
        self.observation_note = f"{len(obs)}/{int(mask.sum())} depth points"
        if sigma is not None:
            self.observation_note += (f", depth noise {sigma * 1000:.1f} mm"
                                      + f" (TOL {self.tol * 1000:.1f} mm)")
        if use_foreground and np.any(mask & ~keep):
            self.observation_note += ", support removed"
        if np.any(keep & ~obs_mask):
            self.observation_note += ", isolated depth components removed"
        self.obs = obs[rng.permutation(len(obs))[:MODEL_FINE_POINTS]]
        self.obs_tree = cKDTree(self.obs)
        self._obb_points = cam[obs_mask if use_foreground else keep]
        self._obb = None

    def set_tol(self, tol):
        """The fit's TOL, and the lengths in it."""
        self.tol = tol
        self.plane_tol = PLANE_TOLS * tol
        self.hidden_m = MODEL_HIDDEN_TOLS * tol
        self.cover_noise = MODEL_COVER_NOISE_TOLS * tol

    def narrow_tol(self, terms, R, t, scale):
        """The fit's TOL once the pose search has a pose and so a size: its
        mesh part, FIT_TOL_SIZE of the mesh's diagonal at that size (searched
        with what the view showed of the part, less where it was hidden), but
        no more than TOL_SIGMAS of the residuals' own spread (resid_sigma) where
        the mesh lies closer to the part than that allows for; at least the
        sensor's part. The terms at the pose, at the TOL now (None where it
        changed by less than NARROW_TOL_MIN)."""
        mesh_part = self.model.tol_share * float(np.linalg.norm(np.broadcast_to(scale, (3,)) * self.model.extent))
        spread = terms.get("resid_sigma")
        if spread is not None:
            mesh_part = min(mesh_part, TOL_SIGMAS * spread)
        tol = max(self.sensor_tol, mesh_part)
        if NARROW_TOL_MIN * self.tol <= tol <= self.tol / NARROW_TOL_MIN:
            return None
        self.set_tol(tol)
        return self.terms(scale, R, t, MODEL_SAMPLES, MODEL_FINE_POINTS)

    @property
    def obb(self):
        """The observed points' minimal box, (R, centre, extent): where the
        pose search starts (candidates(), start_scales()); scoring a given
        pose (carried_fit_rejection()) needs none, and it costs most of the
        set-up."""
        if self._obb is None:
            self._obb = min_obb(self._obb_points)
        return self._obb

    def linear(self, scale):
        """The mesh-frame linear map of a scale: one number scales the mesh
        uniformly, three scale it along its own box axes (see
        MODEL_STRETCH_MAX)."""
        f = np.broadcast_to(np.asarray(scale, np.float64), (3,))
        B = self.model.box[:3, :3]
        return (B * f) @ B.T

    def cover_seen(self, R, t, terms, scale, note, per_axis=False):
        """The pose, its scale raised, uniformly, until the mesh spans the
        observed points along each of its axes; see MODEL_COVER_NOISE_TOLS.

        With `per_axis`, where the overall size came from elsewhere (the
        other parts' scale), only the axes short of what was seen are raised,
        within MODEL_STRETCH_MAX: raised whole, PP-S4-1's tester, 2.415 by
        the sample and its mesh narrower than the part, came out 214 mm long
        of 182 to be 62 mm wide."""
        extent = np.broadcast_to(np.asarray(scale, np.float64), (3,)) * self.model.extent
        # Along each axis, of the points within the mesh's cross-section across
        # it (its extent there, and a TOL): a part longer than its mesh goes on
        # along the axis, not beside it. Off it, they are the mask's spill --
        # RR-S1-1's ring, the bar it is on in its mask at half the image's
        # resolution, was raised 1.5 times by the bar's points (2026-11-18).
        centre = R @ (self.linear(scale) @ self.model.box[:3, 3]) + t
        along = (self.obs - centre) @ (R @ self.model.box[:3, :3])
        half = extent / 2 + self.tol
        span = np.zeros(3)
        for k in range(3):
            across = [j for j in range(3) if j != k]
            inside = np.all(np.abs(along[:, across]) <= half[across], axis=1)
            if inside.sum() >= MIN_BOX_POINTS:
                span[k] = (np.percentile(along[inside, k], 98) - np.percentile(along[inside, k], 2)
                           - self.cover_noise)
        factor = float(np.max(span / extent))
        if factor <= MODEL_SCALE_FINE_RATIO:
            return R, t, terms, scale, note
        raised = np.asarray(scale, np.float64) * factor
        raised = float(raised) if np.ndim(raised) == 0 else raised
        if per_axis:
            axes = _within_stretch(np.broadcast_to(np.asarray(scale, np.float64), (3,))
                                   * np.maximum(span / extent, 1.0), self.model.scale_range, self.model.stretch_max)
            if axes is not None:
                raised = axes
        if _overall(raised) > self.model.scale_range[1]:
            return R, t, terms, scale, note
        centre = R @ (self.linear(scale) @ self.model.box[:3, 3]) + t
        t2 = centre - R @ (self.linear(raised) @ self.model.box[:3, 3])
        moved = self.terms(raised, R, t2, MODEL_SAMPLES, MODEL_FINE_POINTS)
        if moved is None:
            return R, t, terms, scale, note
        R2, t2, moved, _ = self.polish(raised, R, t2, moved)
        # What was seen past the mesh is the part only if the view takes the
        # mesh drawn over it: it fits no worse than the view can tell
        # (turn_margin()), and is no less a fit. The mask's spill is seen too:
        # RR-S1-1's ring at half the image's resolution, the bar it is on in
        # its mask, through its own plane, was raised 1.49 times, contradiction
        # 0.07 to 0.67, and lost (2026-11-18).
        rank = lambda w: 2 if not model_pose_rejection(w) else 1 if not model_pose_rejection(w, coarse=True) else 0
        # A raise that makes a refused fit one is the view's taking it: RI-S4-2's
        # ring at half the image's resolution, refused for its points 35 mm off
        # the mesh, fitted raised 1.32 times at more cost. Of two refused,
        # nothing is within what the view cannot tell: the raise must cost less
        # (RI-S4-1's ring, 77 % hidden, raised 4.41 times at 2.0 of 0.33).
        if rank(moved) != rank(terms):
            refuse = rank(moved) < rank(terms)
        else:
            slack = turn_margin(terms, moved) if rank(terms) else 0.0
            refuse = moved["cost"] > terms["cost"] + slack
        if refuse:
            return R, t, terms, scale, note + (f"; not raised {factor:.2f} times to cover what was seen: "
                                               f"cost {moved['cost']:.3f} against {terms['cost']:.3f}, "
                                               f"contradiction {moved['contradiction']:.2f}")
        for key in ("measured", "scale_note"):
            if key in terms:
                moved[key] = terms[key]
        if per_axis and np.ndim(raised):
            return R2, t2, moved, raised, note + f"; raised to {_describe(raised)} to cover what was seen"
        return R2, t2, moved, raised, note + f"; raised {factor:.2f} times to cover what was seen"

    def known_shape_fit(self, terms, R, t, scale, quality):
        """(terms, R, t, scale, quality): the pose, its ends and its turn
        about its length settled, at the shape measured on earlier views
        (known_shape), fitted about this one's centre, where it costs no more
        than MODEL_KNOWN_SHAPE_MIN_GAIN over this one and fits as well -- then
        measured, by those views; else this one."""
        known = self.known_shape
        if np.allclose(np.broadcast_to(np.asarray(scale, np.float64), (3,)), known,
                       rtol=MODEL_SCALE_FINE_RATIO - 1):
            return terms, R, t, scale, quality
        mean = self.model.points.mean(0)
        centre = R @ (self.linear(scale) @ mean) + t
        Rk, tk, k_terms = self.icp(known, R, centre - R @ (self.linear(known) @ mean), MODEL_FINE_ROUNDS,
                                   MODEL_SAMPLES, MODEL_FINE_POINTS, 2 * self.tol)
        if k_terms is None:
            return terms, R, t, scale, quality
        Rk, tk, k_terms, _ = self.polish(known, Rk, tk, k_terms)
        k_quality = ("fit" if not model_pose_rejection(k_terms)
                     else "coarse" if not model_pose_rejection(k_terms, coarse=True) else None)
        costs = f"{k_terms['cost']:.3f} against {terms['cost']:.3f} at {_describe(scale)}"
        rank = {"fit": 2, "coarse": 1, None: 0}
        if k_terms["cost"] > terms["cost"] + MODEL_KNOWN_SHAPE_MIN_GAIN or rank[k_quality] < rank[quality]:
            terms["scale_note"] = terms.get("scale_note", "") + (f"; the size measured before, "
                                                                 f"{_describe(known)}, fits worse: {costs}")
            terms["known_shape"] = "beaten"
            return terms, R, t, scale, quality
        k_terms["measured"], k_terms["known_shape"] = True, "kept"
        k_terms["scale_note"] = (terms.get("scale_note", "")
                                 + f"; kept the size measured before, {_describe(known)}: {costs}")
        return k_terms, Rk, tk, known, k_quality

    def axes_seen(self, R, t, scale, ends=True):
        """Per mesh box axis, whether the view shows the part's length along
        it: the observed points span MODEL_STRETCH_SEEN of the mesh's extent
        there, and (`ends`) both ends show (ends_seen()). A mesh stretched
        past what was seen spans less of it, and fails."""
        scale = np.broadcast_to(np.asarray(scale, np.float64), (3,))
        along = self.obs @ (R @ self.model.box[:3, :3])
        span = (np.percentile(along, 100 - MODEL_SEEN_PERCENTILE, axis=0)
                - np.percentile(along, MODEL_SEEN_PERCENTILE, axis=0))
        seen = span >= MODEL_STRETCH_SEEN * scale * self.model.extent
        return seen & self.ends_seen(R, t, scale) if ends else seen

    def ends_seen(self, R, t, scale):
        """Per mesh box axis, whether the view shows both of the mesh's ends
        along it; see MODEL_END_HIDDEN."""
        return ~self.hidden_ends(R, t, scale).any(axis=1)

    def hidden_ends(self, R, t, scale):
        """Per mesh box axis, whether its end at -axis and at +axis is hidden
        (3 x 2); see MODEL_END_HIDDEN."""
        M = self.linear(scale)
        P = self.model.points @ (R @ M).T + t
        seen = _seen_from_camera(P, self.K, _overall(scale) * self.model.spacing, self.depth.shape)
        hidden = seen.copy()                      # off the image unless shown otherwise
        idx, u, v, d = self.lines_of_sight(P, seen)
        blocked = (d > 0) & (d < P[idx, 2] - self.hidden_m)
        if self.plane is not None:
            # Beside the mask at the mesh's depth, off the support: an occluder
            # touching it, as terms() counts it.
            n, p0 = self.plane
            blocked |= (~self.grown[v, u] & (d > 0) & (np.abs(d - P[idx, 2]) <= self.tol)
                        & ((self.cam[v, u] - p0) @ n < -self.plane_tol))
        hidden[idx] = blocked
        centre = R @ (M @ self.model.box[:3, 3]) + t
        out = np.zeros((3, 2), bool)
        for k in range(3):
            along = (P - centre) @ (R @ self.model.box[:3, k])
            lo, hi = np.quantile(along, [MODEL_END_SHARE, 1 - MODEL_END_SHARE])
            for e, end in enumerate((along <= lo, along >= hi)):
                front = end & seen
                out[k, e] = front.sum() >= 10 and hidden[front].mean() > MODEL_END_HIDDEN
        return out

    def ends_open(self, R, t, scale):
        """Per mesh box axis, whether just past both of the mesh's ends the
        view shows the part stopping there; see MODEL_END_PROBE_TOLS."""
        M = self.linear(scale)
        P = self.model.points @ (R @ M).T + t
        seen = _seen_from_camera(P, self.K, _overall(scale) * self.model.spacing, self.depth.shape)
        centre = R @ (M @ self.model.box[:3, 3]) + t
        h, w = self.depth.shape
        out = np.ones(3, bool)
        for k in range(3):
            axis = R @ self.model.box[:3, k]
            along = (P - centre) @ axis
            lo, hi = np.quantile(along, [MODEL_END_SHARE, 1 - MODEL_END_SHARE])
            for end, outward in ((along <= lo, -axis), (along >= hi, axis)):
                probe = P[end & seen] + MODEL_END_PROBE_TOLS * self.tol * outward
                if self.plane is not None:
                    n, p0 = self.plane
                    probe = probe[(probe - p0) @ n <= self.plane_tol]      # not into the support
                if len(probe) < 10:
                    continue
                uv = np.round(_pixels(probe, self.K)).astype(np.int64)
                inside = ((probe[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w)
                          & (uv[:, 1] >= 0) & (uv[:, 1] < h))
                if inside.sum() < 10:
                    out[k] = False                  # nothing past this end in view
                    continue
                u, v = uv[inside, 0], uv[inside, 1]
                d = self.depth[v, u]
                # Not the part itself in front (its own mask), but anything
                # else, however near its edge: seen nearly end-on, the ring
                # hiding the bar's middle was a few pixels across.
                nearer = (d > 0) & (d < probe[inside, 2] - self.hidden_m) & ~self.mask[v, u]
                if self.plane is not None:
                    n, p0 = self.plane
                    nearer &= (self.cam[v, u] - p0) @ n < -self.plane_tol
                if nearer.mean() > MODEL_END_HIDDEN:
                    out[k] = False
        return out

    def ends_stop(self, R, t, scale):
        """Per mesh box axis, per end (-axis, +axis), whether just past it
        the view shows the part stopping there (as ends_open(), end by end;
        False where nothing past it is in view), or the support does: past
        an end resting on it is the support."""
        M = self.linear(scale)
        P = self.model.points @ (R @ M).T + t
        seen = _seen_from_camera(P, self.K, _overall(scale) * self.model.spacing, self.depth.shape)
        centre = R @ (M @ self.model.box[:3, 3]) + t
        h, w = self.depth.shape
        out = np.zeros((3, 2), bool)
        for k in range(3):
            axis = R @ self.model.box[:3, k]
            along = (P - centre) @ axis
            lo, hi = np.quantile(along, [MODEL_END_SHARE, 1 - MODEL_END_SHARE])
            for e, (end, outward) in enumerate(((along <= lo, -axis), (along >= hi, axis))):
                probe = P[end & seen] + MODEL_END_PROBE_TOLS * self.tol * outward
                if self.plane is not None:
                    n, p0 = self.plane
                    under = (probe - p0) @ n > self.plane_tol
                    if under.sum() >= 10 and (~under).sum() < 10:
                        out[k, e] = True              # it ends on the support
                        continue
                    probe = probe[~under]
                uv = np.round(_pixels(probe, self.K)).astype(np.int64)
                inside = ((probe[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h))
                if inside.sum() < 10:
                    continue
                u, v = uv[inside, 0], uv[inside, 1]
                d = self.depth[v, u]
                nearer = (d > 0) & (d < probe[inside, 2] - self.hidden_m) & ~self.mask[v, u]
                if self.plane is not None:
                    n, p0 = self.plane
                    nearer &= (self.cam[v, u] - p0) @ n < -self.plane_tol
                out[k, e] = nearer.mean() <= MODEL_END_HIDDEN
        return out

    def axial(self, terms, R, t, scale, quality):
        """(terms, R, t, quality, report): the pose slid along the mesh's
        long box axis onto its one end the view shows where it is off it by
        more than the tolerance, and the report of model_axial; see the
        comment after MODEL_SUPPORT_TOP_SIGMAS."""
        scale3 = np.array(np.broadcast_to(np.asarray(scale, np.float64), (3,)))
        k = int(np.argmax(scale3 * self.model.extent))
        M = self.linear(scale3)
        centre = R @ (M @ self.model.box[:3, 3]) + t
        axis = R @ self.model.box[:3, k]
        hidden = self.hidden_ends(R, t, scale3)[k]
        stops = self.ends_stop(R, t, scale3)[k]
        seen = [bool(not hidden[e] and stops[e]) for e in range(2)]
        along = (self.model.points @ (R @ M).T + t - centre) @ axis
        obs = (self.obs - centre) @ axis
        mesh_ends = [float(along.min()), float(along.max())]
        # The tenth point from either end: as few as tell where an end is
        # (hidden_ends()); a percentile is inside a small end face.
        ordered = np.sort(obs)
        few = min(10, len(ordered)) - 1
        obs_ends = [float(ordered[few]), float(ordered[-1 - few])]
        slide = 0.0
        if seen[0] != seen[1]:
            e = 0 if seen[0] else 1
            if abs(obs_ends[e] - mesh_ends[e]) > self.tol:
                slide = obs_ends[e] - mesh_ends[e]
        if slide:
            t2 = t + slide * axis
            terms2 = self.terms(scale, R, t2, MODEL_SAMPLES, MODEL_FINE_POINTS)
            quality2 = (None if terms2 is None else "fit" if not model_pose_rejection(terms2)
                        else "coarse" if not model_pose_rejection(terms2, coarse=True) else None)
            rank = {"fit": 2, "coarse": 1, None: 0}
            if terms2 is not None and rank[quality2] >= rank[quality]:
                for key in ("measured", "known_shape"):
                    if key in terms:
                        terms2[key] = terms[key]
                terms2["scale_note"] = (terms.get("scale_note", "")
                                        + f"; slid {slide * 1000:+.1f} mm along its length onto the end seen")
                terms, t, quality = terms2, t2, quality2
                mesh_ends = [m + slide for m in mesh_ends]
            else:
                slide = 0.0
        # In the mesh's frame, along its box axis k from the box's centre, at the fit's scale.
        report = {"axis": self.model.box[:3, k].tolist(),
                  "ends": [{"seen": seen[e], "mesh_m": round(mesh_ends[e], 4),
                            "seen_to_m": round(obs_ends[e], 4)} for e in range(2)],
                  "slid_m": round(slide, 4)}
        if not any(seen):
            # Slid either way as far as it still spans what was seen.
            report["slide_range_m"] = [round(min(0.0, obs_ends[1] - mesh_ends[1]), 4),
                                       round(max(0.0, obs_ends[0] - mesh_ends[0]), 4)]
        if not all(seen):
            # A length whose end the view did not show is the mesh's, at its
            # proportions: as far off as those may be (the mesh's stretch_max)
            # either way of the overall scale, and no shorter than what was
            # seen of it. RI-S1-1's bar, its end in the hand, went out 262 mm
            # long of 340 -- its mesh 1.28 times too thick for its length --
            # and nothing in the reply said the length was a guess
            # (2026-11-19). So the reply gives the range, and, where one end
            # was seen, where along the axis the other may be.
            length = mesh_ends[1] - mesh_ends[0]
            span = obs_ends[1] - obs_ends[0]
            bound = float(getattr(self.model, "stretch_max", MODEL_STRETCH_MAX))
            nominal = _overall(scale3) * float(self.model.extent[k])
            lo = max(span, nominal / bound)
            hi = max(length, nominal * bound, lo)
            report["length_m"] = round(length, 4)
            report["length_range_m"] = [round(lo, 4), round(hi, 4)]
            if seen[0] != seen[1]:
                anchor = mesh_ends[0] if seen[0] else mesh_ends[1]
                report["hidden_end_range_m"] = ([round(anchor + lo, 4), round(anchor + hi, 4)] if seen[0]
                                                else [round(anchor - hi, 4), round(anchor - lo, 4)])
        return terms, R, t, quality, report

    def lines_of_sight(self, P, seen):
        """The seen mesh points' indices and pixels, and the observed depth
        there (0 where there is none)."""
        h, w = self.depth.shape
        uv = np.round(_pixels(P, self.K)).astype(np.int64)
        idx = np.flatnonzero(seen & (uv[:, 0] >= 0) & (uv[:, 0] < w)
                             & (uv[:, 1] >= 0) & (uv[:, 1] < h))
        u, v = uv[idx, 0], uv[idx, 1]
        return idx, u, v, self.depth[v, u]

    def colour_contradiction(self, scale, R, t):
        """Of the pixels the camera sees the mesh's samples on the part at
        at this pose, the share whose colour is none of the mesh's
        (LOCATE_COLOUR_DELTA); 0 without colours on both."""
        if self.lab is None:
            return 0.0
        P = self.model.points @ (R @ self.linear(scale)).T + t
        seen = _seen_from_camera(P, self.K, _overall(scale) * self.model.spacing, self.depth.shape)
        idx, u, v, d = self.lines_of_sight(P, seen)
        on = (d > 0) & (np.abs(d - P[idx, 2]) <= self.tol) & self.mask[v, u]
        if on.sum() < LOCATE_COLOUR_MIN_POINTS:
            return 0.0
        off = self.model.chroma_tree.query(self.lab[v[on], u[on], 1:])[0]
        return float(np.mean(off > LOCATE_COLOUR_DELTA))

    def place(self, scale, R, mesh_n):
        """Translation putting the mesh's unhidden side on the observed points.

        The side the camera would see, less what is hidden behind something:
        put a part's hidden end into the average and the mesh lands short of
        where the seen end is.
        """
        X = self.model.points[:mesh_n] @ self.linear(scale).T
        spacing = _overall(scale) * self.model.spacing * np.sqrt(MODEL_SAMPLES / mesh_n)
        target = self.obs.mean(0)
        t = target - R @ X.mean(0)
        for _ in range(3):
            P = X @ R.T + t
            idx, _, _, d = self.lines_of_sight(
                P, _seen_from_camera(P, self.K, spacing, self.depth.shape))
            idx = idx[(d <= 0) | (d >= P[idx, 2] - self.hidden_m)]
            if len(idx):
                t = t + target - P[idx].mean(0)
        return t

    def terms(self, scale, R, t, mesh_n, obs_n):
        """Correspondences of one pose and its cost; see the comment above
        MODEL_SAMPLES for what is counted."""
        M = self.linear(scale)
        P = self.model.points[:mesh_n] @ (R @ M).T + t
        # Normals go by the inverse transpose; only their line is used.
        N = self.model.normals[:mesh_n] @ np.linalg.inv(M)
        N = (N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-15)) @ R.T
        spacing = _overall(scale) * self.model.spacing * np.sqrt(MODEL_SAMPLES / mesh_n)
        seen = _seen_from_camera(P, self.K, spacing, self.depth.shape)
        obs = self.obs[:obs_n]
        out = {"pairs": [], "P": P}

        # Observed points onto the mesh's visible surface.
        if seen.sum() < 3:
            return None
        seen_idx = np.flatnonzero(seen)
        dist_a, near = cKDTree(P[seen_idx]).query(obs)
        near = seen_idx[near]
        out["pairs"].append((P[near], obs, N[near], dist_a))

        # Visible mesh points against what the camera sees there.
        idx, u, v, d = self.lines_of_sight(P, seen)
        in_mask = self.grown[v, u]
        open_ = (d > 0) & (d >= P[idx, 2] - self.hidden_m)
        # Outside the mask, off the support and no further than the mesh
        # point: hidden too.
        unmasked = np.zeros(len(idx), bool)
        if self.plane is not None:
            n, p0 = self.plane
            unmasked = (open_ & ~in_mask & (d <= P[idx, 2] + self.tol)
                        & ((self.cam[v, u] - p0) @ n < -self.plane_tol))
        with_depth = max(1, int(np.sum(d > 0)))
        out["hidden"] = float(np.sum(((d > 0) & ~open_) | unmasked)) / with_depth
        out["unmasked"] = float(np.sum(unmasked)) / with_depth
        open_ &= ~unmasked
        idx, u, v, d, in_mask = idx[open_], u[open_], v[open_], d[open_], in_mask[open_]
        # In the mask: paired with the surface seen along the same ray. Outside
        # it: the camera sees past the mesh point, so pull it to the object.
        target = self.cam[v, u]
        if (~in_mask).any():
            target[~in_mask] = self.obs[self.obs_tree.query(P[idx[~in_mask]])[1]]
        dist_b = np.linalg.norm(P[idx] - target, axis=1)
        bad_b = ~in_mask | (np.abs(d - P[idx, 2]) > self.tol)
        out["pairs"].append((P[idx], target, N[idx], dist_b))

        # Mesh points under the support surface.
        below = np.zeros(len(P))
        if self.plane is not None:
            n, p0 = self.plane
            below = np.maximum((P - p0) @ n, 0.0)
        out["below"] = below

        sat = lambda dist: np.minimum(dist / self.tol, 1.0) ** 2
        cost_a, cost_c = sat(dist_a), sat(below)
        cost_b = np.where(bad_b, 1.0, sat(dist_b)) if len(idx) else np.zeros(1)
        out["cost"] = cost_a.mean() + (cost_b.mean() if len(idx) else 0.0) + cost_c.mean()
        # How much the cost would change on another draw of as many points:
        # the standard error of each mean it is the sum of (turn_margin()).
        out["cost_se"] = float(np.sqrt(cost_a.var() / max(len(cost_a), 1) + cost_b.var() / max(len(cost_b), 1)
                                       + cost_c.var() / max(len(cost_c), 1)))
        out["fitness"] = float(np.mean(dist_a < self.tol))
        out["contradiction"] = float(bad_b.mean()) if len(idx) else 0.0
        out["penetration"] = float(np.mean(below > self.tol))
        out["open_points"] = len(idx)
        out["obs_points"], out["mesh_points"] = len(dist_a), len(P)
        out["p95_distance"] = float(np.percentile(dist_a, 95))
        # The residuals' own spread, as a normal one's sigma: of the observed
        # points within the TOL, 1.4826 times their median distance.
        inside = dist_a[dist_a < self.tol]
        out["resid_sigma"] = float(1.4826 * np.median(inside)) if len(inside) >= MIN_BOX_POINTS else None
        out["width"] = float(np.sort(np.broadcast_to(scale, (3,)) * self.model.extent)[1])
        out["tol"] = self.tol
        return out

    def step(self, R, t, terms, radius, scale=None, free=(), bounds=None):
        """One Gauss-Newton step of the pose, about the mesh's centre; and of
        the log of `scale` along the box axes `free` (indices), about the
        mesh's box centre, within `bounds` (two triples of scales).
        Returns R, t and the scale."""
        P = terms["P"]
        c = P.mean(0)
        free = list(free)
        n_par = 6 + len(free)
        if free:
            scale = np.broadcast_to(np.asarray(scale, np.float64), (3,)).copy()
            axes = (R @ self.model.box[:3, :3])[:, free]
            box_c = R @ (self.linear(scale) @ self.model.box[:3, 3]) + t
        H, g = np.zeros((n_par, n_par)), np.zeros(n_par)
        lam = MODEL_POINT_WEIGHT
        for p, q, n, dist in terms["pairs"]:
            use = dist < radius
            if not use.any():
                continue
            p, q, n = p[use], q[use], n[use]
            d, r = p - c, p - q
            J = np.zeros((len(p), 3, n_par))
            J[:, 0, 1], J[:, 0, 2], J[:, 1, 0] = d[:, 2], -d[:, 1], -d[:, 2]
            J[:, 1, 2], J[:, 2, 0], J[:, 2, 1] = d[:, 0], d[:, 1], -d[:, 0]
            J[:, :, 3:6] = np.eye(3)
            if free:
                # Stretched along a box axis, a point moves along it by its
                # distance from the box centre along it.
                J[:, :, 6:] = axes[None] * ((p - box_c) @ axes)[:, None, :]
            nJ = np.einsum("ki,kij->kj", n, J)
            k = 1.0 / len(p)
            H += k * (lam * np.einsum("kij,kil->jl", J, J) + (1 - lam) * nJ.T @ nJ)
            g += k * (lam * np.einsum("kij,ki->j", J, r)
                      + (1 - lam) * nJ.T @ np.einsum("ki,ki->k", n, r))
        below = terms["below"] > 0
        if below.any():
            n = self.plane[0]
            a = np.hstack([np.cross(P[below] - c, n), np.tile(n, (below.sum(), 1))]
                          + ([((P[below] - box_c) @ axes) * (n @ axes)] if free else []))
            k = 1.0 / len(P)
            H += k * a.T @ a
            g += k * a.T @ terms["below"][below]
        if not np.any(H):
            return R, t, scale
        xi = -np.linalg.solve(H + (1e-6 * np.trace(H) / n_par + 1e-12) * np.eye(n_par), g)
        E = _rotation(xi[:3])
        R, t = E @ R, E @ (t - c) + c + xi[3:6]
        if free:
            stretched = scale.copy()
            stretched[free] *= np.exp(np.clip(xi[6:], -np.log(MODEL_STRETCH_STEP), np.log(MODEL_STRETCH_STEP)))
            if bounds is not None:
                stretched = np.clip(stretched, *bounds)
            stretched = _within_stretch(stretched, self.model.scale_range, self.model.stretch_max)
            if stretched is not None:
                centre = R @ (self.linear(scale) @ self.model.box[:3, 3]) + t
                scale = stretched
                t = centre - R @ (self.linear(scale) @ self.model.box[:3, 3])
        return R, t, scale

    def icp(self, scale, R, t, rounds, mesh_n, obs_n, radius):
        for i in range(rounds):
            terms = self.terms(scale, R, t, mesh_n, obs_n)
            if terms is None:
                return R, t, None
            R, t, _ = self.step(R, t, terms, max(self.tol, radius * 0.75 ** i))
        return R, t, self.terms(scale, R, t, mesh_n, obs_n)

    def stretch(self, scale, R, t, free, rounds=MODEL_STRETCH_ROUNDS, radius=None):
        """ICP over the pose and, with it, the scale along the box axes
        `free` (three booleans), which only grows, by MODEL_STRETCH_MAX at
        most: R, t, terms and the three scales. Its pairs are ICP's: a mesh
        too short for what was seen is drawn out by the points past its ends,
        as far as the camera seeing past it lets it."""
        radius = MODEL_SEARCH_TOLS * self.tol if radius is None else radius
        scale = np.broadcast_to(np.asarray(scale, np.float64), (3,)).copy()
        bounds = scale.copy(), scale * self.model.stretch_max
        free = np.flatnonzero(free)
        for i in range(rounds):
            terms = self.terms(scale, R, t, MODEL_SAMPLES, MODEL_FINE_POINTS)
            if terms is None:
                return R, t, None, scale
            R, t, scale = self.step(R, t, terms, max(MODEL_STRETCH_RADIUS_TOLS * self.tol, radius * 0.85 ** i), scale, free,
                                    bounds)
        return R, t, self.terms(scale, R, t, MODEL_SAMPLES, MODEL_FINE_POINTS), scale

    def polish(self, scale, R, t, terms, free_scale=False, free_axes=False):
        """Descend the cost itself, a step along or about each of the mesh's
        box axes at a time -- and, if `free_scale`, of the mesh's scale, or if
        `free_axes`, of its scale along each box axis. Returns R, t, terms and
        the scale (three numbers once `free_axes`).

        ICP's squared residuals are not the cost: along a direction little
        holds -- a thin part whose near end is hidden slides along its length
        -- the few outliers it drops can carry it millimetres off the pose
        the cost prefers.
        """
        seen = np.ones(3, bool)
        if free_axes:
            scale = np.broadcast_to(np.asarray(scale, np.float64), (3,)).copy()
            seen = self.axes_seen(R, t, scale)
        axes = R @ self.model.box[:3, :3]
        centre = R @ (self.linear(scale) @ self.model.box[:3, 3]) + t
        lo, hi = self.model.scale_range
        for step_m, step_rad, step_s in zip(np.asarray(MODEL_POLISH_STEPS_TOLS) * self.tol,
                                            np.radians(MODEL_POLISH_STEPS_DEG),
                                            MODEL_POLISH_STEPS_SCALE):
            for _ in range(MODEL_POLISH_ROUNDS):
                best = None
                moves = [(axis, sign, turn) for axis, sign, turn
                         in itertools.product(range(3), (1.0, -1.0), (False, True))]
                if free_scale:
                    moves += [(None, 1.0, False), (None, -1.0, False)]
                if free_axes:
                    moves += [(("stretch", k), sign, False) for k in range(3) if seen[k]
                              for sign in (1.0, -1.0)]
                tries = []
                for axis, sign, turn in moves:
                    sm = scale
                    if axis is None or isinstance(axis, tuple):
                        if axis is None:
                            sm = np.clip(scale * (1 + sign * step_s), lo, hi)
                        else:
                            sm = scale.copy()
                            sm[axis[1]] *= 1 + 2 * sign * step_s
                            if not _stretch_allowed(sm, self.model.scale_range, self.model.stretch_max):
                                continue
                        sm = float(sm) if np.ndim(sm) == 0 else sm
                        Rm, tm = R, centre - R @ (self.linear(sm) @ self.model.box[:3, 3])
                    elif turn:
                        E = _rotation(sign * step_rad * axes[:, axis])
                        Rm, tm = E @ R, E @ (t - centre) + centre
                    else:
                        Rm, tm = R, t + sign * step_m * axes[:, axis]
                    tries.append((sm, Rm, tm))
                # Each move's cost, FIT_WORKERS at a time; the best chosen in
                # the moves' order, by the same gains as one after another.
                best_cost, best_pose = None, None
                for (sm, Rm, tm), cost in zip(tries, fit_jobs(self, "cost", tries, 2)):
                    gain = (terms["cost"] if best_pose is None else best_cost) - (np.inf if cost is None else cost)
                    if gain > (0.0 if best_pose is not None else MODEL_POLISH_MIN_GAIN):
                        best_cost, best_pose = cost, (Rm, tm, sm)
                if best_pose is None:
                    break
                best = self.terms(best_pose[2], best_pose[0], best_pose[1], MODEL_SAMPLES, MODEL_FINE_POINTS)
                terms, (R, t, scale) = best, best_pose
                axes = R @ self.model.box[:3, :3]
                centre = R @ (self.linear(scale) @ self.model.box[:3, 3]) + t
        return R, t, terms, scale

    def candidates(self, on_support=True):
        """Starting rotations: the mesh's box axes laid on the observed box's
        axes every way round, and -- the observed box tilts when part of the
        object is hidden -- with `on_support`, each box axis standing on the
        support surface at a range of headings."""
        box = self.model.box[:3, :3]
        R_obs, _, extent_obs = self.obb
        out = [R_obs @ P @ box.T for P in _signed_permutations()]
        if on_support and self.plane is not None:
            up = -self.plane[0]
            major = R_obs[:, int(np.argmax(extent_obs))]
            x0 = major - (major @ up) * up
            if np.linalg.norm(x0) < 1e-6:
                x0 = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 1.0, 0.0])
            x0 /= np.linalg.norm(x0)
            y0 = np.cross(up, x0)
            for k in range(3):
                for sign in (1.0, -1.0):
                    a = sign * box[:, k]
                    b1 = box[:, (k + 1) % 3]
                    B_model = np.column_stack([b1, np.cross(a, b1), a])
                    for yaw in np.radians(np.arange(0, 360, MODEL_YAW_STEP_DEG)):
                        e1 = np.cos(yaw) * x0 + np.sin(yaw) * y0
                        B_cam = np.column_stack([e1, np.cross(up, e1), up])
                        out.append(B_cam @ B_model.T)
        return out

    def start_scales(self):
        """Scales to search poses at: 1; the observed size's when the mesh is
        plainly another size, comparing the boxes' middle sides -- the
        smallest side seen is a one-sided view's depth (half a cylinder), the
        largest is cut short by whatever hides or crops the part; and, when
        the observed box's proportions plainly differ from the mesh's, those
        proportions, three numbers along the mesh's box axes matched side for
        side by length. A one-sided view can show the wrong proportions -- a
        bar seen from one side looks flat -- so this is only a start: the
        pose search keeps whichever fits best. Proportions past what may be
        stretched are brought within it, not dropped: the simulated bar seen
        from one side, 334 by 26 by 16 mm, looked flatter than any stretch
        allows, and the starts left all fitted it too short. A known scale
        is the only start."""
        if self.model.known_scale is not None:
            return [self.model.known_scale]
        observed, mesh = np.sort(self.obb[2]), np.sort(self.model.extent)
        s = float(np.clip(observed[1] / mesh[1], *self.model.scale_range))
        native = self.model.unit             # the mesh at its own size
        out = [native]
        if abs(np.log(s / native)) > np.log(MODEL_SCALE_START_RATIO):
            out.append(s)
        ratio = observed / mesh
        relative = ratio / _overall(ratio)
        if np.max(np.abs(np.log(relative))) > np.log(MODEL_STRETCH_START_RATIO):
            # Its proportions differ, so no one side gives its size: all three do.
            f = np.empty(3)
            f[np.argsort(self.model.extent)] = (np.clip(_overall(ratio), *self.model.scale_range)
                                                * np.clip(relative, 1 / self.model.stretch_max,
                                                          self.model.stretch_max))
            f = _within_stretch(f, self.model.scale_range, self.model.stretch_max)
            if f is not None:
                out.append(f)
        return out

    def run(self):
        started = time.time()
        budget = getattr(self, "budget", None)
        mesh_n, obs_n, rounds = MODEL_COARSE
        starts, jobs = [], []
        for which, s in enumerate(self.start_scales()):
            for R in self.candidates():
                starts.append((which, s))
                jobs.append((s, R, mesh_n, obs_n, rounds))
        coarse = []
        # Independent starts, fitted FIT_WORKERS at a time, in this order.
        for (which, s), (R, t, terms) in zip(starts, fit_jobs(self, "coarse", jobs)):
            if terms is not None:
                coarse.append((terms["cost"], which, R, t, s))
        if not coarse:
            return None
        coarse.sort(key=lambda item: item[0])
        kept = coarse[:MODEL_KEEP]
        for which in {item[1] for item in coarse}:
            kept += [item for item in coarse if item[1] == which][:MODEL_KEEP_PER_START]
        kept = sorted({id(item): item for item in kept}.values(), key=lambda item: item[0])
        jobs = [(s, R, t, MODEL_FINE_ROUNDS, MODEL_SAMPLES, MODEL_FINE_POINTS, 2 * self.tol)
                for _, _, R, t, s in kept]
        refined = [(terms, R, t, s) for (_, _, _, _, s), (R, t, terms) in zip(kept, fit_jobs(self, "icp", jobs, 2))
                   if terms is not None]
        if not refined:
            return None
        results, done = [], []
        # A minimum combined cost can still violate one of the acceptance
        # gates. Try the other refined rotations before declaring no pose.
        for terms, R, t, s in sorted(refined, key=lambda item: item[0]["cost"]):
            if budget is not None and results and time.time() - started > budget:
                # The rest is the retries' (MODEL_INIT_FIT_BUDGET_SEC): what
                # goes out is the best of those tried, not of all, so not
                # certain, and the retries' full search replaces it (fit_rank()).
                self.search_cut = True
                for r in results:
                    r[0]["scale_note"] = (r[0].get("scale_note", "")
                                          + f"; the init's search stopped at {time.time() - started:.1f} s, "
                                            f"its other starts left to the retries")
                break
            # Starts that refined to the same pose need measuring once.
            if any(np.allclose(np.broadcast_to(s, (3,)), np.broadcast_to(s2, (3,)), rtol=0.02)
                   and np.linalg.norm(t - t2) < self.tol
                   and np.degrees(np.arccos(np.clip((np.trace(R.T @ R2) - 1) / 2, -1, 1))) < 3
                   for R2, t2, s2 in done):
                continue
            done.append((R, t, s))
            start = s
            if np.ndim(s):
                # A start's proportions found the pose. They stand only along
                # the axes the view spans, both ends seen; along the others
                # -- a bar's depth, seen from one side; a length one end of
                # which is hidden, seen short -- the part takes the overall
                # size of the rest, about the same centre. (Not the ends'
                # test: a ring's diameter with an end behind its bar kept,
                # it came out 51 mm of 60.)
                seen = self.axes_seen(R, t, s)
                if not seen.all():
                    centre = R @ (self.linear(s) @ self.model.box[:3, 3]) + t
                    s = np.where(seen, s, _overall(s[seen]) if seen.any() else _overall(s))
                    s = _overall(s) if np.allclose(s, s[0]) else s
                    t = centre - R @ (self.linear(s) @ self.model.box[:3, 3])
                    terms = self.terms(s, R, t, MODEL_SAMPLES, MODEL_FINE_POINTS)
                    if terms is None:
                        continue
            R, t, terms, _ = self.polish(s, R, t, terms)
            scale, note = s, f"not measured, {terms['hidden']:.0%} hidden"
            terms["measured"] = False
            known = self.model.known_scale is not None
            hinted = False
            if (not known and terms["hidden"] > MODEL_SCALE_MAX_HIDDEN and self.scale_hint is not None
                    and abs(np.log(_overall(s) / self.scale_hint)) > np.log(MODEL_SCALE_FINE_RATIO)):
                shape = s * (self.scale_hint / _overall(s)) if np.ndim(s) else self.scale_hint
                mean = self.model.points.mean(0)
                centre = R @ (self.linear(s) @ mean) + t
                Rh, th, h_terms = self.icp(shape, R, centre - R @ (self.linear(shape) @ mean), MODEL_FINE_ROUNDS,
                                           MODEL_SAMPLES, MODEL_FINE_POINTS, 2 * self.tol)
                if h_terms is not None:
                    Rh, th, h_terms, _ = self.polish(shape, Rh, th, h_terms)
                    if not model_pose_rejection(h_terms, coarse=True):
                        R, t, terms, scale, s = Rh, th, h_terms, shape, shape
                        terms["measured"] = False
                        note, hinted = f"not measured, {terms['hidden']:.0%} hidden", True
            located = False
            if (MODEL_LOCATED_FALLBACK and not known and not hinted and terms["hidden"] > MODEL_SCALE_MAX_HIDDEN
                    and self.located is not None and model_pose_rejection(terms, coarse=True)):
                # Neither its start nor the others' scale: the proportions it
                # was located at (MODEL_LOCATED_FALLBACK).
                shape = np.asarray(self.located, np.float64)
                shape = float(shape) if np.ndim(shape) == 0 else shape
                mean = self.model.points.mean(0)
                centre = R @ (self.linear(s) @ mean) + t
                Rl, tl, l_terms = self.icp(shape, R, centre - R @ (self.linear(shape) @ mean), MODEL_FINE_ROUNDS,
                                           MODEL_SAMPLES, MODEL_FINE_POINTS, 2 * self.tol)
                if l_terms is not None:
                    Rl, tl, l_terms, _ = self.polish(shape, Rl, tl, l_terms)
                    if not model_pose_rejection(l_terms, coarse=True):
                        R, t, terms, scale, s = Rl, tl, l_terms, shape, shape
                        terms["measured"] = False
                        note, located = f"not measured, {terms['hidden']:.0%} hidden", True
            if known:
                terms["measured"], note = True, "known"
            elif terms["hidden"] <= MODEL_SCALE_MAX_HIDDEN:
                measured, shape, Rs, ts, ts_terms, edge = self.measure_scale(R, t, s)
                if ts_terms is None:
                    note = "could not be measured"
                elif edge:
                    note = f"best {measured:.3f} at the edge of {self.model.scale_range[0]}-{self.model.scale_range[1]}"
                elif np.ndim(s) == 0 and abs(np.log(measured / self.model.unit)) <= np.log(MODEL_SCALE_FINE_RATIO):
                    # The mesh's own size, as far as this search can tell:
                    # keep exactly it (1 in metres), at the re-fit's pose if that is better.
                    native = self.model.unit
                    t1 = ts + Rs @ ((measured - native) * self.model.points.mean(0))
                    t1_terms = self.terms(native, Rs, t1, MODEL_SAMPLES, MODEL_FINE_POINTS)
                    if t1_terms is not None:
                        R1, t1, t1_terms, _ = self.polish(native, Rs, t1, t1_terms)
                        if s != native or t1_terms["cost"] < terms["cost"]:
                            R, t, terms, scale = R1, t1, t1_terms, native
                    note = f"measured {measured:.3f}, kept {native:g}"
                elif ts_terms["cost"] < terms["cost"] - MODEL_SCALE_MIN_GAIN:
                    at_start = terms
                    R, t, terms, scale = self.polish(shape, Rs, ts, ts_terms, free_scale=True)
                    note = (f"measured; at {_describe(s)}: fitness {at_start['fitness']:.2f}, "
                            f"contradiction {at_start['contradiction']:.2f}")
                else:
                    note = f"measured {measured:.3f}, too little gain to apply"
                # Then its proportions, on the same clear enough view: by the
                # descent, and drawn out with the pose (stretch(); its pose
                # alone descended after, so that it only grows).
                # An end hidden can be the mesh's own doing: too short, it
                # ends behind what is in front of the part. RR-S1-2's bar,
                # measured 0.71 of the mesh by its width, ended behind the
                # ring near its far end, the bar going on past it, and was
                # raised whole instead (cover_seen()), 31 mm thick of 27. So
                # a length the view spans is drawn out too, and kept if, so
                # drawn, its ends show.
                seen = self.axes_seen(R, t, scale)
                spanned = self.axes_seen(R, t, scale, ends=False)
                jobs = [(scale, R, t, terms, None, seen)]
                jobs += [(scale, R, t, terms, free, seen)
                         for free in ([seen, spanned] if (spanned & ~seen).any() else [seen]) if free.any()]
                # Independent tries, FIT_WORKERS at a time, in this order.
                tried_all = fit_jobs(self, "proportions", jobs, 2)
                if any(isinstance(tried, str) and tried == "edge" for tried in tried_all):
                    self.stretch_wanted = True
                tries = [tried for tried in tried_all if tried is not None and not isinstance(tried, str)]
                Rp, tp, p_terms, p_scale = min(tries, key=lambda r: r[2]["cost"])
                if p_terms["cost"] < terms["cost"] - MODEL_STRETCH_MIN_GAIN:
                    R, t, terms, scale = Rp, tp, p_terms, p_scale
                    note += f"; proportions measured, {_describe(scale)}"
                # Measured only if, as fitted, the two longest lengths show their ends.
                longest = np.argsort(np.broadcast_to(scale, (3,)) * self.model.extent)[1:]
                ends = self.ends_seen(R, t, scale)[longest].all()
                stops = self.ends_open(R, t, scale)[longest].all()
                terms["measured"] = ts_terms is not None and not edge and bool(ends) and bool(stops)
                if ts_terms is not None and not edge and not ends:
                    note += "; not counted as measured: an end of the part is hidden"
                elif ts_terms is not None and not edge and not stops:
                    note += "; not counted as measured: the part may go on behind something past an end"
            if hinted:
                note += f"; searched on at the other parts' scale, {self.scale_hint:.3f}, too hidden for its own"
            if located:
                note += f"; searched on at the scale it was located at, {_describe(scale)}, too hidden for its own"
            if not terms["measured"]:
                R, t, terms, scale, note = self.cover_seen(R, t, terms, scale, note, per_axis=hinted or located)
            if np.ndim(start) or start != self.model.unit:
                note += f"; pose searched at {_describe(start)}"
            terms["scale_note"] = note
            quality = ("fit" if not model_pose_rejection(terms)
                       else "coarse" if not model_pose_rejection(terms, coarse=True) else None)
            if quality == "fit":
                return terms, R, t, scale, quality
            results.append((terms, R, t, scale, quality))
        # No pose fits well: the best coarse one, else the best, refused.
        coarse = [r for r in results if r[4] == "coarse"]
        best = min(coarse or results, key=lambda r: r[0]["cost"])
        return best

    def proportions_try(self, scale, R, t, terms, free, seen):
        """One try at the part's proportions in run(): with `free` None, the
        pose's descent with its scale free along each box axis; else drawn
        out along the box axes `free` (stretch()) and its pose descended
        after -- None where that reaches the edge of what may be stretched,
        or draws a length out that `seen` does not, but its ends do not then
        show. (R, t, terms, scale) as polish() returns them."""
        if free is None:
            return self.polish(scale, R, t, terms, free_axes=True)
        bound = np.broadcast_to(np.asarray(scale, np.float64), (3,)) * self.model.stretch_max
        Rs2, ts2, s_terms, s2 = self.stretch(scale, R, t, free)
        if s_terms is None:
            return None
        if np.any(s2[free] >= MODEL_STRETCH_EDGE * bound[free]):
            return "edge"                 # wants past the bound (MODEL_STRETCH_MIN_GAIN's comment)
        if (free & ~seen).any() and not self.ends_seen(Rs2, ts2, s2)[free & ~seen].all():
            return None
        return self.polish(s2, Rs2, ts2, s_terms)

    def settle_turns(self, terms, R, t, scale, quality):
        """(terms, R, t, scale, quality, doubts): of the pose and the mesh
        turned onto its own box (ObjectModel.box_turns), each refitted alike,
        the best fit, then the cheapest; and the turns the view cannot tell
        from the one returned (MODEL_TURN_COST_SLACK), relative to it (4x4)."""
        extent = np.broadcast_to(scale, (3,)) * self.model.extent
        B, c = self.model.box[:3, :3], self.model.box[:3, 3]
        M = self.linear(scale)
        centre = R @ (M @ c) + t
        round_k = round_axis(self.model, scale)
        same = [np.eye(3)] + list(self.model.box_same)
        near = np.cos(np.radians(MODEL_DOUBT_APART_DEG))

        def alike(Ra, Rb):
            # One pose as the other but for a turn that leaves the part as it was.
            D = B.T @ Ra.T @ Rb @ B
            return ((round_k is not None and D[round_k, round_k] > near)
                    or any((np.trace(S.T @ D) - 1) / 2 > near for S in same))

        rank = {"fit": 2, "coarse": 1, None: 0}

        def margin(way, other):
            # How far apart the view cannot tell two ways round (turn_margin()).
            return turn_margin(way[0], other[0])

        ways = [(terms, R, t, quality)]
        jobs = []
        for P in self.model.box_turns:
            values, vectors = np.linalg.eig(P)
            axis = np.real(vectors[:, np.argmin(np.abs(values - 1))])
            if round_k is not None and abs(axis[round_k]) > 0.99:
                continue                       # no other way round about it
            to = np.argmax(np.abs(P), axis=0)
            if np.any(np.maximum(extent[to], extent) > self.model.stretch_max * np.minimum(extent[to], extent)):
                continue
            R2 = R @ B @ P @ B.T
            # A half turn is polished whatever its first refit costs (MODEL_DOUBT_HALF_TURNS).
            half = MODEL_DOUBT_HALF_TURNS and np.allclose(np.abs(P), np.eye(3))
            jobs.append((scale, R2, centre - R2 @ (M @ c), terms["cost"] + 2 * margin(ways[0], ways[0]), half))
        # The pose the search found is refitted as the others are, so no way
        # round has had more of the search than another: which one the search
        # settled on hangs on where it started (the observed box's axes), and
        # near-equal ways round went out one or the other as a band 0.2 px
        # wider or narrower moved the starts (2026-11-17). Not where there is
        # no other way round to weigh it against.
        if jobs:
            jobs.insert(0, (scale, R, t, np.inf, True))

        def judged(turned):
            R2, t2, terms2, _ = turned
            return terms2, R2, t2, ("fit" if not model_pose_rejection(terms2)
                                    else "coarse" if not model_pose_rejection(terms2, coarse=True) else None)

        # Each way round refitted and polished on its own, FIT_WORKERS at a
        # time; taken in this order (_turn_start()).
        results = fit_jobs(self, "turn", jobs, 2)
        if results and results[0] is not None:
            again = judged(results[0])
            if rank[again[3]] >= rank[quality]:
                ways[0] = again
            results = results[1:]
        for turned in results:
            if turned is None:
                continue
            terms2, R2, t2, quality2 = judged(turned)
            if quality2 is not None and not any(alike(way[1], R2) for way in ways):
                ways.append((terms2, R2, t2, quality2))
        # The best fit, then the cheapest, goes out; the rest within the margin
        # go in doubt with it.
        kept = min(ways, key=lambda way: (-rank[way[3]], way[0]["cost"]))
        note = ""
        if kept is not ways[0]:
            note = f"; turned onto its box another way: cost {kept[0]['cost']:.3f} against {ways[0][0]['cost']:.3f}"
        doubts = [way for way in sorted(ways, key=lambda way: way[0]["cost"])
                  if way is not kept and rank[way[3]] >= rank[kept[3]]
                  and way[0]["cost"] <= kept[0]["cost"] + margin(kept, way)]
        if doubts:
            costs = ", ".join(f"{way[0]['cost']:.3f}" for way in doubts)
            note += f"; {len(doubts)} other way{'s' if len(doubts) > 1 else ''} round in doubt, costing {costs}"
        # The rest that fit at all: kept for the binding, not in doubt (MODEL_DOUBT_HALF_TURNS).
        rest = [way for way in ways if way is not kept and not any(way is d for d in doubts)]
        terms2, R, t, quality = kept
        if terms2 is not terms:
            terms2["measured"] = terms.get("measured", False)
        terms2["scale_note"] = terms.get("scale_note", "") + note
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = R, t
        out = []
        for way in doubts:
            other = np.eye(4)
            other[:3, :3], other[:3, 3] = way[1], way[2]
            out.append(np.linalg.inv(pose) @ other)
        self.unlikely = []
        for way in rest:
            other = np.eye(4)
            other[:3, :3], other[:3, 3] = way[1], way[2]
            self.unlikely.append(np.linalg.inv(pose) @ other)
        return terms2, R, t, scale, quality, out

    def rest_on_support(self, terms, R, t, scale, quality):
        """(terms, R, t, scale, quality, sources): the pose, as high as the
        view sees its top along the box axis nearest the support's normal
        where it rests on the support and the view does not measure that
        axis, its bottom on the support (MODEL_SUPPORT_TOP_PERCENTILE); and
        per box axis, where its size came from: "measured" by the view,
        "support", or "mesh" (the mesh's proportion at the fit's scale)."""
        scale3 = np.array(np.broadcast_to(np.asarray(scale, np.float64), (3,)))
        seen = self.axes_seen(R, t, scale3)
        sources = ["measured" if s else "mesh" for s in seen]
        if self.plane is None:
            return terms, R, t, scale, quality, sources
        n, p0 = self.plane
        up = -n                                  # n points into the support
        axes = R @ self.model.box[:3, :3]
        k = int(np.argmax(np.abs(axes.T @ up)))
        if seen[k] or abs(axes[:, k] @ up) < np.cos(np.radians(MODEL_DOUBT_APART_DEG)):
            return terms, R, t, scale, quality, sources
        M = self.linear(scale3)
        heights = (self.model.points @ (R @ M).T + t - p0) @ up
        extent = scale3[k] * self.model.extent[k]
        bottom = heights.min()
        if abs(bottom) > (1 - 1 / self.model.stretch_max) * extent + self.tol:
            return terms, R, t, scale, quality, sources          # not resting there
        top = (np.percentile((self.obs - p0) @ up, MODEL_SUPPORT_TOP_PERCENTILE)
               - MODEL_SUPPORT_TOP_SIGMAS * (self.sigma or 0.0))
        if top <= 0 or abs(heights.max() - top) > extent:
            return terms, R, t, scale, quality, sources          # the top seen is not its own
        lo, hi = self.model.scale_range
        scaled = scale3.copy()
        scaled[k] = float(np.clip(top / self.model.extent[k], lo, hi))
        if abs(scaled[k] / scale3[k] - 1) < MODEL_SCALE_FINE_RATIO - 1:
            sources[k] = "support"
            return terms, R, t, scale, quality, sources
        # Scaled about the box centre, then down onto the support.
        c = self.model.box[:3, 3]
        t2 = R @ (M @ c) + t - R @ (self.linear(scaled) @ c)
        low = ((self.model.points @ (R @ self.linear(scaled)).T + t2 - p0) @ up).min()
        t2 = t2 - low * up
        terms2 = self.terms(scaled, R, t2, MODEL_SAMPLES, MODEL_FINE_POINTS)
        if terms2 is None:
            return terms, R, t, scale, quality, sources
        quality2 = ("fit" if not model_pose_rejection(terms2)
                    else "coarse" if not model_pose_rejection(terms2, coarse=True) else None)
        rank = {"fit": 2, "coarse": 1, None: 0}
        if rank[quality2] < rank[quality]:
            return terms, R, t, scale, quality, sources
        for key in ("measured", "known_shape"):
            if key in terms:
                terms2[key] = terms[key]
        terms2["scale_note"] = (terms.get("scale_note", "") + f"; {extent * 1000:.1f} mm along its box axis {k}, "
                                f"unmeasured, set to the {top * 1000:.1f} mm the view sees its top over the support")
        sources[k] = "support"
        return terms2, R, t2, scaled, quality2, sources

    def measure_scale(self, R, t, scale=1.0):
        """The overall scale this rotation's fit, at `scale` and keeping its
        proportions, prefers: (overall scale, scale, R, t, terms, at the
        range's edge). Each scale is re-fit by ICP about the mesh's centre, so
        the part stays where it was seen."""
        mean = self.model.points.mean(0)
        centre = R @ (self.linear(scale) @ mean) + t
        proportions = np.asarray(scale, np.float64) / _overall(scale)
        lo, hi = self.model.scale_range

        def fits(scales, R0, mesh_n, obs_n):
            # Each scale re-fit on its own, FIT_WORKERS at a time, in this order.
            shapes = [s * proportions if np.ndim(proportions) else s for s in scales]
            jobs = [(shape, R0, centre - R0 @ (self.linear(shape) @ mean), MODEL_SCALE_ROUNDS, mesh_n, obs_n,
                     2 * self.tol) for shape in shapes]
            return [(s, Rs, ts, terms) for s, (Rs, ts, terms) in zip(scales, fit_jobs(self, "icp", jobs, 4))
                    if terms is not None]

        # A bracket about the start's size, moved on while its best is at
        # its end: the start is near the size, and the range is wide.
        b_lo = max(lo, _overall(scale) / MODEL_SCALE_BRACKET)
        b_hi = min(hi, _overall(scale) * MODEL_SCALE_BRACKET)
        coarse, tried = [], set()
        while True:
            steps = np.log(b_hi / b_lo) / np.log(MODEL_SCALE_COARSE_RATIO)
            grid = np.exp(np.linspace(np.log(b_lo), np.log(b_hi), int(np.ceil(steps)) + 1))
            coarse += fits([s for s in grid if round(float(np.log(s)), 6) not in tried], R, *MODEL_COARSE[:2])
            tried.update(round(float(np.log(s)), 6) for s in grid)
            if not coarse:
                return 1.0, scale, R, t, None, False
            # On only past the edge of everything searched so far: the edge
            # between two brackets is inside, and turning back at it would
            # never end.
            best = min(coarse, key=lambda r: r[3]["cost"])[0]
            searched = np.exp(sorted(tried))
            if best <= searched[0] * MODEL_SCALE_COARSE_RATIO and searched[0] > lo * MODEL_SCALE_FINE_RATIO:
                b_lo, b_hi = max(lo, searched[0] / MODEL_SCALE_BRACKET ** 2), searched[0]
            elif best >= searched[-1] / MODEL_SCALE_COARSE_RATIO and searched[-1] < hi / MODEL_SCALE_FINE_RATIO:
                b_lo, b_hi = searched[-1], min(hi, searched[-1] * MODEL_SCALE_BRACKET ** 2)
            else:
                break
        s0, R0 = min(coarse, key=lambda r: r[3]["cost"])[:2]
        # Re-fit about the best coarse scale with every point, out to the
        # neighbouring grid scales.
        k = int(np.ceil(np.log(MODEL_SCALE_COARSE_RATIO) / np.log(MODEL_SCALE_FINE_RATIO)))
        fine = fits([float(np.clip(s0 * MODEL_SCALE_FINE_RATIO ** i, lo, hi)) for i in range(-k, k + 1)], R0,
                    MODEL_SAMPLES, MODEL_FINE_POINTS)
        if not fine:
            return 1.0, scale, R, t, None, False
        s, Rs, ts, terms = min(fine, key=lambda r: r[3]["cost"])
        edge = s <= lo * MODEL_SCALE_FINE_RATIO or s >= hi / MODEL_SCALE_FINE_RATIO
        shape = s * proportions if np.ndim(proportions) else s
        return s, shape, Rs, ts, terms, edge


# The pose search's starts (_ModelFit.run(), _quick_fit()) are independent:
# each is placed and fitted by ICP on its own, and only their results are
# compared after. They are fitted FIT_WORKERS at a time, each worker a process
# of its own -- a start is numpy calls on a few thousand points, too small for
# threads to gain under the GIL or for the GPU one at a time -- and their
# results are taken in the order the starts come, so the fit comes out as
# fitted one start after another, to the bit. On PP-S1-1 (this machine: Ryzen
# 5 3600, 6 cores, 12 threads), 45.6 s of the 48.8 s of locating and fitting
# the init's two meshes was those starts' ICP. 0 or 1: fitted here, as
# pose_server imported as a module does unless start_fit_pool() is called; the
# server sets it (--fit-workers).
FIT_WORKERS = 0
# The server's --fit-workers default: the CPU's threads less four for the
# server's own (the track thread, the fit's serial steps between batches), at
# most 8. Locating and fitting PP-S1-1's two meshes, serial 43.0 s; with 5
# workers 15.7 s, 8 12.9 s, 11 12.3 s (2026-11-12, this machine's 12 threads).
FIT_WORKERS_DEFAULT = max(0, min(8, (os.cpu_count() or 1) - 4))
# Fewer coarse starts than this are fitted here: the view (some 10 MB) goes to
# each worker with its share, and costs about as much as a dozen of them.
FIT_PARALLEL_MIN = 8
# The workers all start with the server; a worker that has not imported in this
# long fails the start, and the server fits here instead.
FIT_POOL_START_S = 120.0
# Independent fits made at once, each on a thread of its own (side_by_side()):
# the proposals a locate fits, the parts an init fits. One fit at a time kept
# 3-4 of the 6 cores busy, its in-process steps leaving the workers idle
# (2026-11-12, PP-S1-2 / RI-S2-1).
FIT_SIDE_BY_SIDE = 4
# An init's mesh fits and its first tracked frame (SAM2 taking the clicks,
# TAPIR's first points) need nothing of each other: the fits run on a thread
# of their own while the frame is tracked on the init's (fit_boxes()'s
# meanwhile) -- on the init's own thread, as grad mode and SAM2's bf16
# autocast hold only there (no_grad_thread()). The fits are CPU, in the
# workers; the frame GPU. Made one after the other they took 4.5-5.0 s and
# 1.8-2.2 s of a 9.4 s init (2026-11-16, PP-S1-1 / RI-S1-1, warm server).
INIT_STEP_WITH_FITS = True
# A server's first init also loaded SAM2 and TAPIR (6.2 s) and warmed CUDA in
# its first tracked frame (5.7 s against 1.8 s after): 28 s for HS-S1-1, its
# fits 12 s (2026-11-16). So the server does both once as it starts, on a
# made-up frame of this size (rows, columns) -- SAM2 and TAPIR resize every
# frame to their own input size -- unless started with --no-warm-up.
WARM_UP_SHAPE = (480, 640)
# What the pose search does not read, left out of the worker's copy of a fit:
# the frame in CIELAB (7 MB) and the observed box's points. The depth is the
# camera points' third column, made again there.
_FIT_UNSHIPPED = ("lab", "_obb_points", "_obb", "depth")
# Left out of a coarse start's terms: the point arrays only the next ICP step
# reads. A coarse start's result is compared by its scores alone.
_FIT_POINT_TERMS = ("pairs", "P", "below")
_fit_pool = None
_fit_pool_workers = 0
_fit_pool_lock = threading.Lock()
# A fit's large arrays -- the view's camera points, its masks, its observed
# points and their tree, the mesh's samples and faces -- go to the workers
# once per fit, not with every call: written once to a file in shared memory
# (FIT_SHIP_DIR), named by a token the fit keeps, and kept by each worker that
# reads it (its last FIT_SHIP_KEPT). A call sends the rest, small, each time,
# so a worker rebuilds the fit as it is then; an array replaced is a new
# token. Every call shipped all of it before, 8.6 MB a time (2026-11-20).
FIT_SHIP_MIN_BYTES = 64 * 1024
FIT_SHIP_KEPT = 8
FIT_SHIP_DIR = None
_fit_ship_tokens = collections.deque()
_worker_shipped = collections.OrderedDict()


def _heavy(value):
    return ((isinstance(value, np.ndarray) and value.nbytes >= FIT_SHIP_MIN_BYTES)
            or isinstance(value, cKDTree))


def _ship_parts(fit):
    """(light fit, heavy fields {("fit"|"model", name): value}) of `fit` as it is."""
    heavy = {}
    light = object.__new__(_ModelFit)
    for k, v in fit.__dict__.items():
        if k in _FIT_UNSHIPPED or k.startswith("_ship"):
            continue
        if k == "model":
            model = object.__new__(type(v))
            for mk, mv in v.__dict__.items():
                if _heavy(mv):
                    heavy[("model", mk)] = mv
                else:
                    model.__dict__[mk] = mv
            light.__dict__["model"] = model
        elif _heavy(v):
            heavy[("fit", k)] = v
        else:
            light.__dict__[k] = v
    return light, heavy


def _ship_token(fit, heavy):
    """The token of `fit`'s heavy fields in FIT_SHIP_DIR, written there now
    where they are new or replaced (by identity) since the last call."""
    signature = tuple(sorted((key, id(value)) for key, value in heavy.items()))
    token = fit.__dict__.get("_ship_token")
    if token is not None and fit.__dict__.get("_ship_signature") == signature:
        return token
    token = uuid.uuid4().hex
    with open(os.path.join(FIT_SHIP_DIR, token + ".partial"), "wb") as fh:
        pickle.dump(heavy, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(os.path.join(FIT_SHIP_DIR, token + ".partial"), os.path.join(FIT_SHIP_DIR, token))
    # Kept alive with the token, so no other array takes their ids.
    fit.__dict__.update(_ship_token=token, _ship_signature=signature, _ship_keep=list(heavy.values()))
    _fit_ship_tokens.append(token)
    while len(_fit_ship_tokens) > 4 * FIT_SHIP_KEPT:
        try:
            os.remove(os.path.join(FIT_SHIP_DIR, _fit_ship_tokens.popleft()))
        except OSError:
            pass
    return token


def _fit_worker_run_shipped(ship_dir, token, blob, kind, jobs):
    heavy = _worker_shipped.get(token)
    if heavy is None:
        with open(os.path.join(ship_dir, token), "rb") as fh:
            heavy = pickle.load(fh)
        _worker_shipped[token] = heavy
        while len(_worker_shipped) > FIT_SHIP_KEPT:
            _worker_shipped.popitem(last=False)
    else:
        _worker_shipped.move_to_end(token)
    fit = pickle.loads(blob)
    for (owner, name), value in heavy.items():
        (fit.model if owner == "model" else fit).__dict__[name] = value
    fit.depth = fit.cam[..., 2]
    return [_FIT_JOBS[kind](fit, *job) for job in jobs]


def _coarse_start(fit, s, R, mesh_n, obs_n, rounds):
    """One start of the coarse pose search: placed, fitted, and its terms' scores."""
    t = fit.place(s, R, mesh_n)
    R, t, terms = fit.icp(s, R, t, rounds, mesh_n, obs_n, MODEL_SEARCH_TOLS * fit.tol)
    if terms is not None:
        terms = {k: v for k, v in terms.items() if k not in _FIT_POINT_TERMS}
    return R, t, terms


def _icp_start(fit, s, R, t, rounds, mesh_n, obs_n, radius):
    return fit.icp(s, R, t, rounds, mesh_n, obs_n, radius)


def _turn_start(fit, scale, R, t, most, half):
    """One way round of settle_turns(): refitted, and polished unless it
    costs more than `most` (a half turn always is); None where not."""
    R, t, terms = fit.icp(scale, R, t, MODEL_FINE_ROUNDS, MODEL_SAMPLES, MODEL_FINE_POINTS, 2 * fit.tol)
    if terms is None or (not half and terms["cost"] > most):
        return None
    return fit.polish(scale, R, t, terms)


def _cost_job(fit, scale, R, t):
    """One polish move's cost (polish()), None where no mesh point is in view."""
    terms = fit.terms(scale, R, t, MODEL_SAMPLES, MODEL_FINE_POINTS)
    return None if terms is None else terms["cost"]


_FIT_JOBS = {"coarse": _coarse_start, "icp": _icp_start, "turn": _turn_start,
             "proportions": lambda fit, *job: fit.proportions_try(*job), "cost": _cost_job}


def _fit_worker_init(server_pid):
    """A worker dies with the server, however the server ends: stopped by
    pkill (SIGTERM, which runs no finally, so no stop_fit_pool()) or killed,
    the workers were left waiting for work, 95 MB each (2026-11-12). Linux
    kills it when the thread that started it ends -- the server's main
    thread, which starts every worker (start_fit_pool())."""
    try:
        import ctypes
        import signal
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, int(signal.SIGKILL))
    except Exception:
        pass
    if os.getppid() != server_pid:             # gone before the above took
        os._exit(0)


def _fit_worker_ready():
    """A worker's pid, and whether it imported torch (it should not)."""
    return os.getpid(), "torch" in sys.modules


def _fit_worker_run(blob, kind, jobs):
    fit = pickle.loads(blob)
    fit.depth = fit.cam[..., 2]
    return [_FIT_JOBS[kind](fit, *job) for job in jobs]


def _remove_dead_ship_dirs(base):
    """Remove the fit ship directories in `base` left by servers no longer
    running (killed, so their atexit never ran): those whose pid is gone."""
    for name in os.listdir(base):
        if not name.startswith("pose_server_fits_"):
            continue
        try:
            pid = int(name[len("pose_server_fits_"):].split("_", 1)[0])
        except ValueError:
            continue
        try:
            os.kill(pid, 0)
            continue                      # still running (or not ours to judge)
        except ProcessLookupError:
            pass
        except PermissionError:
            continue
        shutil.rmtree(os.path.join(base, name), ignore_errors=True)


def start_fit_pool(workers):
    """Start `workers` fit worker processes (FIT_WORKERS), all now; the
    number started, 0 where none could be and the fits stay here. Spawned,
    not forked: a fork of a process holding CUDA cannot use it, nor be sure
    of anything else the server's threads hold."""
    global _fit_pool, _fit_pool_workers, FIT_WORKERS, FIT_SHIP_DIR
    workers = int(workers)
    if workers <= 1 or _fit_pool is not None:
        return _fit_pool_workers
    started = time.time()
    if FIT_SHIP_DIR is None:
        base = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
        _remove_dead_ship_dirs(base)
        FIT_SHIP_DIR = tempfile.mkdtemp(prefix=f"pose_server_fits_{os.getpid()}_", dir=base)
        atexit.register(shutil.rmtree, FIT_SHIP_DIR, True)
    # Each worker takes the environment it starts in: no GPU, one BLAS thread
    # (a worker's share of the starts is its whole core), and the flag that
    # leaves torch and the pipeline out of its import (IN_FIT_WORKER).
    env = {FIT_WORKER_ENV: "1", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    saved = {k: os.environ.get(k) for k in env}
    pool = None
    try:
        os.environ.update(env)
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_fit_worker_init, initargs=(os.getpid(),))
        # A submit with no worker idle starts one: these start every worker now,
        # in the environment above.
        ready = [pool.submit(_fit_worker_ready) for _ in range(workers)]
    except Exception as exc:
        print(f"[server] fit workers not started ({type(exc).__name__}: {exc}); fitting here", flush=True)
        ready = []
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    try:
        replies = [f.result(timeout=FIT_POOL_START_S) for f in ready]
    except Exception as exc:
        print(f"[server] fit workers not started ({type(exc).__name__}: {exc}); fitting here", flush=True)
        replies = []
    if not replies:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        return 0
    if any(torch_loaded for _, torch_loaded in replies):
        print("[server] WARNING: a fit worker imported torch; each holds some 450 MB instead of 110", flush=True)
    _fit_pool, _fit_pool_workers, FIT_WORKERS = pool, workers, workers
    print(f"[server] {workers} fit workers started in {time.time() - started:.1f} s", flush=True)
    return workers


def stop_fit_pool():
    global _fit_pool, _fit_pool_workers, FIT_WORKERS
    with _fit_pool_lock:
        pool, _fit_pool, _fit_pool_workers, FIT_WORKERS = _fit_pool, None, 0, 0
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)


def fit_jobs(fit, kind, jobs, min_jobs=FIT_PARALLEL_MIN):
    """Each of `jobs` (argument tuples) done by _FIT_JOBS[kind] on `fit`, in
    order: shared among the fit workers where there are at least `min_jobs`
    (FIT_WORKERS), else here -- the same results either way. A pool that has
    broken (a worker killed) is given up, and the fits are done here."""
    global _fit_pool, _fit_pool_workers, FIT_WORKERS
    jobs = list(jobs)
    run = _FIT_JOBS[kind]
    pool, workers = _fit_pool, _fit_pool_workers
    if pool is None or len(jobs) < max(2, min_jobs):
        return [run(fit, *job) for job in jobs]
    # The fit's large arrays once (FIT_SHIP_MIN_BYTES), the rest with each call.
    light, heavy = _ship_parts(fit)
    token = _ship_token(fit, heavy)
    blob = pickle.dumps(light, protocol=pickle.HIGHEST_PROTOCOL)
    # Consecutive shares, about equal: a start's cost is set by its rounds and points.
    bounds = np.linspace(0, len(jobs), min(workers, len(jobs)) + 1).round().astype(int)
    try:
        futures = [pool.submit(_fit_worker_run_shipped, FIT_SHIP_DIR, token, blob, kind, jobs[a:b])
                   for a, b in zip(bounds[:-1], bounds[1:])]
        return [result for future in futures for result in future.result()]
    except BrokenProcessPool as exc:
        with _fit_pool_lock:
            if _fit_pool is pool:
                _fit_pool, _fit_pool_workers, FIT_WORKERS = None, 0, 0
                print(f"[server] fit workers lost ({exc}); fitting here from now on", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
        return [run(fit, *job) for job in jobs]


def _share_bound(share, count):
    """A share's bound on `count` points: `share` and TOL_SIGMAS of the
    standard error a share that size has on that many (binomial) -- on 3000
    points 0.8's is 0.02, on 100 points 0.10. A part seen on few points is no
    worse a fit for its fitness's coming out a few points under the share by
    chance (2026-11-18: of 768 poses on the right place on the 21 scenes, 21
    more were taken so, and none of 628 off it)."""
    if not count:
        return share                      # no count to judge it on: the share as it is
    return share + TOL_SIGMAS * float(np.sqrt(share * (1 - share) / int(count)))


def p95_limit(terms, coarse=False):
    """The 95th-percentile point distance (m) a pose -- a coarse one, if
    `coarse` -- is taken at: in the view's TOL, and no more than its share of
    the part's width where it is known (model_pose_rejection())."""
    p95, share, floor = ((MODEL_COARSE_MAX_P95_DISTANCE_TOLS, MODEL_COARSE_MAX_P95_SHARE, MODEL_COARSE_P95_FLOOR_TOLS)
                         if coarse else (MODEL_MAX_P95_DISTANCE_TOLS, MODEL_MAX_P95_SHARE, MODEL_P95_FLOOR_TOLS))
    p95, floor = p95 * terms["tol"], floor * terms["tol"]
    if "width" in terms:
        p95 = min(p95, max(floor, share * terms["width"]))
    return p95


def points_stray(terms) -> bool:
    """Whether a fit's points take in what is not the part: the mesh explains
    them as a coarse pose would -- fitness, contradiction, penetration all
    within -- but for a tail of them past its 95th-percentile distance
    (p95_limit()). RR-S1-2's bar, a hand gripping its end, was fitted with
    fitness 0.83-0.90 and its points 95-100 mm off at the 95th percentile (2-4
    mm where nothing touched it, HS-S1-1): the hand's points in its segment,
    they stretched its size across 0.83 for 0.61, its radius 17-21 mm for
    13.5 (2026-11-22). Its click is to be made again (fit_boxes())."""
    return (terms["p95_distance"] > p95_limit(terms, coarse=True)
            and not model_pose_rejection(dict(terms, p95_distance=0.0), coarse=True))


class FitRefusal(str):
    """Why a mesh did not fit the view (fit_model_box()), whether its points
    take in what is not the part (points_stray()), and the pose the mesh came
    to on them, (R, t, scale) in the camera, or None."""

    def __new__(cls, text, strays=False, pose=None):
        out = super().__new__(cls, text)
        out.strays = bool(strays)
        out.pose = pose
        return out


def model_pose_rejection(terms, coarse=False):
    """Why the evidence falls short of a pose -- of a coarse one, if `coarse`
    -- or "" if it does not; not a judgement of the supplied object's identity.
    The shares are what is called a fit, the same for any part (the comment
    after MODEL_MIN_FITNESS); each is judged with its sampling error on as
    many points as the view has (_share_bound())."""
    fitness, contradiction, penetration = (
        (MODEL_COARSE_MIN_FITNESS, MODEL_COARSE_MAX_CONTRADICTION, MODEL_COARSE_MAX_PENETRATION) if coarse else
        (MODEL_MIN_FITNESS, MODEL_MAX_CONTRADICTION, MODEL_MAX_PENETRATION))
    p95 = p95_limit(terms, coarse)
    failed = []
    fitness = 1 - _share_bound(1 - fitness, terms.get("obs_points"))
    contradiction = _share_bound(contradiction, terms.get("open_points"))
    penetration = _share_bound(penetration, terms.get("mesh_points"))
    if terms["fitness"] < fitness:
        failed.append(f"fitness < {fitness:.2f}")
    if terms["contradiction"] > contradiction:
        failed.append(f"contradiction > {contradiction:.2f}")
    if terms["penetration"] > penetration:
        failed.append(f"penetration > {penetration:.2f}")
    if terms["open_points"] < MIN_BOX_POINTS:
        failed.append(f"fewer than {MIN_BOX_POINTS} visible mesh depth samples")
    if terms["p95_distance"] > p95:
        failed.append(f"95th-percentile point distance > {p95 * 1000:.0f} mm")
    return "; ".join(failed)


def fit_model_box(model, view, K, rng=None, through=None, scale_hint=None, located=None, image=None,
                  known_shape=None, budget=None):
    """Box of one object from its mesh: ((T_cam_box, extent, note), None) if
    the mesh fits the view, (None, reason) if not.

    The mesh is placed by ICP from a spread of starting rotations, each scored
    by what the camera sees along every mesh point's line of sight (see the
    comment above MODEL_SAMPLES). Its box then goes where the mesh went, at
    the mesh's size whatever share of the object was in view. `known_shape`, the part's
    scales along the mesh's box axes measured on earlier views, is kept
    where this view's own costs no less (MODEL_KNOWN_SHAPE_MIN_GAIN);
    model.fitted_known_shape says "kept", or "beaten" by this view's.
    """
    model.fitted_pose = model.fitted_scale = model.fitted_linear = None
    model.fitted_quality, model.fitted_measured = None, False
    model.fitted_search_cut = False
    model.fitted_tol = None
    model.fitted_spin_doubt = model.fitted_ends_doubt = model.fitted_id = None
    model.fitted_doubts, model.fitted_unlikely = [], []
    model.fitted_known_shape = None
    fit = _ModelFit(model, view, K, rng or np.random.default_rng(0), through, scale_hint, located, image,
                    known_shape)
    fit.budget = budget                    # MODEL_INIT_FIT_BUDGET_SEC, at the init only
    result = fit.run()
    if result is None:
        return None, "no pose left any mesh point in view"
    terms, R, t, scale, quality = result
    if quality is not None:
        narrowed = fit.narrow_tol(terms, R, t, scale)
        if narrowed is not None:
            for key, value in terms.items():
                narrowed.setdefault(key, value)
            narrowed["scale_note"] = narrowed.get("scale_note", "") + f"; TOL set to {fit.tol * 1000:.1f} mm by the fit's residuals"
            terms = narrowed
            quality = ("fit" if not model_pose_rejection(terms)
                       else "coarse" if not model_pose_rejection(terms, coarse=True) else None)
    axis_source = axial = None
    if quality is None:
        # Refused, perhaps only for a height the view could not measure: into
        # the support (MODEL_SUPPORT_TOP_PERCENTILE).
        terms, R, t, scale, quality, axis_source = fit.rest_on_support(terms, R, t, scale, quality)
    doubts = []
    if quality is not None:
        # Relative to the pose, before the shape measured before moves it
        # (MODEL_KNOWN_SHAPE_MIN_GAIN).
        terms, R, t, scale, quality, doubts = fit.settle_turns(terms, R, t, scale, quality)
        settled_linear = fit.linear(scale)
    if quality is not None and fit.known_shape is not None:
        terms, R, t, scale, quality = fit.known_shape_fit(terms, R, t, scale, quality)
    if quality is not None:
        rescued = axis_source
        terms, R, t, scale, quality, axis_source = fit.rest_on_support(terms, R, t, scale, quality)
        if rescued is not None:
            # Set by the support before, measured by the view only at that size.
            axis_source = ["support" if a == "support" else b for a, b in zip(rescued, axis_source)]
        terms, R, t, quality, axial = fit.axial(terms, R, t, scale, quality)
    stats = (f"fitness {terms['fitness']:.2f}, contradiction {terms['contradiction']:.2f}, "
             f"penetration {terms['penetration']:.2f}, {terms['hidden']:.0%} hidden "
             f"({terms['unmasked']:.0%} outside the mask at the mesh's depth), "
             f"p95 distance {terms['p95_distance'] * 1000:.1f} mm, "
             f"scale {_describe(scale)} ({terms['scale_note']}), {fit.observation_note}")
    if quality is None:
        return None, FitRefusal(f"{stats}; {model_pose_rejection(terms, coarse=True)}", points_stray(terms),
                                (R, t, scale))
    # The mesh's points are scaled before they are turned and moved: the
    # object is fitted_pose applied to fitted_linear @ the mesh, which for a
    # uniform scale is fitted_scale * the mesh.
    M = fit.linear(scale)
    # The ways round were turned about the box's centre at the size of the fit
    # they were weighed on; the size set after (the support, the shape known
    # before) moves that centre in the mesh's frame by (M - M_then) c -- by
    # nothing where the mesh's origin is its centre, by 19 mm on a sample whose
    # mesh came 0.2 m off its origin, and the right way round in doubt was no
    # longer the right one (2026-11-18). Turned about the centre at this size.
    if doubts or getattr(fit, "unlikely", None):
        shift = (M - settled_linear) @ model.box[:3, 3]

        def recentred(D):
            D = np.array(D, np.float64)
            D[:3, 3] += (np.eye(3) - D[:3, :3]) @ shift
            return D

        doubts = [recentred(D) for D in doubts]
        fit.unlikely = [recentred(D) for D in getattr(fit, "unlikely", [])]
    T = np.eye(4)
    T[:3, :3] = R @ model.box[:3, :3]
    T[:3, 3] = R @ (M @ model.box[:3, 3]) + t
    model.fitted_pose = np.eye(4)
    model.fitted_pose[:3, :3], model.fitted_pose[:3, 3] = R, t
    model.fitted_scale = _overall(scale)
    model.fitted_linear = M
    model.fitted_quality, model.fitted_measured = quality, bool(terms["measured"])
    model.fitted_search_cut = bool(getattr(fit, "search_cut", False))
    model.fitted_tol = fit.tol
    model.fitted_stretch_wanted = bool(getattr(fit, "stretch_wanted", False)) and quality == "fit"
    # How far the mesh lies off its part, as this fit shows (see FIT_TOL_SIZE).
    # Of any fine fit, its size measured or not: an end hidden takes away its
    # points, not the truth of the distances of those seen (2026-11-18: on
    # meshes 4.5 % rougher none of the fine fits had both ends in view, and
    # nothing was learnt). Not of a coarse one: that may be the wrong pose.
    model.fitted_mesh_error = None
    if quality == "fit":
        try:
            check = fit.terms(scale, R, t, MODEL_SAMPLES, MODEL_FINE_POINTS)
            near, observed, normals, _ = check["pairs"][0]
            off = np.abs(np.sum((observed - near) * normals, axis=1))
            noise = 1.96 * (fit.sigma or 0.0)
            diagonal = float(np.linalg.norm(np.broadcast_to(np.asarray(scale, np.float64), (3,)) * model.extent))
            p95 = float(np.percentile(off, 95))
            model.fitted_mesh_error = float(np.sqrt(max(p95 ** 2 - noise ** 2, 0.0))) / max(diagonal, 1e-9)
        except Exception:
            model.fitted_mesh_error = None
    model.fitted_known_shape = terms.get("known_shape")
    # Along each box axis, where its size came from (MODEL_SUPPORT_TOP_PERCENTILE).
    model.fitted_axis_source = axis_source
    model.fitted_axial = axial                  # see _ModelFit.axial()
    model.fitted_id = uuid.uuid4().hex
    # Every way round in doubt; the first about the long box axis, and the
    # first about another, also as the spin's and the ends' (turn_kind()).
    model.fitted_doubts = doubts
    model.fitted_unlikely = list(getattr(fit, "unlikely", []))
    model.fitted_spin_doubt = next((d for d in doubts if turn_kind(model, d[:3, :3]) == "spin"), None)
    model.fitted_ends_doubt = next((d for d in doubts if turn_kind(model, d[:3, :3]) == "ends"), None)
    label = ("mesh pose fit" if quality == "fit"
             else f"coarse mesh pose fit ({model_pose_rejection(terms)})")
    return (T, np.broadcast_to(scale, (3,)) * model.extent, f"{label}, {stats}"), None


def retry_mesh_fit(model, mask, metres, K, through=None, scale_hint=None, located=None, image=None,
                   known_shape=None):
    """Worker owns its model state and frame snapshot, never the live pipeline."""
    model = copy.copy(model)
    try:
        view = observe(mask, metres, K)
        if view is None:
            return None, "insufficient current depth", model
        fit, reason = fit_model_box(model, view, K, through=through, scale_hint=scale_hint, located=located,
                                    image=image, known_shape=known_shape)
        return fit, reason, model
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", model


def fitted_axis_scale(model):
    """The fitted mesh's scale along its box axes."""
    linear = getattr(model, "fitted_linear", None)
    linear = linear if linear is not None else model.fitted_scale * np.eye(3)
    B = model.box[:3, :3]
    return np.diag(B.T @ np.asarray(linear, np.float64) @ B)


def fitted_box(model):
    """(T, extent): the fitted mesh's box in the frame of its fitted_pose,
    as fit_model_box() makes it."""
    B, scale = model.box[:3, :3], fitted_axis_scale(model)
    linear = getattr(model, "fitted_linear", None)
    linear = np.asarray(linear if linear is not None else model.fitted_scale * np.eye(3), np.float64)
    T = np.eye(4)
    T[:3, :3] = model.fitted_pose[:3, :3] @ B
    T[:3, 3] = model.fitted_pose[:3, :3] @ (linear @ model.box[:3, 3]) + model.fitted_pose[:3, 3]
    return T, scale * model.extent


def refit_turn(model, pose):
    """turn_kind() of `pose` against the fitted mesh's pose where it is
    turned MODEL_REFIT_TURN_DEG or more from it, else None."""
    D = model.fitted_pose[:3, :3].T @ pose[:3, :3]
    if np.degrees(np.arccos(np.clip((np.trace(D) - 1) / 2, -1, 1))) < MODEL_REFIT_TURN_DEG:
        return None
    return turn_kind(model, D)


def turn_kind(model, D):
    """"spin" for a turn D (relative, in the mesh's frame) about the fitted
    mesh's long box axis, "ends" about another, None about the long axis of a
    mesh round about it (round_axis()), which is no other way round."""
    # D + D^T = 2 cos(a) I + 2 (1 - cos(a)) u u^T: u the top eigenvector.
    axis = np.linalg.eigh(D + D.T)[1][:, -1]
    scale = fitted_axis_scale(model)
    long_k = int(np.argmax(scale * model.extent))
    if abs(axis @ model.box[:3, long_k]) < np.sqrt(0.5):
        return "ends"
    return None if round_axis(model, scale) == long_k else "spin"


def carried_view(model, mask, metres, K, through=None):
    """This frame's view of a fitted part, as a _ModelFit to score or
    correct its carried pose on; None where the view shows too little."""
    view = observe(mask, metres, K)
    if view is None:
        return None
    fit = _ModelFit(model, view, K, np.random.default_rng(0), through)
    # How far the mesh lies off its part is the mesh's, as its fit settled it
    # (narrow_tol()); the sensor's part is this frame's.
    settled = getattr(model, "fitted_tol", None)
    if settled is not None:
        fit.set_tol(max(fit.sensor_tol, settled))
    return fit


def correct_carried_pose(fit, model, pose, scale=None):
    """The carried mesh pose corrected on this frame's view `fit`
    (carried_view()), and its cost there; `pose` itself where the correction
    gains less than MODEL_CORRECT_MIN_GAIN. See MODEL_CORRECT_ROUNDS.
    `scale`: along the mesh's box axes, the fit's where None."""
    scale = fitted_axis_scale(model) if scale is None else scale
    start = fit.terms(scale, pose[:3, :3], pose[:3, 3], *MODEL_CHECK_POINTS)
    if start is None:
        return pose, np.inf
    R, t, terms = fit.icp(scale, pose[:3, :3], pose[:3, 3], MODEL_CORRECT_ROUNDS,
                          *MODEL_CHECK_POINTS, MODEL_CORRECT_RADIUS_TOLS * fit.tol)
    if terms is None or start["cost"] - terms["cost"] < MODEL_CORRECT_MIN_GAIN:
        return pose, float(start["cost"])
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R, t
    return out, float(terms["cost"])


def correct_max_m(model):
    """The move a correction may make and be a refinement (MODEL_CORRECT_MAX_DEG):
    the fitted mesh's far points turned that much about its centre."""
    diagonal = float(np.linalg.norm(fitted_axis_scale(model) * model.extent))
    return np.sin(np.radians(MODEL_CORRECT_MAX_DEG)) * diagonal / 2


# The pose prior of a part the robot holds (meta["priors"][i]["source"]): sent
# by a robot turning a part in the hand before the camera, where the tracker
# lost it as it was picked (the fingers over it, its segment gone) and does
# not look for it again while the fingertips touch it. Tried as any prior is
# (apply_priors()), on the view where the mesh lies so (prior_region()).
HOLD_PRIOR_SOURCE = "robot_hold"


def prior_region(model, pose, metres, K):
    """Where the part may be on this frame, the robot saying its mesh lies at
    `pose` (camera frame; model.fitted_linear its shape): the mesh's
    silhouette so, grown by half the part's own diagonal -- how far off a
    hold of it may say -- of the pixels whose depth lies within that of the
    mesh's nearest and farthest. A part in the hand the tracker has lost
    there (its segment gone), turned before the camera to be seen."""
    h, w = metres.shape
    linear = getattr(model, "fitted_linear", None)
    linear = np.eye(3) * (getattr(model, "fitted_scale", None) or 1.0) if linear is None else linear
    rendered = mesh_depth(model, pose, linear, K, (h, w))
    seen = np.isfinite(rendered)
    if not seen.any():
        return seen
    reach = float(np.linalg.norm(fitted_axis_scale(model) * model.extent)) / 2
    near, far = float(rendered[seen].min()), float(rendered[seen].max())
    radius = max(1, int(np.ceil(float(K[0, 0]) * reach / max(near, 1e-3))))
    grown = cv2.dilate(seen.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))) > 0
    valid = np.isfinite(metres) & (metres > 0)
    return grown & valid & (metres > near - reach) & (metres < far + reach)


def correction_size(model, pose, corrected):
    """How far `corrected` puts the fitted mesh from `pose`: its box
    centre's move (m) and the turn (deg)."""
    B = model.box[:3, :3]
    centre = (B * fitted_axis_scale(model)) @ B.T @ model.box[:3, 3]
    move = np.linalg.norm((corrected[:3, :3] - pose[:3, :3]) @ centre + corrected[:3, 3] - pose[:3, 3])
    turn = np.degrees(np.linalg.norm(cv2.Rodrigues(pose[:3, :3].T @ corrected[:3, :3])[0]))
    return float(move), float(turn)


def carried_fit_rejection(model, pose, mask, metres, K, through=None, fit=None, scale=None):
    """Why the fitted mesh, at `pose` in this frame's camera (its fit carried
    here by the tracked motion), no longer lies on what the camera sees of
    the part; "" where it does, or where the view shows too little to say.
    `fit`: this frame's carried_view(), where already made; `scale`: along
    the mesh's box axes, the fit's where None.

    Only the observed points' side of the coarse acceptance is asked: their
    share on the mesh and their 95th-percentile distance. A few millimetres
    of tracked depth already put a thin part seen face-on past the coarse
    contradiction (a ring 5 mm deep: over 0.25), and a part held up has for
    its "support" whatever is behind it."""
    if fit is None:
        fit = carried_view(model, mask, metres, K, through)
    if fit is None:
        return ""
    terms = fit.terms(fitted_axis_scale(model) if scale is None else scale, pose[:3, :3], pose[:3, 3],
                      *MODEL_CHECK_POINTS)
    if terms is None:
        return "no mesh point in view"
    failed = []
    if terms["fitness"] < MODEL_COARSE_MIN_FITNESS:
        failed.append(f"fitness {terms['fitness']:.2f} < {MODEL_COARSE_MIN_FITNESS:.2f}")
    p95 = min(MODEL_COARSE_MAX_P95_DISTANCE_TOLS * fit.tol,
              max(MODEL_COARSE_P95_FLOOR_TOLS * fit.tol, MODEL_COARSE_MAX_P95_SHARE * terms["width"]))
    if terms["p95_distance"] > p95:
        failed.append(f"95th-percentile point distance {terms['p95_distance'] * 1000:.0f} mm > {p95 * 1000:.0f} mm")
    return "; ".join(failed)


def sample_evidence(fit, P, spacing):
    """Per point of P (this frame's camera, a way round's samples `spacing`
    apart): (on, through) -- the camera sees the part's surface at it, in
    the mask and at its depth; or sees past it, inside the mask (a hole) or
    clear of it, or the bare support there. Neither where it is hidden, on
    the far side, off the image or a little in front -- the cases as
    _ModelFit.terms() counts them -- nor past it within MODEL_MASK_GROW_PX of
    the mask's edge: a sample on a side seen edge-on lies a pixel either way
    of the outline, and past it there is whatever is behind the part."""
    on, through = np.zeros(len(P), bool), np.zeros(len(P), bool)
    seen = _seen_from_camera(P, fit.K, spacing, fit.depth.shape)
    idx, u, v, d = fit.lines_of_sight(P, seen)
    in_mask = fit.grown[v, u]
    open_ = (d > 0) & (d >= P[idx, 2] - fit.hidden_m)
    if fit.plane is not None:
        n, p0 = fit.plane
        open_ &= ~(~in_mask & (d <= P[idx, 2] + fit.tol) & ((fit.cam[v, u] - p0) @ n < -fit.plane_tol))
    on[idx] = open_ & in_mask & (np.abs(d - P[idx, 2]) <= fit.tol)
    through[idx] = open_ & (~in_mask | (fit.mask[v, u] & (d > P[idx, 2] + fit.tol)))
    return on, through


def mesh_depth(model, pose, linear, K, shape):
    """The depth (m) of the fitted mesh's nearest surface at each pixel of an
    image of `shape`, the mesh at `pose` and scaled by `linear` (as
    fitted_linear); inf where it casts no silhouette. Its faces are painted
    far to near, each at its mean depth."""
    h, w = shape
    out = np.full((h, w), np.inf, np.float32)
    P = (model.mesh_v @ np.asarray(linear, np.float64).T) @ pose[:3, :3].T + pose[:3, 3]
    uv = _pixels(P, K)
    f = model.mesh_f
    keep = np.all(P[f, 2] > 1e-3, axis=1)
    near = (uv[f, 0].max(1) >= 0) & (uv[f, 0].min(1) < w) & (uv[f, 1].max(1) >= 0) & (uv[f, 1].min(1) < h)
    f = f[keep & near & np.all(np.abs(uv[f]) < 4 * max(h, w), axis=(1, 2))]
    if not len(f):
        return out
    z = P[f, 2].mean(1)
    order = np.argsort(-z)
    for pts, d in zip(np.round(uv[f[order]] * 16).astype(np.int32), z[order]):
        cv2.fillConvexPoly(out, pts, float(d), lineType=cv2.LINE_8, shift=4)
    return out


def _outline(S):
    """The pixels of a silhouette on its edge."""
    S8 = S.astype(np.uint8)
    return S & ~(cv2.erode(S8, np.ones((3, 3), np.uint8)) > 0)


# How far past the silhouettes ends_evidence() looks: its outlines (a 3x3
# erosion) and the band about them (a 9x9 dilation), each a pixel more.
ENDS_EVIDENCE_REACH_PX = 9 // 2 + 3


def ends_evidence(fit, Zj, Zk):
    """Two ways round's silhouettes against this frame's mask (see
    ENDS_SILHOUETTE_SIGMAS): `Zj`, `Zk` their mesh_depth()s. {"for_j",
    "for_k": the pixels for each, "told": both, "area": the silhouettes'
    union, "noise": the margin the view's own mismatch allows, "winner":
    "j", "k" or None, "why": where None}. Counted on the silhouettes' box
    and ENDS_EVIDENCE_REACH_PX about it: nothing farther out counts."""
    Sj, Sk = np.isfinite(Zj), np.isfinite(Zk)
    rows, cols = np.flatnonzero((Sj | Sk).any(1)), np.flatnonzero((Sj | Sk).any(0))
    d, mask = fit.depth, fit.mask
    if len(rows):
        r = ENDS_EVIDENCE_REACH_PX
        box = (slice(max(0, rows[0] - r), rows[-1] + r + 1), slice(max(0, cols[0] - r), cols[-1] + r + 1))
        Zj, Zk, Sj, Sk, d, mask = Zj[box], Zk[box], Sj[box], Sk[box], d[box], mask[box]
    return _ends_evidence(Sj, Sk, Zj, Zk, d, mask, fit.tol)


def _ends_evidence(Sj, Sk, Zj, Zk, d, mask, tol):
    """ends_evidence() on arrays: the silhouettes, their depths, this frame's
    depth and mask, and the TOL."""
    area = int((Sj | Sk).sum())
    has = np.isfinite(d) & (d > 0)

    def judge(only, Z):
        # Pixels only this way round covers: (for it, against it).
        usable = only & has & ~(d < Z - tol)               # nothing nearer hides it
        on = usable & mask & (d <= Z + tol)
        return int(on.sum()), int((usable & ~on).sum())

    on_j, off_j = judge(Sj & ~Sk, Zj)
    on_k, off_k = judge(Sk & ~Sj, Zk)
    n_j, n_k = on_j + off_k, on_k + off_j
    told = n_j + n_k
    # The view's own mismatch: where the two agree, way round j's silhouette
    # against the mask, near its outline (within a few pixels either side),
    # per pixel of that outline.
    both = Sj == Sk
    line_j = _outline(Sj) & both
    near = cv2.dilate(line_j.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    seen_part = mask & has & (np.abs(d - np.where(Sj, Zj, d)) <= tol)
    hidden = Sj & has & (d < Zj - tol)
    wrong = near & both & has & ~hidden & (Sj != seen_part)
    rate = float(wrong.sum()) / max(1, int(line_j.sum()))
    # Along the outline of where the two differ, as many could fall either way.
    differ = Sj != Sk
    grown = cv2.dilate(differ.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    line_d = int(((_outline(Sj) | _outline(Sk)) & grown).sum())
    noise = rate * line_d + ENDS_SILHOUETTE_SIGMAS * np.sqrt(max(told, 1))
    out = {"for_j": n_j, "for_k": n_k, "told": told, "area": area, "noise": round(noise, 1),
           "winner": None, "why": None}
    if abs(n_j - n_k) <= noise:
        out["why"] = (f"{abs(n_j - n_k)} px apart, within the {noise:.0f} the view's own mismatch "
                      f"({rate:.2f} px a pixel of outline, {line_d} of it) and the count's spread allow")
    else:
        out["winner"] = "j" if n_j > n_k else "k"
    return out


def ends_extent(fit, model, pose, scale):
    """Along the fitted mesh's long box axis, at `pose` and `scale` (along
    the mesh's box axes): its two ends' points (camera frame), how far each
    lies from where the part seen ends there (m), whether each is hidden
    (_ModelFit.hidden_ends()), and its length (m). See ENDS_EXTENT_SHARE."""
    scale = np.array(np.broadcast_to(np.asarray(scale, np.float64), (3,)))
    R, t = pose[:3, :3], pose[:3, 3]
    k = int(np.argmax(scale * model.extent))
    M = fit.linear(scale)
    centre = R @ (M @ model.box[:3, 3]) + t
    axis = R @ model.box[:3, k]
    along = (model.points @ (R @ M).T + t - centre) @ axis
    mesh_ends = np.array([along.min(), along.max()])
    ordered = np.sort((fit.obs - centre) @ axis)
    few = min(10, len(ordered)) - 1
    seen_ends = np.array([ordered[few], ordered[-1 - few]])
    return {"points": centre + mesh_ends[:, None] * axis, "off": np.abs(mesh_ends - seen_ends),
            "hidden": np.asarray(fit.hidden_ends(R, t, scale)[k], bool), "length": float(np.ptp(mesh_ends))}


def ends_extent_vote(a, b, share_min=ENDS_EXTENT_SHARE):
    """Two ways round's ends_extent()s, `a` and `b`, against each other:
    {"off_j", "off_k": each one's distance from where the part ends, summed
    over the ends neither hides (m), "share": their difference over the
    part's length, "ends": how many ends told, "winner": "j", "k" or None,
    "why"}. An end of one is the other's nearest end, end for end."""
    off_a = off_b = 0.0
    used = 0
    for e in range(2):
        f = int(np.argmin(np.linalg.norm(b["points"] - a["points"][e], axis=1)))
        if a["hidden"][e] or b["hidden"][f]:
            continue
        off_a, off_b, used = off_a + a["off"][e], off_b + b["off"][f], used + 1
    length = 0.5 * (a["length"] + b["length"])
    share = abs(off_a - off_b) / length if length > 0 else 0.0
    out = {"off_j": round(off_a, 4), "off_k": round(off_b, 4), "share": round(share, 4), "ends": used,
           "winner": None, "why": None}
    if not used:
        out["why"] = "both ends hidden"
    elif share < share_min:
        out["why"] = f"{share:.1%} of its length apart, under {share_min:.1%}"
    else:
        out["winner"] = "j" if off_a < off_b else "k"
    return out


def _ends_flipped(r):
    """A pair's ends' evidence (Binding.vote_ends()) with j and k swapped."""
    swap = {"j": "k", "k": "j"}
    sil, ext = r["silhouette"], r["extent"]
    return {"silhouette": {**sil, "for_j": sil["for_k"], "for_k": sil["for_j"], "winner": swap.get(sil["winner"])},
            "extent": {**ext, "off_j": ext["off_k"], "off_k": ext["off_j"], "winner": swap.get(ext["winner"])}}


def rest_margin(P, plane, tol):
    """How far inside (+, metres) or outside (-) the polygon of its points
    within `tol` of the support `plane` a part at P (its surface samples,
    camera frame) has its centre, both seen along the support's normal; None
    where no point is that near the support."""
    n, p0 = plane
    near = P[(P - p0) @ n >= -tol]
    if not len(near):
        return None
    e1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    E = np.column_stack([e1, np.cross(n, e1)])
    q, c = (near - p0) @ E, (P.mean(0) - p0) @ E
    try:
        hull = ConvexHull(q)
    except Exception:
        hull = None                            # a point or a line: nothing inside it
    if hull is not None:
        # Facets as n.x + b <= 0 inside: the least distance in to an edge.
        inside = -(hull.equations[:, :2] @ c + hull.equations[:, 2])
        if inside.min() >= 0:
            return float(inside.min())
        corners = q[hull.vertices]
        edges = list(zip(corners, np.roll(corners, -1, axis=0)))
    else:
        order = np.argsort(q @ (q[-1] - q[0]) if len(q) > 1 else np.zeros(len(q)))
        edges = [(q[order[0]], q[order[-1]])]
    out = np.inf
    for a, b in edges:
        ab = b - a
        s = np.clip((c - a) @ ab / max(ab @ ab, 1e-18), 0.0, 1.0)
        out = min(out, float(np.linalg.norm(c - a - s * ab)))
    return -out


def same_way(model, Ra, Rb):
    """Whether rotations Ra and Rb put the fitted mesh the same way round:
    within MODEL_DOUBT_APART_DEG but for a turn that leaves it as it was
    (ObjectModel.box_same, the axis it is round about); as settle_turns()."""
    B = model.box[:3, :3]
    D = B.T @ Ra.T @ Rb @ B
    near = np.cos(np.radians(MODEL_DOUBT_APART_DEG))
    k = round_axis(model, fitted_axis_scale(model))
    return ((k is not None and D[k, k] > near)
            or any((np.trace(S.T @ D) - 1) / 2 > near for S in [np.eye(3)] + list(model.box_same)))


def model_doubts(model):
    """A fitted mesh's ways round in doubt (relative turns): its fit's, then
    a refit's held back (MODEL_REFIT_TURN_DEG)."""
    doubts = list(getattr(model, "fitted_doubts", None) or [])
    for d in (getattr(model, "fitted_spin_doubt", None), getattr(model, "fitted_ends_doubt", None)):
        if d is not None and not any(np.allclose(d, k) for k in doubts):
            doubts.append(d)
    return doubts


class Binding:
    """Which way round a fitted mesh lies on its part: the published pose and
    the other ways round that fit, relative to it (turns[0] the identity),
    each carried by the part's one motion; the votes of the frames looked at
    so far, pair by pair; which are ruled out, and by what. See
    BINDING_CONFIDENCE."""

    def __init__(self, model, before=None):
        self.model = model
        self.fit_id = model.fitted_id
        self.scale = fitted_axis_scale(model)
        self.spacing = _overall(self.scale) * model.spacing
        # Each way round's scale along the mesh's box axes and its samples so
        # scaled, in its own frame (scale_for()).
        self.turns, self.scales, self.Xs, self.ids, self.alive, self.ruled = [], [], [], [], [], []
        self.n_samples = len(model.points)
        # Per ordered pair (j, k), over the frames looked at: votes for j and
        # for k, the frames that gave one, and the samples they agree on seen
        # through and told (BINDING_CONFIDENCE).
        self.tally = {}
        self.unstable = []
        # Ways round that fit the init's view but beyond the doubt's margin
        # (MODEL_DOUBT_HALF_TURNS): hypotheses, not sent in doubt.
        self.unlikely = []
        self.frames, self.tol, self.by, self.number = 0, getattr(model, "fitted_tol", None) or 0.006, None, 0
        # The published way round's pose on the last frame looked at.
        self.last_view = None
        self._trees, self._apart = {}, {}
        # The last frame's silhouettes of the ways round turned end for end,
        # {(j, k): ends_evidence()} (ENDS_SILHOUETTE_SIGMAS).
        self.last_ends = {}
        self.add(np.eye(4))
        for d in model_doubts(model):
            self.add(d)
        for d in getattr(model, "fitted_unlikely", None) or []:
            k = self.add(d)
            if k is not None:
                self.unlikely[k] = True
        if before is not None:
            self.carry(before)

    def add(self, turn):
        """A way round (a turn relative to the published pose), unless it is
        one held already, alive or ruled out; its index or None."""
        if any(same_way(self.model, held[:3, :3], turn[:3, :3]) for held in self.turns):
            return None
        self.turns.append(np.asarray(turn, np.float64))
        self.scales.append(self.scale_for(self.turns[-1]))
        B = self.model.box[:3, :3]
        self.Xs.append(self.model.points @ ((B * self.scales[-1]) @ B.T).T)
        self.number += 1
        self.ids.append(f"{self.fit_id[:6]}-{self.number}")
        self.alive.append(True)
        self.ruled.append(None)
        self.unstable.append(0)
        self.unlikely.append(False)
        return len(self.turns) - 1

    def carry(self, before):
        """The ways round of an earlier fit's binding and their verdicts, onto
        these where one is the same way round, else as more: the same samples
        of the same mesh, in the init's frame both (before.model.fitted_pose,
        self.model.fitted_pose)."""
        if before.n_samples != self.n_samples:
            return
        mine = [self.model.fitted_pose @ T for T in self.turns]
        onto = {}
        for j, T in enumerate(before.turns):
            theirs = before.model.fitted_pose @ T
            k = next((k for k, m in enumerate(mine) if same_way(self.model, m[:3, :3], theirs[:3, :3])), None)
            if k is None:
                # A way round the new fit did not leave in doubt: kept, living
                # or not, so a settled binding stays settled.
                k = self.add(np.linalg.inv(self.model.fitted_pose) @ theirs)
                mine.append(theirs)
            if k is None:
                continue
            onto[j] = k
            self.unlikely[k] = self.unlikely[k] and before.unlikely[j]
            if not before.alive[j]:
                self.alive[k], self.ruled[k] = False, before.ruled[j]
        for (j, k), counts in before.tally.items():
            if j in onto and k in onto:
                self.tally[(onto[j], onto[k])] = dict(counts)
        self.frames, self.by = before.frames, before.by
        if before.last_view is not None:
            # In this fit's frame: the same part, seen the same.
            self.last_view = before.last_view @ np.linalg.inv(before.model.fitted_pose) @ self.model.fitted_pose

    def scale_for(self, turn):
        """A way round's scale along the mesh's box axes, for its `turn`
        relative to the published one: the published one's, each axis's
        taken to the axis the turn takes it to -- so it fills the box the
        published one was fitted to, as a turn onto its own box does
        (settle_turns()); left as it is where the turn takes no box axis onto
        another. Kept as fitted, a quarter turn of a part fitted a little
        longer one way than the other is out of its box by the difference,
        and the view's evidence is against it for that alone."""
        B = self.model.box[:3, :3]
        P = np.abs(np.round(B.T @ turn[:3, :3] @ B))
        if not (np.allclose(P.sum(0), 1) and np.allclose(P.sum(1), 1)):
            return self.scale
        return P.T @ self.scale

    def placed(self, k):
        """Way round k's samples in the published mesh's frame."""
        T = self.turns[k]
        return self.Xs[k] @ T[:3, :3].T + T[:3, 3]

    def apart(self, j, k):
        """Which of way round j's samples lie farther than a TOL from
        way round k's surface: where j has something k has not."""
        if (j, k) not in self._apart:
            if k not in self._trees:
                self._trees[k] = cKDTree(self.placed(k))
            self._apart[(j, k)] = self._trees[k].query(self.placed(j))[0] > self.tol
        return self._apart[(j, k)]

    def living(self):
        return [k for k, alive in enumerate(self.alive) if alive]

    def observe(self, fit, pose, at_rest):
        """One frame's evidence: `fit` its carried_view(), `pose` the
        published way round's corrected pose there; `at_rest` whether the
        part may be judged at rest on the support (no robot on it). Whether
        the frame was looked at."""
        living = self.living()
        if len(living) < 2:
            return False
        if self.last_view is not None:
            move, turn = correction_size(self.model, self.last_view, pose)
            if move <= correct_max_m(self.model) and turn <= MODEL_CORRECT_MAX_DEG:
                return False                   # the view looked at already
        self.last_view = pose
        margins, verdicts, placed, poses = {}, {}, {}, {}
        for k in living:
            posed = pose @ self.turns[k]
            if k:
                # Its own few rounds on this frame's depth, as the published
                # one had (check_carried_meshes()); not kept, so it cannot walk.
                posed, _ = correct_carried_pose(fit, self.model, posed, self.scales[k])
            poses[k] = posed
            P = placed[k] = self.Xs[k] @ posed[:3, :3].T + posed[:3, 3]
            verdicts[k] = sample_evidence(fit, P, self.spacing)
            if at_rest and fit.plane is not None:
                margins[k] = rest_margin(P, fit.plane, fit.tol)
        self.tol = fit.tol
        cast = self.vote(verdicts, placed)
        self.vote_ends(fit, poses, cast)
        for k in living:
            m = margins.get(k)
            if (m is not None and m < -fit.tol
                    and any(margins.get(j) is not None and margins[j] >= 0 for j in living if j != k)):
                self.unstable[k] += 1
        self.frames += 1
        return True

    def feature_groups(self, P_j, P_k, tree_k=None):
        """Way round j's features against k, each posed on this frame (P_j,
        P_k): its samples farther than the tolerance from k's, where it has
        what k has not, joined where within MODEL_FRAGMENT_GAP spacings of each
        other, those of MIN_BOX_POINTS samples or more; and where they agree.
        Where they are posed, each corrected on the frame: where they differ
        as turned relative to each other, the corrected one lies on the part
        -- on the tester turning, those samples of the wrong way round came
        out seen on in 13 votes of 36."""
        tree_k = cKDTree(P_k) if tree_k is None else tree_k
        apart = tree_k.query(P_j, distance_upper_bound=self.tol)[0] > self.tol
        idx = np.flatnonzero(apart)
        out = []
        if len(idx) >= MIN_BOX_POINTS:
            pairs = cKDTree(P_j[idx]).query_pairs(MODEL_FRAGMENT_GAP * self.spacing, output_type="ndarray")
            graph = coo_matrix((np.ones(len(pairs), np.uint8), (pairs[:, 0], pairs[:, 1])),
                               shape=(len(idx), len(idx)))
            count, labels = connected_components(graph, directed=False)
            out = [idx[labels == c] for c in range(count) if np.sum(labels == c) >= MIN_BOX_POINTS]
        return out, ~apart

    def vote(self, verdicts, placed):
        """One frame's votes, {k: (on, through)} of the living ways round,
        each posed on the frame as `placed` ({k: samples}), into the tally of
        each ordered pair (BINDING_CONFIDENCE); the ordered pairs that had a
        vote."""
        voted = set()
        unit = min(1.0, (self.spacing / self.tol) ** 2)
        trees = {k: cKDTree(P) for k, P in placed.items()}      # each way round's, once
        groups = {(j, k): self.feature_groups(placed[j], placed[k], trees[k])
                  for j in placed for k in placed if j != k}
        for j, (on_j, through_j) in verdicts.items():
            for k, (on_k, through_k) in verdicts.items():
                if j == k:
                    continue
                counts = self.tally.setdefault((j, k), dict(votes_j=0, votes_k=0, frames=0, wrong=0, told=0,
                                                            kinds=[0, 0, 0, 0]))
                cast = 0
                # j's features: seen on, for j; seen through, for k. k's: the other way.
                for side, (features, (on, through)) in enumerate(((groups[(j, k)][0], (on_j, through_j)),
                                                                  (groups[(k, j)][0], (through_k, on_k)))):
                    for c in features:
                        n_for, n_against = int(on[c].sum()), int(through[c].sum())
                        if (n_for + n_against) * unit < 1 or n_for == n_against:
                            continue
                        counts["votes_j" if n_for > n_against else "votes_k"] += 1
                        # Which: j's own seen on / through, k's seen through / on.
                        counts["kinds"][2 * side + (n_for < n_against)] += 1
                        cast += 1
                counts["frames"] += bool(cast)
                if cast:
                    voted.add((j, k))
                agree_j, agree_k = groups[(j, k)][1], groups[(k, j)][1]
                counts["wrong"] += int(through_j[agree_j].sum()) + int(through_k[agree_k].sum())
                counts["told"] += int((on_j | through_j)[agree_j].sum()) + int((on_k | through_k)[agree_k].sum())
        return voted

    def vote_ends(self, fit, poses, voted):
        """This frame's votes on each pair of living ways round, each at its
        pose here (`poses`): their silhouettes' (ENDS_SILHOUETTE_SIGMAS) and,
        turned end for end, their ends' (ENDS_EXTENT_SHARE), into the tally as
        the features' votes go; `voted`, the ordered pairs vote() counted this
        frame for already. Every pair, not only those end for end: a plate
        turned over about its length shows its notch on the other side, and
        the silhouettes tell that as they tell its ends (2026-11-17)."""
        self.last_ends = {}
        living = sorted(poses)
        pairs = [(j, k) for j in living for k in living if j < k]
        if not pairs:
            return
        B = self.model.box[:3, :3]
        involved = {x for pair in pairs for x in pair}
        depth = {k: mesh_depth(self.model, poses[k], (B * self.scales[k]) @ B.T, fit.K, fit.depth.shape)
                 for k in involved}
        extent = {k: ends_extent(fit, self.model, poses[k], self.scales[k]) for k in involved}
        for j, k in pairs:
            result = self.last_ends[(j, k)] = {"silhouette": ends_evidence(fit, depth[j], depth[k]),
                                               "extent": ends_extent_vote(extent[j], extent[k], 2 * getattr(
                                                   self.model, "tol_share", FIT_TOL_SIZE) / TOL_PER_P95)}
            cast = False
            for kind, r in result.items():
                if r["winner"] is None:
                    continue
                winner = j if r["winner"] == "j" else k
                for a, b in ((j, k), (k, j)):
                    counts = self.tally.setdefault((a, b), dict(votes_j=0, votes_k=0, frames=0, wrong=0, told=0,
                                                                kinds=[0, 0, 0, 0]))
                    counts["votes_j" if winner == a else "votes_k"] += 1
                    counts.setdefault(kind, [0, 0])[0 if winner == a else 1] += 1
                    if (a, b) not in voted and not cast:
                        counts["frames"] += 1
                cast = True

    def ends_favoured(self):
        """The living way round the last frame's silhouettes and ends favour
        (vote_ends()), where that is not the published one: each pair's
        evidence counted for one and against the other, the one with the most
        for it over all its pairs -- not only over the published one: two ways
        round each voted for over it, the one also voted for over the other
        (PP-S1-1's sample, 54 px to 12, 2026-11-17) -- and that more than the
        published one, and voted for over it, not against; None where none."""
        living = [k for k in range(len(self.alive)) if self.alive[k]]
        net = {k: 0 for k in living}
        beats = set()
        for (j, k), r in self.last_ends.items():
            if j not in net or k not in net:
                continue
            for e in r.values():
                if e["winner"] is None:
                    continue
                win, lose = (j, k) if e["winner"] == "j" else (k, j)
                net[win] += 1
                net[lose] -= 1
                beats.add((win, lose))
        if 0 not in net:
            return None
        best = max((k for k in living if k != 0), key=lambda k: net[k], default=None)
        if best is None or net[best] <= net[0] or (0, best) in beats or (best, 0) not in beats:
            return None
        return best

    def odds(self):
        """(the error rate over every living pair, None) and, per living pair
        (j, k), the evidence for j over k: (the log odds of j against k and
        the mesh not matching there, the votes for j, for k, the frames that
        voted). See BINDING_CONFIDENCE."""
        living = self.living()
        out, all_wrong, all_told = {}, 0, 0
        for j in living:
            for k in living:
                if j == k:
                    continue
                c = self.tally.get((j, k), dict(votes_j=0, votes_k=0, frames=0, wrong=0, told=0))
                all_wrong, all_told = all_wrong + c["wrong"], all_told + c["told"]
                eps = (c["wrong"] + 1) / (c["told"] + 2)
                a, b = c["votes_j"], c["votes_k"]
                for_j = a * np.log(1 - eps) + b * np.log(eps)
                for_k = b * np.log(1 - eps) + a * np.log(eps)
                neither = (a + b) * np.log(0.5)
                out[(j, k)] = (float(for_j - np.logaddexp(for_k, neither)), a, b, c["frames"])
        return (all_wrong + 1) / (all_told + 2), None, out

    def decide(self):
        """Rule out what the evidence and the support rule out; the way round
        to publish (an index), and whether anything changed."""
        living = self.living()
        if len(living) < 2:
            return living[0] if living else 0, False
        changed = False
        # On SURFACE_MIN_SEEN views, as any evidence: on one, a mesh a little
        # off its part's underside, its bottom tilted by its error, touches the
        # support at one end, its centre off that end -- HS-S1-1's tester, its
        # right way round and that end for end, both ruled out on the init's
        # view alone, the two left both wrong (2026-11-18).
        for k in living:
            if (self.unstable[k] >= SURFACE_MIN_SEEN
                    and any(self.unstable[j] < SURFACE_MIN_SEEN for j in living if j != k)):
                self.alive[k], self.ruled[k], changed = False, ("support", "it could not rest on the support so"), True
        living = self.living()
        _, _, pairs = self.odds()
        need = np.log(BINDING_CONFIDENCE / (1 - BINDING_CONFIDENCE))
        out = {}
        for (j, k), (log_odds, votes_j, votes_k, frames) in pairs.items():
            if log_odds >= need and frames >= SURFACE_MIN_SEEN:
                out[k] = ("evidence", f"the views' evidence for {self.ids[j]}: {votes_j} votes to {votes_k} "
                                      f"on {frames} frames, odds {np.exp(log_odds):.3g} to 1")
        if out and len(out) < len(living):
            for k, why in out.items():
                self.alive[k], self.ruled[k] = False, why
            changed = True
        if changed:
            kinds = {self.ruled[k][0] for k in range(len(self.turns)) if not self.alive[k]}
            self.by = " and ".join(sorted(kinds))
        living = self.living()
        for k in living:
            # More votes for it than against it, against the published one: in doubt.
            if self.unlikely[k] and k and (k, 0) in pairs and pairs[(k, 0)][1] > pairs[(k, 0)][2]:
                self.unlikely[k], changed = False, True
        if self.alive[0]:
            return 0, changed
        post = self.posterior()
        return max(living, key=lambda k: post.get(k, 0.0)), changed

    def posterior(self):
        """Each living way round's probability on the evidence so far."""
        living = self.living()
        if len(living) < 2:
            return {k: 1.0 for k in living}
        _, _, pairs = self.odds()
        eps = {pair: (self.tally.get(pair, {}).get("wrong", 0) + 1) / (self.tally.get(pair, {}).get("told", 0) + 2)
               for pair in pairs}
        score = {j: sum((pairs[(j, k)][1] - pairs[(j, k)][2]) * np.log((1 - eps[(j, k)]) / eps[(j, k)])
                        for k in living if k != j) / (len(living) - 1) for j in living}
        top = max(score.values())
        weight = {j: np.exp(s - top) for j, s in score.items()}
        total = sum(weight.values())
        return {j: float(weight[j] / total) for j in living}

    def choose(self, ident, by="person"):
        """Accept a living way round by id. Camera choices keep alternatives
        for later views; person/robot choices rule the alternatives out.
        update_bindings() publishes the accepted camera choice."""
        if ident not in self.ids or not self.alive[self.ids.index(ident)]:
            return False
        keep = self.ids.index(ident)
        if by == "camera":
            # One automatic view is revisable: keep the other hypotheses
            # available to the independent views gathered during motion.
            self.by = by
            return True
        for k in self.living():
            if k != keep:
                self.alive[k], self.ruled[k] = False, (by, f"the {by} chose {ident}")
        self.by = by
        return True

    def rebase(self, k):
        """Make way round k the published one: the turns, re-expressed
        relative to it, k's first; its features' caches dropped."""
        order = [k] + [j for j in range(len(self.turns)) if j != k]
        if self.last_view is not None:
            self.last_view = self.last_view @ self.turns[k]
        base = np.linalg.inv(self.turns[k])
        self.turns = [base @ self.turns[j] for j in order]
        for name in ("scales", "Xs", "ids", "alive", "ruled", "unstable", "unlikely"):
            setattr(self, name, [getattr(self, name)[j] for j in order])
        self.unlikely[0] = False
        self.scale = self.scales[0]
        new = {old: i for i, old in enumerate(order)}
        self.tally = {(new[j], new[k]): c for (j, k), c in self.tally.items()}
        self.last_ends = {(new[j], new[k]) if new[j] < new[k] else (new[k], new[j]):
                          (r if new[j] < new[k] else _ends_flipped(r))
                          for (j, k), r in self.last_ends.items()}
        self._trees, self._apart = {}, {}

    def in_doubt(self):
        """The living ways round sent in doubt (model_doubts): all but the
        published one and those still unlikely (MODEL_DOUBT_HALF_TURNS)."""
        return [k for k in self.living() if k and not self.unlikely[k]]

    def state(self):
        living = self.living()
        if len(living) > 1:
            return "ambiguous"
        return "locked" if len(self.turns) > 1 else "single"

    def features(self, k, pose, K, shape):
        """Up to BINDING_FEATURE_PX pixels of way round k's samples where it
        differs from the other living ones, at `pose` (the published one's,
        this frame's camera)."""
        living = [j for j in self.living() if j != k]
        if not living:
            return []
        apart = np.any([self.apart(k, j) for j in living], axis=0)
        T = pose @ self.turns[k]
        P = self.Xs[k][apart] @ T[:3, :3].T + T[:3, 3]
        uv = _pixels(P, K)
        h, w = shape
        uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)]
        if len(uv) > BINDING_FEATURE_PX:
            uv = uv[np.linspace(0, len(uv) - 1, BINDING_FEATURE_PX).astype(int)]
        return np.round(uv).astype(int).tolist()

    def reply(self, held, pose, K, shape):
        """model_binding: the state, what settled it, and each living way
        round -- its id, its turn relative to the published pose (`held` the
        turn about the mesh's round axis bboxes() publishes it with), its
        probability, and where its features are in the image."""
        post = self.posterior()
        eps, unit, _ = self.odds() if len(self.living()) > 1 else (None, None, None)
        inv_held = np.linalg.inv(held)
        return {"state": self.state(), "by": self.by if self.state() != "single" else "init",
                "confidence": BINDING_CONFIDENCE, "frames": self.frames,
                "error_rate": None if eps is None else round(float(eps), 4),
                "ruled_out": [{"id": self.ids[k], "by": self.ruled[k][0], "why": self.ruled[k][1]}
                              for k in range(len(self.turns)) if not self.alive[k]],
                "hypotheses": [{"id": self.ids[k], "turn": (inv_held @ self.turns[k] @ held).tolist(),
                                "p": round(post.get(k, 1.0), 4), "in_doubt": not self.unlikely[k],
                                "features_px": self.features(k, pose, K, shape)} for k in self.living()],
                "ends_evidence": [{"ids": [self.ids[j], self.ids[k]],
                                   "votes": {kind: self.tally.get((j, k), {}).get(kind, [0, 0])
                                             for kind in ("silhouette", "extent")},
                                   "last": {"silhouette": {key: r["silhouette"][key]
                                                           for key in ("for_j", "for_k", "told", "area", "why")},
                                            "extent": {key: r["extent"][key]
                                                       for key in ("off_j", "off_k", "share", "ends", "why")}}}
                                  for (j, k), r in self.last_ends.items()]}


# Locating an object by its mesh, for an object sent with a mesh and no click.
# SAM2 proposes masks where its grid of prompt points falls, and the depth
# image's regions (depth_proposals()) are proposed beside them. Those mostly
# on the robot, as the client sends it, are not parts. The mesh's box rules
# most of the rest out by size and proportions, which is cheap; each left is
# fitted roughly (LOCATE_SCREEN), the LOCATE_FITS best of those properly
# (_quick_fit()), and the best fit's visible surface, projected into the
# image, gives the click: points inside the part, as a person would pick
# them. The click comes from the fitted mesh rather than from the proposal,
# because a proposal can be a piece of the part (the label on a can) that
# the mesh fits as well.
#
# Proportions only rule out; they do not rank: a reconstruction's are not
# the part's. RI-S1-1's bar, 340 x 27 mm, is seen 12.9 to 1 against the
# mesh's 386 x 39, 9.8 to 1.
#
# SAM2 did propose RI-S1-1's bar, but its mask took in a row of the ground
# behind it and was ruled out by its size (LOCATE_RATIO_SLACK). With the
# arm's links taken as the robot's, the bar was then not located at all;
# before, the bar's mesh, stretched, had taken the arm's upper link. Held in
# the air, the bar stands 40 cm clear of the cart and more of the ground in
# depth.
#
# A depth region is one where neighbouring pixels differ in depth by
# MODEL_DEPTH_GAP_SHARE at most (as connected_depth_foreground() joins them), or
# by this share of their depth: a surface seen edge-on steps further a row
# the further away it is, and the ground beyond the bench, 17 cm a row at
# 10 m, came out a region per row. A part standing clear of what is behind
# it comes out whole, one resting on the bench is part of the bench's region
# and left to SAM2. Regions smaller than LOCATE_DEPTH_MIN_PX are the depth's
# speckle. A SAM2 mask with LOCATE_DEPTH_SAME_IOU with a region is that
# region, but for its edge's spill onto what is behind -- SAM2's bar took in
# the row of ground under it, 3.2 m further -- and the region is taken in
# its place.
LOCATE_DEPTH_GAP_SHARE = 0.03
# (Since 2026-11-18 as few as a fit takes, MIN_BOX_POINTS: a region's size
# against each mesh's is judged in square metres at its depth
# (LOCATE_AREA_SLACK), so a part 2 cm across at 0.8 m, some 60 px, is
# proposed too; 100 px of a 640 px image ruled it out whatever its size. No
# region of the 21 scenes was between the two.)
LOCATE_DEPTH_MIN_PX = MIN_BOX_POINTS
LOCATE_DEPTH_SAME_IOU = 0.8
#
# A proposal's area, in square metres at its depth, against the mesh's
# smallest and largest face at the ends of MODEL_SCALE_RANGE, these factors
# beyond: the bench, a wall, a speck are out.
LOCATE_AREA_SLACK = (0.25, 1.5)
# For a mesh of known scale, the size and area are those at it, this factor
# either way for depth noise and a mask's spill.
LOCATE_KNOWN_SCALE_SLACK = 1.25
# One view of a box shows two of its extents. The proposal's two longest
# must be in the proportions of one such pair of the mesh's, within what the
# fit may stretch it to: each axis MODEL_STRETCH_MAX either way. At 1.6 the
# simulated bar, 337 x 21 mm seen from one side, missed the mesh's 9.8 to 1
# by 1.61. The box is of the proposal's main depth component, as the fit
# takes it (connected_depth_foreground()): left in, the few pixels of SAM2's
# bar mask on the ground 3.2 m behind it made a box 988 x 334 mm.
LOCATE_RATIO_SLACK = MODEL_STRETCH_MAX ** 2
# A proposal further off than that may be the part with another in it: SAM2
# took RR-S1-1's bar with the ring on it as one, 265 x 56 mm, x2.05 off the
# bar's mesh, when a person's hand held the bar's end (2026-10-31); the bar
# fitted nothing else, and the ring was located on the hand. Two parts in one
# proposal make any proportions, so every such proposal is fitted too, last,
# as it is, where nothing in the slack fits: the fit's cost saturates on the
# other part's points (MODEL_SEARCH_TOLS), and its acceptance judges what is
# left. (Until 2026-11-17 only those off by 3 times at most, each first
# thinned where it was 1.5 times wider than most of its length -- the ring on
# the bar; fitted whole, the 21 scenes came out the same, HS-S1-1's ring held
# with the bar now located too.)
# A proposal within this many pixels of the image's edge is cut by it.
LOCATE_EDGE_CUT_PX = 2
# Points of a proposal's surface its box is fit to; min_obb() of every pixel
# of a large proposal costs more than the rest of the screening together.
LOCATE_BOX_POINTS = 2000
# The rough fit: mesh points, observed points and ICP rounds, at the
# observed box's axes only -- about 1 s a proposal against 10 s for the
# proper fit. At most LOCATE_SCREENS proposals, nearest the mesh's
# proportions first, keep the init within the client's allowance.
LOCATE_SCREEN = (500, 300, 4)
LOCATE_SCREENS = 24
LOCATE_FITS = 2
# A mesh sent with its colours (meta["models"][i]["colours"]) is also scored by
# them where it is located: of the pixels its samples are seen on the part at,
# the share whose colour is none of the mesh's -- this far in CIELAB's a*b*
# from every one of its samples' -- counts against the fit as contradicted depth
# does. Lightness is left out: shading and gloss change it, and a texture's
# lightness is another camera's. On YCB-V and LM-O test images the right part's
# pixels came out 4 % none of its colours (median; 30 % at the 90th percentile),
# another part's 34 %. A shape alike in size and form is so told apart by its
# colour, where its colours differ. Under this many pixels, colour says nothing.
LOCATE_COLOUR_DELTA = 15.0
LOCATE_COLOUR_MIN_POINTS = 30
# Scales the best proper fit at one scale is fitted again at; see _quick_fit().
LOCATE_SCALE_STEPS = 7
# A proposal this much on another object -- one located before, a clicked
# one's segment -- is that object; this much on the robot, the robot. Less
# on them, it is taken without their pixels.
LOCATE_TAKEN_SHARE = 0.5
# What is left of a proposal mostly on another object can be one more: SAM2
# took RR-S1-2's ring in with the bar it is on, and the ring was not located
# while the bar's pixels were in it. But what is left is mostly the other
# object's own -- the pixels its fitted mesh missed: RI-S4-1's ring was
# located on 132 px left of the bar's proposal (score 1.00, a mesh shrunk to
# it), PP-S1-3's sample on 335 px left of the tester's, each in place of its
# own proposal. So it is proposed only to an object no round locates
# otherwise, opened by the band the located parts are grown by
# (LOCATE_REGION_GROW_PX), so the slivers along the other's edge go; what is
# left goes through the same screens as any proposal, size, proportions and
# fit. (Until 2026-11-17 a 5 px opening and 20 % of the proposal left too;
# without them the 21 scenes came out the same.)
# Meshes of one reconstruction come out at one scale: the parts' scales agree
# -- 2.4 both of two, 0.85 and 0.75 of two others. Of meshes from anywhere else
# nothing follows, so this holds only where the client says so
# (meta["same_scale_meshes"]). Then a part located at a scale off the others' by
# more than this is located again among fits at theirs: PP-S4-1's tester,
# half behind the box, lost to a patch of the box's handle at 0.46, the
# sample at 2.41; at the sample's, its own proposal fitted at 2.21. One the
# others' scale finds none for keeps where it was. As far off as one mesh's
# proportions may be (MODEL_STRETCH_MAX): the simulated parts' meshes, one
# reconstruction, are at 0.95 (ring) and 0.75 (bar) of theirs, 1.26 apart --
# their proportions off by up to 1.39 (2026-11-17); RR-S1-1's ring, located on
# the hand of the person holding the bar, was at 1.23 to the bar's 0.75 at
# times (2026-10-31), 1.64 apart.
LOCATE_SCALE_AGREE = MODEL_STRETCH_MAX
# A mesh's scale is its reconstruction's, the same from scene to scene: the
# ring measured 0.870-0.915 in RI-S1-1 to RR-S1-1. So the server keeps each
# mesh's measured shapes -- the scales along its box axes of fine fits whose
# size the view showed, not one an earlier view's gave
# (MODEL_KNOWN_SHAPE_MIN_GAIN), the last LOCATE_KNOWN_SCALES by the mesh's id,
# for as long as it runs; one that fitted its view better than theirs takes
# their place, not a median of the two that no view measured (the tester
# first seen turned over, 154 mm, then upright, 181: 167 mm, 8 mm off on the
# view turned over after). Every fit tries their median, axis by axis
# (fit_model_box()'s known_shape), and a part too hidden to measure its own
# size (MODEL_SCALE_MAX_HIDDEN) takes its overall size (fit_model_box()'s
# scale_hint) before the other parts' scale. Not
# to rule out where a part is located: a part half hidden is located at the
# scale of what shows -- PP-S4-2's tester, behind the box, at 1.83 of its
# 2.49 -- as far off as the box's latch was located as the ring (0.66 of
# 0.87) in RI-S4-2.
LOCATE_KNOWN_SCALES = 10
# A part located on a fit scored under LOCATE_WEAK is looked for again once
# the others are (locate_unclicked()), and moved where it fits better by
# LOCATE_BETTER. Parts seen whole fit 0.94-0.99; RR-S1-1's ring on the hand,
# 0.68, its own pixels 0.99.
LOCATE_WEAK = 0.8
# Where locating by mesh cannot be sure, the person is asked rather than a
# guess made: an object fitting no proposal, fitting its own under
# LOCATE_ASK_BELOW (once looked for again, LOCATE_WEAK), or fitting another place
# within LOCATE_BETTER as well (sharing no more than LOCATE_TAKEN_SHARE of the
# smaller region) is sent back unlocated with the init's reply (NeedClick,
# "need_click"), with where it fits best, for its click. Clicked, it is not
# looked for: its click is its segment. LOCATE_ASK_BELOW is set by the user's
# scenes, not derived: the tester, located right at 0.76-0.80 in PP-S1-3,
# PP-S1-4, PP-S3-1 and PP-S3-2, was asked for under LOCATE_WEAK to no gain
# (2026-11-04); a T-LESS part located wrong at 0.77, which a click had put
# right, is no longer asked for.
LOCATE_ASK_BELOW = 0.75
LOCATE_BETTER = 0.1
# Points per click, kept this far inside the visible surface's edge.
LOCATE_PROMPTS = 3
LOCATE_EDGE_PX = SEGMENT_EDGE_PX
# Where a located part's mesh is seen (mesh_click()'s region) falls short of
# what of it a proposal holds: RR-S1-2's ring, located on the bar, left its
# rim in the bar's proposal, points 3 cm off the bar, and the bar was not
# located (p95 41 mm); taken out grown 2 px, or the mesh sampled six times as
# densely, still not. So a located part is taken out of the proposals the
# others are located in grown LOCATE_REGION_GROW_PX: the bar was located then
# (0.65, 321 mm long), from what was left of its proposal -- unless so the
# proposal is mostly on it, or too little is left of it for what is left to
# be looked at (LOCATE_DEPTH_MIN_PX), when it is taken out as it is: grown, the
# bar RR-S1-1's ring is on took the ring with it from the proposal they
# share. And out of the others' init masks (fit_boxes()), but
# never out of a located part's own region: grown into it, the bar's region
# cut the ring the bar runs through to two arcs, fitted at scale 3.5; not
# grown at all, the ring's mask kept the bar beside it and was not fitted.
# The masks tracked on later frames are cut so too (without_others()): cut by
# the regions alone, the ring's carried fit was off the part from the first
# frame on (p95 62 mm), the bar beside it in its mask.
LOCATE_REGION_GROW_PX = SEGMENT_EDGE_PX
# SAM2 can take two parts touching for one: on a live RR-S1-2 the ring's
# clicks gave a segment holding the bar, from the ring to the fist and past
# it (1371 px), and the bar's held the ring; the ring's fit failed on it (p95
# 210 mm), and the tracker gave the bar's pixels to the ring, the bar keeping
# 2 points. So a part's segment holding another part's click is the two of
# them: that click goes in as a negative point for it (separate_parts()), to
# the init's segments and to the tracker alike -- the ring's segment then 955
# px, the ring alone, fitted at 0.99, the bar's the bar without the ring.

# Obstacles: what else stands in the view, found with no click and no mesh.
# An init that asks for them (meta["discover"], how many at most) has them
# looked for among the same proposals the parts are located in (SAM2's and
# the depth's), once the parts are: each proposal, largest first, less the
# parts' pixels -- clicked, located -- and the robot's; less its support too
# (observe()), and cut into its depth-connected pieces (MODEL_DEPTH_GAP_SHARE),
# each of OBSTACLE_MIN_PX or more a candidate. A candidate is an obstacle if
# it stands OBSTACLE_MIN_HEIGHT_M or more off its support (or has none: an
# arm in the air), measures OBSTACLE_EXTENT_M across, lies within
# OBSTACLE_MAX_DEPTH_M -- past it is the room -- and is not a surface: no
# more than OBSTACLE_FLAT_SHARE of it within PLANE_TOLS of one plane where
# it is OBSTACLE_FLAT_M or more across (the bench, a wall; a box seen face
# on is less). One OBSTACLE_OVERLAP on an obstacle kept already is that one.
# Its box is that of its points.
#
# An obstacle is then followed by the depth, not tracked: tracked as a part,
# it took the point tracker's points as one -- RI-S1-1, its two parts beside
# two arms and a hand, went from 1.0 s a frame to 109 s on the sixth, the GPU
# full. On each frame (follow_obstacles()) it is a piece, cut as above, of
# what is near where it was (its last region's bounds, OBSTACLE_FOLLOW_PX
# round): less the robot, the parts' masks and the planes cut from it when it
# was found -- its support, the bench under a box; the camera stands still.
# Of those pieces the one most on its last region -- by intersection over
# union, OBSTACLE_FOLLOW_SHARE or more; moved off it, the share of the piece
# in it grown OBSTACLE_FOLLOW_PX, the piece within OBSTACLE_FOLLOW_SIZE of
# its size -- is where it is now, its box that piece's: a person's arm does
# not move as one body. Not found OBSTACLE_LOST_FRAMES frames running, it is
# lost.
OBSTACLE_MIN_PX = 300
OBSTACLE_MIN_HEIGHT_M = 0.02
OBSTACLE_EXTENT_M = (0.03, 1.5)
OBSTACLE_MAX_DEPTH_M = 2.0
OBSTACLE_FLAT_SHARE = 0.9
OBSTACLE_FLAT_M = 0.5
OBSTACLE_OVERLAP = 0.3
OBSTACLE_FOLLOW_PX = 40
OBSTACLE_FOLLOW_SHARE = 0.3
OBSTACLE_FOLLOW_SIZE = 3.0
OBSTACLE_LOST_FRAMES = 3
# Points a followed obstacle's box is fitted to, on every frame: min_obb() of
# LOCATE_BOX_POINTS took 48 ms an obstacle; a box to keep clear of needs less.
OBSTACLE_BOX_POINTS = 400
# A followed box must not turn where the obstacle does not. min_obb() gives
# its axes in no order and either way round, and of a draw of points another
# box each time: on one image replayed, RI-S4-2's box, 272 x 269 mm, turned
# 89 deg from frame to frame (16 frames of 29), the same box. So the points
# fitted are every n-th of the piece, not a draw -- the same piece, the same
# box -- and its axes are taken in the order and the way round (one of the 24
# _signed_permutations()) nearest the last box's. What turn is left is
# smoothed as the parts' poses are (PoseSmoother): a turn well inside
# OBSTACLE_SMOOTH_FULL_DEG moves the box OBSTACLE_SMOOTH_MIN_ALPHA of the way,
# a larger one all of it. Its centre and size are not smoothed: a box that
# lags an arm is one the arm has left.
OBSTACLE_SMOOTH_MIN_ALPHA = 0.3
OBSTACLE_SMOOTH_FULL_DEG = 20.0
# Two obstacles touching -- an arm and the hand at its end, millimetres apart
# in the depth -- are one piece on some frames and two on others, and the one
# followed first took both: under 2 mm of depth noise the arm's box came out
# 178 mm long, then 295, the hand lost on 26 frames of 30. So a pixel is
# looked at only for the obstacle whose last region is nearest it.


def _quick_fit(model, view, K, screen=False, through=None, image=None):
    """The mesh's pose on one proposal, for telling proposals apart: the
    coarse stage of _ModelFit.run(), its best start refined, and nothing of
    the scale measuring after (fit_boxes() fits the chosen one in full).
    Returns (score, (R, t, scale)) or None: a pose the coarse acceptance
    takes over one it does not, then its share of the observed points on
    the mesh less its share seen through or pressed into what is there.
    With `screen`, the rough fit that ranks proposals (LOCATE_SCREEN): the
    starts at the observed box's axes only, and no refining.

    The observed size is always among the starts. Nothing here measures the
    scale, and a mesh some 25 % off the part's size, within
    MODEL_SCALE_START_RATIO, fails the coarse acceptance at 1: the ring's
    69 mm mesh on the 60 mm simulated ring, standing face-on, overhung its
    rim onto the background behind (contradiction 0.37), and the ring was
    not located on its own mask; at the observed size, 0.14.

    Nor does the observed size always give the part's: half behind a box in
    RI-S4-2 the ring showed half its width, and at 1 it fitted with fitness
    0.99 and contradiction 0.26; RR-S1-2's bar, the ring on it in the same
    mask, came out 0.61 times the mesh's length and 1.09 its width. So the
    best pose at one scale is fitted again at scales about it, up to
    MODEL_SCALE_START_RATIO either way (LOCATE_SCALE_STEPS of them), and
    the best of all is taken. Where none of those passes the coarse
    acceptance, the best is fitted once more with its proportions free, as
    the full fit fits them: the bar's mesh, 386 x 39 mm against the part's
    340 x 27, fits it at no one scale -- at 0.77, contradiction 0.28. A mesh
    of known scale is fitted at it alone.
    `through`, pixels that join the view's pieces (_ModelFit's); `image`,
    the frame (RGB), for a mesh with colours (LOCATE_COLOUR_DELTA)."""
    fit = _ModelFit(model, view, K, np.random.default_rng(0), through=through, image=image)
    mesh_n, obs_n, rounds = LOCATE_SCREEN if screen else MODEL_COARSE
    starts = fit.start_scales()
    observed = float(np.clip(np.sort(fit.obb[2])[1] / np.sort(model.extent)[1], *model.scale_range))
    if model.known_scale is None and not any(np.ndim(s) == 0 and np.isclose(s, observed) for s in starts):
        starts.append(observed)
    best = uniform = None
    scales, jobs = [], []
    for s in starts:
        for R in fit.candidates(on_support=not screen):
            scales.append(s)
            jobs.append((s, R, mesh_n, obs_n, rounds))
    # Independent starts, fitted FIT_WORKERS at a time, in this order.
    for s, (R, t, terms) in zip(scales, fit_jobs(fit, "coarse", jobs)):
        if terms is not None and (best is None or terms["cost"] < best[0]):
            best = (terms["cost"], R, t, s, terms)
        if terms is not None and np.ndim(s) == 0 and (uniform is None or terms["cost"] < uniform[0]):
            uniform = (terms["cost"], R, t, s, terms)
    if best is None:
        return None

    def scored(R, t, s, terms):
        return ((not model_pose_rejection(terms, coarse=True),
                 terms["fitness"] - terms["contradiction"] - terms["penetration"]
                 - fit.colour_contradiction(s, R, t)), (R, t, s))

    _, R, t, s, terms = best
    if screen:
        return scored(R, t, s, terms)
    # The best start refined, and the best at one scale fitted again at scales
    # about it: independent, fitted FIT_WORKERS at a time, in this order.
    jobs = [(s, R, t, MODEL_FINE_ROUNDS, MODEL_SAMPLES, MODEL_FINE_POINTS, 2 * fit.tol)]
    scales = [s]
    if uniform is not None and model.known_scale is None:
        _, Ru, tu, su, _ = uniform
        mean = model.points.mean(0)
        centre = Ru @ (su * mean) + tu
        span = np.log(MODEL_SCALE_START_RATIO)
        for ratio in np.exp(np.linspace(-span, span, LOCATE_SCALE_STEPS)):
            si = float(np.clip(su * ratio, *model.scale_range))
            jobs.append((si, Ru, centre - Ru @ (si * mean), MODEL_FINE_ROUNDS, MODEL_SAMPLES,
                         MODEL_FINE_POINTS, 2 * fit.tol))
            scales.append(si)
    found = [scored(Ri, ti, si, termsi) for si, (Ri, ti, termsi) in zip(scales, fit_jobs(fit, "icp", jobs, 2))
             if termsi is not None]
    if found and not max(found, key=lambda item: item[0])[0][0] and model.known_scale is None:
        _, (R, t, s) = max(found, key=lambda item: item[0])
        terms = fit.terms(s, R, t, MODEL_SAMPLES, MODEL_FINE_POINTS)
        tries = [fit.polish(s, R, t, terms, free_axes=True)] if terms is not None else []
        seen = fit.axes_seen(R, t, s)
        if seen.any():
            Ri, ti, termsi, si = fit.stretch(s, R, t, seen)
            if termsi is not None:
                tries.append(fit.polish(si, Ri, ti, termsi, free_axes=True))
        found += [scored(Ri, ti, si, termsi) for Ri, ti, termsi, si in tries if termsi is not None]
    return max(found, key=lambda item: item[0]) if found else None


def mesh_click(model, R, t, scale, metres, K):
    """The click a mesh at a pose gives: (points, region), the points inside
    the region of the image where the camera sees the mesh's surface, or
    None if it sees none of it."""
    B = model.box[:3, :3]
    M = (B * np.broadcast_to(np.asarray(scale, np.float64), (3,))) @ B.T  # _ModelFit.linear()
    P = model.points @ (R @ M).T + t
    spacing = _overall(scale) * model.spacing
    seen = _seen_from_camera(P, K, spacing, metres.shape)
    uv = np.round(_pixels(P[seen], K)).astype(np.int64)
    h, w = metres.shape
    u, v = np.clip(uv[:, 0], 0, w - 1), np.clip(uv[:, 1], 0, h - 1)
    d = metres[v, u]
    # Not where something nearer hides it: that is the occluder's to show.
    shown = np.isfinite(d) & (d > 0) & (np.abs(d - P[seen, 2]) <= MODEL_HIDDEN_TOLS * fit_tol(
        P[seen], K, None, None, getattr(model, "tol_share", FIT_TOL_SIZE)))
    if shown.sum() < MIN_BOX_POINTS:
        return None
    region = np.zeros((h, w), np.uint8)
    region[v[shown], u[shown]] = 1
    # Samples land a few pixels apart; close the gaps between them.
    gap = spacing * K[0, 0] / float(np.median(P[seen][shown, 2]))
    size = 2 * int(np.ceil(gap)) + 1
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
    return region_clicks(region > 0), region > 0


def _grown(region):
    """A located part's region as other parts' proposals lose it (LOCATE_REGION_GROW_PX)."""
    kernel = np.ones((2 * image_px(LOCATE_REGION_GROW_PX, region.shape) + 1,) * 2, np.uint8)
    return cv2.dilate(region.astype(np.uint8), kernel) > 0


def region_clicks(region):
    """Up to LOCATE_PROMPTS points [[x, y], ...] inside `region` (bool), as a
    person would click it: the deepest, then each farthest from those picked,
    at least LOCATE_EDGE_PX in from the edge."""
    inside = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    points = [np.unravel_index(np.argmax(inside), inside.shape)]
    edge = image_px(LOCATE_EDGE_PX, region.shape)
    deep = np.argwhere(inside >= edge)
    # The rest spread over the part, each farthest from those picked.
    while len(points) < LOCATE_PROMPTS and len(deep):
        far = np.min(np.linalg.norm(deep[:, None] - np.asarray(points)[None], axis=2), axis=1)
        if far.max() < 2 * edge:
            break
        points.append(tuple(deep[np.argmax(far)]))
    return [[int(c), int(r)] for r, c in points]


def depth_proposals(metres, K=None):
    """The depth image's regions as masks, of LOCATE_DEPTH_MIN_PX or more."""
    valid = np.isfinite(metres) & (metres > 0)
    count = int(valid.sum())
    if not count:
        return []
    z = np.where(valid, metres, 0.0)
    indices = np.full(metres.shape, -1, dtype=np.int32)
    indices[valid] = np.arange(count)
    sources, targets = [], []
    for a, b in ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])):
        gap = depth_gap_share(K) * np.maximum(z[a], z[b])
        linked = valid[a] & valid[b] & (np.abs(z[a] - z[b]) <= gap)
        sources.append(indices[a][linked])
        targets.append(indices[b][linked])
    sources, targets = np.concatenate(sources), np.concatenate(targets)
    graph = coo_matrix((np.ones(len(sources), dtype=np.uint8), (sources, targets)), shape=(count, count))
    _, labels = connected_components(graph, directed=False)
    regions = np.full(metres.shape, -1, dtype=np.int64)
    regions[valid] = labels
    return [regions == label
            for label in np.flatnonzero(np.bincount(labels) >= LOCATE_DEPTH_MIN_PX)]


def depth_pieces(mask, depth, min_px, K=None):
    """`mask`'s pieces of `min_px` or more, as connected_depth_foreground()
    links pixels (depth_gap_share()), largest first."""
    count = int(mask.sum())
    if count < min_px:
        return []
    indices = np.full(mask.shape, -1, dtype=np.int32)
    indices[mask] = np.arange(count)
    sources, targets = [], []
    for a, b in ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])):
        linked = mask[a] & mask[b] & (np.abs(depth[a] - depth[b])
                                      <= depth_gap_share(K) * np.maximum(depth[a], depth[b]))
        sources.append(indices[a][linked])
        targets.append(indices[b][linked])
    sources, targets = np.concatenate(sources), np.concatenate(targets)
    graph = coo_matrix((np.ones(len(sources), dtype=np.uint8), (sources, targets)), shape=(count, count))
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    out = []
    for label in np.argsort(-sizes):
        if sizes[label] < min_px:
            break
        piece = np.zeros_like(mask)
        piece[mask] = labels == label
        out.append(piece)
    return out


def others_region(index, regions):
    """The pixels of the regions of the objects other than `index`
    (`regions`, one per object, None where it has none), or None."""
    others = [region for j, region in enumerate(regions or ()) if j != index and region is not None]
    return np.any(others, axis=0) if others else None


def without_others(mask, index, regions):
    """Object `index`'s mask less the regions of the others (`regions`, one
    per object, None where it has none): a pixel one object's mesh is seen
    at is not another's -- the ring hung on a bar, in the bar's segment. With
    a region of its own, the others' are grown, as at init (fit_boxes(),
    LOCATE_REGION_GROW_PX), but not into its own."""
    others = others_region(index, regions)
    if others is None:
        return mask
    own = regions[index] if index < len(regions) else None
    if own is not None:
        others = _grown(others) & ~own
    return mask & ~others


def describe_proposal(mask):
    """Where a proposal is, for the server's log."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return "empty"
    return f"{len(ys)} px, u {xs.min()}-{xs.max()} v {ys.min()}-{ys.max()}"


def at_edge(mask, margin=None):
    """Whether `mask` reaches the image's edge: what it shows is cut there."""
    margin = image_px(LOCATE_EDGE_CUT_PX, mask.shape) if margin is None else margin
    return bool(mask[:margin].any() or mask[-margin:].any() or mask[:, :margin].any() or mask[:, -margin:].any())


class NeedClick(Exception):
    """An init locating by mesh could not finish for sure (LOCATE_ASK_BELOW):
    `requests`, per object to click, {"object": index, "why": str,
    "candidates": [click, ...]}, its likeliest places best first; `points`,
    every object's click as far as it is sure, [] for those."""

    def __init__(self, requests, points):
        self.requests, self.points = requests, points
        super().__init__("; ".join(f"object {r['object']}: {r['why']}" for r in requests) + "; click them")


def locate_with_mesh(model, proposals, metres, K, clicked=(), taken=None, robot=None, why=None,
                     rest=False, scales=None, grown=None, image=None, kept=None):
    """Where in the view an object is, by its mesh: (click, region, note,
    (score, overall scale, scale)), the click as mesh_click() gives it and the
    model fitted, or (None, None, reason, None). With `scales`, a range of
    overall scales, a fit outside it is refused, and a rough one ranks last. `clicked` are pixels of other objects, `taken` the
    pixels of other objects -- those already located, those clicked --
    `robot` the robot's pixels: a proposal mostly on them is skipped, one
    partly on them is taken without them -- or, with `rest`, what is left of
    it, the slivers along the other's edge gone. `why`, a list, is given (index, what
    became of it) for each proposal. `grown`, `taken` with the located parts
    grown (LOCATE_REGION_GROW_PX), is what a proposal loses of them where it
    is not mostly on them so and, with `rest`, enough is left; else `taken`
    itself. `kept`, the fits already made, for _kept_quick_fits()."""
    why = [] if why is None else why
    f = float(K[0, 0])
    m = np.sort(model.extent)[::-1]
    low, high = model.scale_range
    if model.known_scale is not None:
        low, high = model.known_scale / LOCATE_KNOWN_SCALE_SLACK, model.known_scale * LOCATE_KNOWN_SCALE_SLACK
    area_range = (m[1] * m[2] * low ** 2 * LOCATE_AREA_SLACK[0],
                  m[0] * m[1] * high ** 2 * LOCATE_AREA_SLACK[1])
    pairs = np.log([m[0] / m[1], m[0] / m[2], m[1] / m[2]])
    h, w = metres.shape
    clicked = [(min(max(int(round(v)), 0), h - 1), min(max(int(round(u)), 0), w - 1))
               for u, v in clicked]
    valid = np.isfinite(metres) & (metres > 0)
    shortlist, on_robot = [], 0
    # Off the mesh's proportions: tried last, should nothing else fit
    # (_locate_fits()); see the comment after LOCATE_RATIO_SLACK.
    mixed = []
    # What joins a proposal's pieces, by its index (connected_depth_foreground()'s
    # `through`): none of its own here.
    throughs = {}

    fitted = []

    def fit_among(candidates, said_too=""):
        return _locate_fits(model, K, candidates, scales, why, said_too, throughs, image, fitted, kept)

    def measured(region, through=None):
        """(view, its two longest extents, longest first), or (None, why not)."""
        try:
            view = observe(region, metres, K)
        except Exception as exc:
            return None, f"not observed ({exc})"
        if view is None or view[2].sum() < MIN_BOX_POINTS:
            return None, "too few points off its support"
        cam = view[0]
        surface = cam[connected_depth_foreground(view[2], cam[..., 2], cam, through)]
        # Its own draw, so a proposal's box does not depend on those before it.
        surface = surface[np.random.default_rng(0).permutation(len(surface))[:LOCATE_BOX_POINTS]]
        return view, np.sort(min_obb(surface)[2])[::-1]
    for index, mask in enumerate(proposals):
        if any(mask[r, c] for r, c in clicked):
            why.append((index, "on another object's click"))
            continue
        if taken is not None:
            # The bar through a ring's hole, in the ring's proposal, is the bar's.
            left = None
            for lose in ([grown, taken] if grown is not None else [taken]):
                share = (mask & lose).sum() / mask.sum()
                left = mask & ~lose
                if share > LOCATE_TAKEN_SHARE:
                    if rest:
                        band = np.ones((2 * image_px(LOCATE_REGION_GROW_PX, metres.shape) + 1,) * 2, np.uint8)
                        left = cv2.morphologyEx(left.astype(np.uint8), cv2.MORPH_OPEN, band) > 0
                    if not rest or left.sum() < LOCATE_DEPTH_MIN_PX:
                        left = None
                        continue
                break
            if left is None:
                why.append((index, f"{share:.0%} on another object"))
                continue
            mask = left
        if robot is not None:
            if (mask & robot).sum() > LOCATE_TAKEN_SHARE * mask.sum():
                on_robot += 1
                why.append((index, f"{(mask & robot).sum() / mask.sum():.0%} on the robot"))
                continue
            # What of the robot it does take in is still not the part.
            mask = mask & ~robot
        z = metres[mask & valid]
        if len(z) < MIN_BOX_POINTS:
            why.append((index, f"{len(z)} depth points"))
            continue
        area = mask.sum() * (np.median(z) / f) ** 2
        if not area_range[0] <= area <= area_range[1]:
            why.append((index, f"{area * 1e4:.3g} cm2, not {area_range[0] * 1e4:.2g}-"
                               f"{area_range[1] * 1e4:.0f}"))
            continue
        view, e = measured(mask)
        if view is None:
            why.append((index, e))
            continue
        if not low * m[2] <= e[0] <= high * m[0]:
            why.append((index, f"{e[0] * 1000:.0f} mm long, not {low * m[2] * 1000:.0f}-"
                               f"{high * m[0] * 1000:.0f}"))
            continue
        miss = float(np.min(np.abs(np.log(e[0] / max(e[1], 1e-4)) - pairs)))
        if miss > np.log(model.stretch_max ** 2):           # LOCATE_RATIO_SLACK, the mesh's own
            said = f"{e[0] * 1000:.0f} x {e[1] * 1000:.0f} mm, proportions x{np.exp(miss):.2f} off the mesh's"
            # Maybe the part with another in it: fitted whole, last.
            mixed.append((miss, index, view))
            why.append((index, said))
            continue
        shortlist.append((miss, index, view))
    best, tried, screened, chosen = fit_among(shortlist)
    views = {index: view for _, index, view in shortlist}
    cut = chosen is not None and at_edge(views[chosen][1])
    if (best is None or not best[0][0] or cut) and mixed:
        # None fits, or only on what the image's edge cuts, whose size and
        # proportions are not seen: the part may be in a proposal with another
        # (see the comment after LOCATE_RATIO_SLACK). RR-S1-1's bar, the ring on it in its proposal,
        # was located on the arm of the person holding it, at the image's edge.
        found = fit_among(mixed, " (with another part in it)")
        if found[0] is not None and found[0][0][0] and (best is None or not best[0][0] or found[0][0] > best[0]):
            best, tried, screened, chosen = found[0], tried + found[1], screened + found[2], found[3]
            shortlist = shortlist + mixed
    if not shortlist and not mixed:
        return None, None, (f"none of {len(proposals)} proposals ({on_robot} on the robot) "
                            f"is the mesh's size and shape"), None
    if best is None or not best[0][0]:
        return None, None, (f"the mesh fits none of {tried} likely proposals "
                            f"({on_robot} on the robot)"), None
    (_, score), pose = best
    found = mesh_click(model, *pose, metres, K)
    if found is None:
        return None, None, "the fitted mesh shows no surface in the view", None
    # Other places it fits about as well (LOCATE_TAKEN_SHARE): their clicks.
    rivals = []
    for (accepted, other), other_pose in fitted:
        if other_pose is pose or not accepted or other < score - LOCATE_BETTER:
            continue
        there = mesh_click(model, *other_pose, metres, K)
        if there is None:
            continue
        shared = (there[1] & found[1]).sum() / max(1, min(there[1].sum(), found[1].sum()))
        if shared <= LOCATE_TAKEN_SHARE and all(
                (there[1] & r[1]).sum() / max(1, min(there[1].sum(), r[1].sum())) <= LOCATE_TAKEN_SHARE
                for r in rivals):
            rivals.append((there[0], there[1], other))
    note = (f"score {score:.2f}, scale {_overall(pose[2]):.2f}, best of {tried} fitted of "
            f"{len(screened)} screened of {len(shortlist)} shortlisted of {len(proposals)} proposals, "
            f"{on_robot} on the robot")
    if rivals:
        note += f"; {len(rivals)} other place{'s' if len(rivals) > 1 else ''} fit within {LOCATE_BETTER:.2f}"
    return found[0], found[1], note, (score, _overall(pose[2]), pose[2], [(c, s) for c, _, s in rivals])


def side_by_side(calls, most=None):
    """Each of `calls` (no-argument callables) made at once, up to `most`
    (FIT_SIDE_BY_SIDE) at a time, on threads of their own: (result,
    exception or None, seconds) of each, in order. The fits share the fit
    workers (fit_jobs(), thread-safe), and each is what it is alone."""
    def made(call):
        started = time.time()
        try:
            return call(), None, time.time() - started
        except Exception as exc:
            return None, exc, time.time() - started
    if len(calls) < 2:
        return [made(call) for call in calls]
    with ThreadPoolExecutor(max_workers=min(len(calls), most or FIT_SIDE_BY_SIDE),
                            thread_name_prefix="fit_side_by_side") as pool:
        return list(pool.map(made, calls))


def _digest(*items):
    """A hash of `items` by content: arrays (dtype, shape, values), and
    tuples, lists, numbers, strings and None in them."""
    h = hashlib.blake2b(digest_size=16)

    def feed(item):
        if isinstance(item, np.ndarray):
            h.update(f"a{item.dtype.str}{item.shape}".encode())
            h.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, (tuple, list)):
            h.update(f"t{len(item)}".encode())
            for part in item:
                feed(part)
        else:
            h.update(f"v{type(item).__name__}:{item!r};".encode())
    for item in items:
        feed(item)
    return h.digest()


def _kept_quick_fits(kept, model, K, items, screen=False, image=None):
    """_quick_fit() of each (view, through) of `items`, those not made yet
    side by side (side_by_side()), or None where one failed. `kept`, one
    locate's fits by their inputs (locate_unclicked()), or None: a later
    round sees most proposals as the one before did, and their fits,
    deterministic, come out the same -- RR-S1-1 and HS-S1-1 fitted 9 of
    theirs twice over (2026-11-12). Each gets a copy, its own object as if
    fitted afresh: locate_with_mesh() tells fits apart by identity."""
    kept = kept if kept is not None else types.SimpleNamespace(done={}, reused=0)
    keys = [(id(model), screen, _digest(view, K, through, image)) for view, through in items]
    new = {}
    for key, (view, through) in zip(keys, items):
        if key in kept.done or key in new:
            kept.reused += 1
        else:
            new[key] = functools.partial(_quick_fit, model, view, K, screen=screen, through=through,
                                         image=image)
    failed = {}
    for key, (found, exc, _) in zip(new, side_by_side(list(new.values()))):
        if exc is None:
            kept.done[key] = found
        else:
            failed[key] = exc
    results = []
    for key in keys:
        if key in failed:
            traceback.print_exception(failed[key])
            results.append(None)
        else:
            results.append(copy.deepcopy(kept.done[key]))
    return results


def _locate_fits(model, K, candidates, scales, why, said_too="", throughs=None, image=None, fitted=None,
                 kept=None):
    """(best fit or None, proposals fitted, screened, the best's proposal) of
    `candidates` [(miss, index, view)] (locate_with_mesh()): rough fits rank
    them, the best LOCATE_FITS are fitted properly; `why` is told of each.
    `throughs`, by index, what joins a candidate's pieces (_quick_fit());
    `fitted`, a list, is given each proper fit within `scales`; `kept`, the
    fits already made (_kept_quick_fits()). The rough fits are made side by
    side, then the proper ones."""
    throughs = throughs or {}
    screened, chosen = [], None
    ranked = sorted(candidates, key=lambda item: item[0])
    for _, index, _ in ranked[LOCATE_SCREENS:]:
        why.append((index, "shortlisted, past the screening cap"))
    ranked = ranked[:LOCATE_SCREENS]
    roughs = _kept_quick_fits(kept, model, K, [(view, throughs.get(index)) for _, index, view in ranked],
                              screen=True, image=image)
    for (_, index, view), found in zip(ranked, roughs):
        if found is None:
            why.append((index, "shortlisted, no rough fit"))
            continue
        within = scales is None or scales[0] <= _overall(found[1][2]) <= scales[1]
        screened.append(((within,) + found[0], index, view))
    screened.sort(key=lambda item: item[0], reverse=True)
    for (_, accepted, rough), index, _ in screened[LOCATE_FITS:]:
        why.append((index, f"rough fit {rough:.2f}" + ("" if accepted else " refused")))
    top = screened[:LOCATE_FITS]
    fines = _kept_quick_fits(kept, model, K, [(view, throughs.get(index)) for _, index, view in top],
                             image=image)
    best, tried = None, len(top)
    for ((_, accepted, rough), index, view), found in zip(top, fines):
        said = f"rough fit {rough:.2f}" + ("" if accepted else " refused")
        if found is None:
            why.append((index, f"{said}, no fit"))
            continue
        overall = _overall(found[1][2])
        if scales is not None and not scales[0] <= overall <= scales[1]:
            why.append((index, f"{said}, fit {found[0][1]:.2f} at scale {overall:.2f}, "
                               f"not {scales[0]:.2f}-{scales[1]:.2f}"))
            continue
        why.append((index, f"{said}, fit {found[0][1]:.2f}" + ("" if found[0][0] else " refused") + said_too))
        if fitted is not None:
            fitted.append(found)
        if best is None or found[0] > best[0]:
            best, chosen = found, index
    return best, tried, screened, chosen if best is not None else None


def decode_models(meta, parts, count):
    """One ObjectModel or None per object, from the frames after rgb and depth.

    meta["models"] holds, per object, None or the vertex and face counts of
    one frame: zlib of little-endian float32 vertices then int32 faces, then
    with "colours" an RGB byte triple per vertex (LOCATE_COLOUR_DELTA) -- and,
    where the mesh's scale to the part is known, that ("known_scale"). A
    model that cannot be used returns None. The caller retains which objects
    requested a mesh so a decode failure cannot enable point-cloud fitting.
    """
    specs = meta.get("models") or []
    models = [None] * count
    frames = iter(parts)
    for index, spec in enumerate(specs[:count]):
        if spec is None:
            continue
        buf = next(frames, None)
        try:
            if buf is None:
                raise ValueError("no frame")
            raw = zlib.decompress(buf)
            nv, nf = int(spec["vertices"]), int(spec["faces"])
            coloured = bool(spec.get("colours"))
            if nv <= 0 or nf <= 0 or len(raw) != 12 * (nv + nf) + (3 * nv if coloured else 0):
                raise ValueError("size does not match its counts")
            vertices = np.frombuffer(raw, "<f4", 3 * nv).reshape(nv, 3)
            faces = np.frombuffer(raw, "<i4", 3 * nf, offset=12 * nv).reshape(nf, 3)
            colours = np.frombuffer(raw, np.uint8, 3 * nv, offset=12 * (nv + nf)).reshape(nv, 3) if coloured else None
            known = spec.get("known_scale")
            unit = spec.get("unit_m")
            unit = 1.0 if unit is None else float(unit)
            stretch = spec.get("stretch_max")
            stretch_declared = stretch is not None
            models[index] = ObjectModel(vertices, faces, colours=colours, unit=unit,
                                        stretch_max=MODEL_STRETCH_MAX if stretch is None else float(stretch))
            # A known scale is the part's size against the mesh's own (1: the
            # mesh is the part's size), so in its units' terms it is that many units.
            known = None if known is None else float(known) * unit
            lo, hi = models[index].scale_range
            if known is not None and not (np.isfinite(known) and lo <= known <= hi):
                raise ValueError(f"known scale {known} outside {lo:.3g}-{hi:.3g} (a part 5 mm to 1 m long)")
            models[index].known_scale = None if known is None else float(known)
            error = spec.get("mesh_error")
            models[index].tol_declared = error is not None
            models[index].stretch_declared = stretch_declared
            if error is not None:
                if not (np.isfinite(error) and 0 <= error < 0.5):
                    raise ValueError(f"mesh_error {error}: a share of the mesh's diagonal, 0 to 0.5")
                models[index].tol_share = float(error) * TOL_PER_P95
            size = " x ".join(f"{v * unit * 1000:.0f}" for v in models[index].extent)
            print(f"[server] object {index}: mesh {nv} vertices, box {size} mm"
                  + ("" if known is None else f", scale known: {float(known):.3g}")
                  + ("" if unit == 1.0 else f", {unit:g} m a unit")
                  + ("" if stretch is None else f", proportions within {float(stretch):g}")
                  + (", with colours" if coloured else ""), flush=True)
        except Exception as exc:
            print(f"[server] object {index}: unusable mesh ({exc}); recognition rejected",
                  flush=True)
    return models


def box_in_camera(motion, box0, extent, K, image_shape):
    """The init-frame box carried into the current camera by the object's motion.

    ``motion`` is the pipeline's motion since the init frame, the same one the
    anchors ride on. ``rect`` is the image-space [x0, y0, x1, y1] of the
    projected corners, None if any corner is behind the camera or the box lies
    fully outside the image.
    """
    motion = as_numpy(motion)
    if motion.shape != (4, 4) or not np.all(np.isfinite(motion)):
        return None
    T_cam_box = motion @ box0
    extent = as_numpy(extent)
    corners = (_CORNER_SIGNS * 0.5 * extent) @ T_cam_box[:3, :3].T + T_cam_box[:3, 3]

    rect = None
    if np.all(corners[:, 2] > 0):
        uv = corners @ K.T
        uv = uv[:, :2] / uv[:, 2:]
        h, w = image_shape
        x0, y0 = np.clip(uv.min(0), 0, [w - 1, h - 1])
        x1, y1 = np.clip(uv.max(0), 0, [w - 1, h - 1])
        if x1 > x0 and y1 > y0:
            rect = [float(x0), float(y0), float(x1), float(y1)]
    return {
        "pose": T_cam_box.tolist(),
        "extent": extent.tolist(),
        "corners": corners.tolist(),
        "rect": rect,
    }


class PoseServer:
    def __init__(self, cfg_path, bind, archive=None, trace_gpu=False):
        # Where each live GPU block was allocated, kept by torch (GPU_HELD_WARN_MIB).
        self.trace_gpu = bool(trace_gpu) and self.start_gpu_trace()
        self.gpu_after_init = None
        self.gpu_growth_said = 0
        # Where bindings settled file what was seen of their parts; see BINDING_FEATURE_PX.
        self.archive_dir = archive
        self.bindings = []
        self.pending_binds = {}
        self.cfg = OmegaConf.load(cfg_path)
        if self.cfg.pipeline.type != "modular":
            raise ValueError(f"expected modular pipeline, got {self.cfg.pipeline.type}")
        self.cfg.pipeline.params.save_pose = False
        self.cfg.pipeline.params.save_meta_data = False

        self.pipeline = None
        self.frame_id = 0
        self.anchors = None
        self.boxes = None
        self.fitted_models = None
        self.model_required = None
        self.model_errors = None
        self.bbox_warned = False
        self.mesh_fit_executor = None
        self.mesh_fit_job = None
        self.mesh_session = 0
        self.surfaces = []
        self.smoother = PoseSmoother()
        self.init_report = None
        # Each mesh's measured scales, by its id, across inits; see LOCATE_KNOWN_SCALES.
        self.known_scales = {}

        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.bind(bind)
        print(f"[server] listening on {bind}", flush=True)

    def release_pipeline(self):
        """Drop the pipeline and its GPU memory before another is built: built
        first, the new one's state lands beside the old one's, and an 8 GB card
        then spills into system memory, where a frame takes tens of seconds.
        The SAM2 and TAPIR models stay, for the next pipeline (keep_models)."""
        self.pipeline = None
        # A fit still running belongs to the old session; stop waiting on it.
        self.mesh_fit_job = None
        Sam2RealTimeSegmenter.drop_kept_state()
        gc.collect()
        # The SAM2 segmenter enters a bf16 autocast at import and never leaves
        # it, so autocast's cache of cast weights is never cleared and keeps
        # every released SAM2's bf16 copy (~430 MiB each) alive.
        torch.clear_autocast_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def start_gpu_trace(self):
        """Have torch's allocator keep the Python stack of each live block
        (GPU_HELD_WARN_MIB): those alive now only, no history to bound."""
        if not torch.cuda.is_available():
            print("[server] --trace-gpu-memory: no CUDA device, nothing to trace", flush=True)
            return False
        torch.cuda.memory._record_memory_history(enabled="state", context="state", stacks="python")
        print("[server] tracing GPU allocations (Python stacks of live blocks)", flush=True)
        return True

    def report_gpu_held(self, when):
        """Say when more than GPU_HELD_WARN_MIB is still allocated `when` (a
        release): what an earlier session left behind; traced, what holds it."""
        if not torch.cuda.is_available():
            return
        held = torch.cuda.memory_allocated() / 2**20 - kept_model_mib()
        if held <= GPU_HELD_WARN_MIB:
            return
        print(f"[server] WARNING: {held:.0f} MiB of GPU memory still allocated {when}, held from an earlier "
              "session; a fresh server holds next to none, and past the card's memory every step spills "
              "into system memory and slows"
              + ("" if self.trace_gpu else " -- restart the server, or run it with --trace-gpu-memory "
                 "to see what holds it"), flush=True)
        if self.trace_gpu:
            self.say_gpu_sites(f"held {when}")
            self.say_gpu_holders()

    def say_gpu_sites(self, what):
        try:
            total, sites = gpu_alloc_sites()
            print(f"[server] GPU {what}: {total:.0f} MiB in live blocks, by where allocated:", flush=True)
            for size, site in sites:
                print(f"[server]   {size:8.1f} MiB  {site}", flush=True)
        except Exception:
            traceback.print_exc()

    def say_gpu_holders(self):
        try:
            reached, graphed, holders = gpu_python_holders(self)
            held = torch.cuda.memory_allocated() / 2**20
            print(f"[server] GPU held: {reached:.0f} MiB reached from Python objects ({graphed:.0f} MiB of it in "
                  f"tensors made with grad on, an autograd graph's), {max(held - reached, 0):.0f} "
                  "MiB not (held from C++, such as a thread's autocast cache); threads alive: "
                  + ", ".join(t.name for t in threading.enumerate()) + "; the Python part by what holds it:",
                  flush=True)
            for size, holder in holders:
                print(f"[server]   {size:8.1f} MiB  {holder}", flush=True)
        except Exception:
            traceback.print_exc()

    def trace_gpu_frame(self, frame_id):
        """Traced, every GPU_TRACE_EVERY_FRAMES track frames: the memory, and
        where it was allocated each time it has grown another
        GPU_HELD_WARN_MIB over the init's."""
        if not self.trace_gpu or frame_id % GPU_TRACE_EVERY_FRAMES:
            return
        self.log_gpu_memory(f"on frame {frame_id}")
        if self.gpu_after_init is None:
            return
        steps = int((torch.cuda.memory_allocated() / 2**20 - self.gpu_after_init) // GPU_HELD_WARN_MIB)
        if steps > self.gpu_growth_said:
            self.gpu_growth_said = steps
            self.say_gpu_sites(f"grown {steps * GPU_HELD_WARN_MIB} MiB or more since the init, on frame {frame_id}")

    def log_gpu_memory(self, when):
        if torch.cuda.is_available():
            print(f"[server] GPU memory {when}: "
                  f"allocated {torch.cuda.memory_allocated() / 2**20:.0f} MiB "
                  f"({kept_model_mib():.0f} MiB of it kept models), "
                  f"reserved {torch.cuda.memory_reserved() / 2**20:.0f} MiB", flush=True)

    def build_pipeline(self):
        self.archive_session()
        self.release_pipeline()
        self.log_gpu_memory("after release")
        self.report_gpu_held("after release")
        self.pipeline = ModularPipeline(self.cfg)
        self.log_gpu_memory("after build")
        self.bindings, self.pending_binds = [], {}
        self.surfaces = []
        self.frame_id = 0
        self.anchors = None
        self.boxes = None
        self.fitted_models = None
        self.model_required = None
        self.model_errors = None
        self.bbox_warned = False
        self.obstacles = []
        self.smoother.reset()

    def fit_boxes(self, rgb, metres, K, points, labels, models, model_required=None, located=None,
                  robot=None, carried=None, meanwhile=None):
        """Each object's box on the init frame, fit to its mask; None where it cannot be.

        `meanwhile`, a no-argument callable, is made on this thread while the
        meshes are fitted (INIT_STEP_WITH_FITS), once the masks are made;
        not made where there is no mesh to fit, nor where the masks fail.

        A part in `carried` (carry_kept_fits()) keeps its fit as last seen
        where this view does not say it lies otherwise.

        A requested mesh gives the box when its pose fits the view, finely or
        coarsely. When it does not, the object is still tracked, by the box of
        its points, and the mesh is retried on later frames; the reply then
        carries no mesh pose for it, so the Agent knows it has only the box.

        Each mask is taken without the regions other objects were located in
        (`located`, {index: region}) and the robot's pixels (`robot`): the
        ring hung on RR-S1-1's bar was in the bar's, and its fit refused (95th
        percentile 15.7 mm off); the obstacle beside the arm and cart took them
        in, and its box came out 730 mm long of 284.
        """
        self.mesh_session = getattr(self, "mesh_session", 0) + 1
        # Clicked parts whose points took in what is not them (points_stray()).
        self.init_strays = {}
        self.models = list(models)
        self.fitted_models = [None] * len(points)
        self.model_required = (list(model_required) if model_required is not None
                               else [model is not None for model in models])
        self.model_errors = [None] * len(points)
        self.model_retry_after = [time.monotonic() + MODEL_RETRY_INTERVAL_SEC] * len(points)
        self.model_check_fails = [0] * len(points)
        self.model_withheld = [False] * len(points)
        self.model_corrections = [None] * len(points)
        self.sent_model_poses = [None] * len(points)          # see held_turn()
        self.bindings = [None] * len(points)                  # see BINDING_CONFIDENCE
        # Where each fit was last on the part, as its motion then, whether it
        # is held there now, and the robot's poses of the parts it set down
        # (see MODEL_CHECK_FAILS).
        self.still_motions = [None] * len(points)
        self.holding = [False] * len(points)
        self.pose_priors, self.last_fitted = {}, {}
        # Each part's hint: the other located parts' scale, where they share
        # one (LOCATE_SCALE_AGREE), for a part too hidden to measure its own
        # (MODEL_SCALE_MAX_HIDDEN).
        # Its own measured scale first (LOCATE_KNOWN_SCALES), and its shape
        # tried on every view (MODEL_KNOWN_SHAPE_MIN_GAIN).
        scales = getattr(self, "locate_scales", {}) if getattr(self, "same_scale", False) else {}
        self.scale_hints = []
        self.shape_hints = [self.known_shape(model) for model in models]
        for index in range(len(points)):
            hint = self.known_scale(models[index])
            others = [s for j, s in scales.items() if j != index]
            if hint is None and others:
                hint = float(np.exp(np.median(np.log(others))))
            self.scale_hints.append(hint)
        try:
            # Same call as a preview, made on the freshly built pipeline before
            # add_user_points(), which is the state a preview sees.
            logits = self.preview_masks(rgb, points, labels)
            masks = (as_numpy(logits) > 0).reshape(len(points), *metres.shape).copy()
        except Exception:
            self.say("no init masks; mesh objects rejected, no boxes")
            self.model_errors = ["no usable initial mask" if required else None
                                 for required in self.model_required]
            traceback.print_exc()
            return [None] * len(points)
        located = located or {}
        cut = [others_region(index, [located.get(j) for j in range(len(masks))]) for index in range(len(masks))]
        for index in range(len(masks)):
            if cut[index] is not None:
                # Grown as when located (LOCATE_REGION_GROW_PX), but never
                # into the part's own region: the ring the bar runs through.
                if located.get(index) is not None:
                    cut[index] = _grown(cut[index]) & ~located[index]
                masks[index] &= ~cut[index]
            if robot is not None:
                masks[index] &= ~robot
        views = []
        for mask in masks:
            try:
                views.append(observe(mask, metres, K))
            except Exception:
                traceback.print_exc()
                views.append(None)
        kept = {}
        for index, fit in (carried or {}).items():
            if not self.model_required[index]:
                continue
            try:
                why = carried_fit_rejection(fit["model"], fit["pose"], masks[index], metres, K, through=cut[index])
            except Exception as exc:
                traceback.print_exc()
                why = f"{type(exc).__name__}: {exc}"
            if why:
                self.say(f"object {index}: its fit as last seen (frame {fit['frame']}) is off this view ({why}); "
                         "fitted afresh")
            else:
                kept[index] = fit
        self.kept_fits = set(kept)
        # The parts' fits are independent: made side by side (FIT_SIDE_BY_SIDE).
        fitting = [index for index, view in enumerate(views)
                   if self.model_required[index] and models[index] is not None and view is not None
                   and index not in kept]
        calls = [functools.partial(
            fit_model_box, models[index], views[index], K, through=cut[index],
            scale_hint=self.scale_hints[index], located=getattr(self, "locate_shapes", {}).get(index),
            image=rgb, known_shape=self.shape_hints[index], budget=MODEL_INIT_FIT_BUDGET_SEC) for index in fitting]
        if meanwhile is None or not calls:
            fits = dict(zip(fitting, side_by_side(calls)))
        else:
            if getattr(self, "init_fit_executor", None) is None:
                self.init_fit_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="init_fits",
                                                            initializer=no_grad_thread)
            job = self.init_fit_executor.submit(side_by_side, calls)
            try:
                meanwhile()
            finally:
                # Waited for even when `meanwhile` failed: the next init shares the workers.
                results = job.result()
            fits = dict(zip(fitting, results))
        boxes = []
        for index, mask in enumerate(masks):
            fit, view = None, views[index]
            if index in kept:
                # In this init's frame, its motion the identity, as a fit made on it.
                model = self.models[index] = self.fitted_models[index] = kept[index]["model"]
                model.fitted_pose = np.asarray(kept[index]["pose"], dtype=np.float64).copy()
                self.bindings[index] = kept[index]["binding"]
                self.sent_model_poses[index] = kept[index]["sent"]
                T, extent = fitted_box(model)
                size = " x ".join(f"{v * 1000:.0f}" for v in extent)
                self.say(f"object {index}: box {size} mm (kept fit {model.fitted_id[:6]} as last seen, frame "
                         f"{kept[index]['frame']}: this view does not say otherwise; "
                         f"{'measured' if getattr(model, 'fitted_measured', True) else 'size not measured'})")
                boxes.append((T, extent))
                continue
            if self.model_required[index] and models[index] is None:
                self.model_errors[index] = "supplied mesh could not be loaded"
                self.say(f"object {index}: supplied mesh could not be loaded; tracking its points' box")
                if view is not None:
                    try:
                        fit = fit_box(mask, metres, K, view=view)
                    except Exception:
                        traceback.print_exc()
            elif self.model_required[index]:
                reason, seconds = "no usable initial depth observation", 0.0
                if view is not None:
                    found, exc, seconds = fits[index]
                    if exc is None:
                        fit, reason = found
                    else:
                        traceback.print_exception(exc)
                        reason = f"{type(exc).__name__}: {exc}"
                self.model_retry_after[index] = time.monotonic() + MODEL_RETRY_INTERVAL_SEC
                if fit is None:
                    self.model_errors[index] = reason
                    if getattr(reason, "strays", False) and index not in (located or {}) and len(points[index]):
                        self.init_strays[index] = reason
                    self.say(f"object {index}: mesh pose unlocalized in {seconds:.1f} s "
                             f"({reason}); tracking its points' box, retrying the mesh on fresh frames")
                    if view is not None:
                        try:
                            fit = fit_box(mask, metres, K, view=view)
                        except Exception:
                            traceback.print_exc()
                else:
                    self.fitted_models[index] = models[index]
                    self.remember_scale(models[index])
                    self.say(f"object {index}: mesh fit in {seconds:.1f} s")
            elif view is not None:
                try:
                    fit = fit_box(mask, metres, K, view=view)
                except Exception:
                    traceback.print_exc()
            if fit is None:
                self.say(f"object {index}: no box")
                boxes.append(None)
                continue
            T, extent, note = fit
            size = " x ".join(f"{v * 1000:.0f}" for v in extent)
            self.say(f"object {index}: box {size} mm ({note})")
            boxes.append((T, extent))
        return boxes

    def retry_pending_meshes(self, motions, frame, regions=None):
        """Poll one background fit and schedule an unlocalized object's fresh view.

        The pipeline supplies current SAM masks in frame.mask after step().
        No preview with the original click coordinates is used here. The
        worker's result is expressed in its capture camera frame; convert it
        to the tracker's reference before propagating to newer frames.
        """
        now = time.monotonic()
        tracking_lost = [bool(getattr(obj, "lost", False)) for obj in self.pipeline.objects]
        job = getattr(self, "mesh_fit_job", None)
        if job is not None:
            future, index, session, capture_inverse, fitted_on = job
            if session == self.mesh_session and tracking_lost[index]:
                # A loss at any point during fitting breaks motion propagation,
                # even if feature tracking recovers before the worker finishes.
                capture_inverse = None
                self.mesh_fit_job = (future, index, session, None, fitted_on)
            if not future.done():
                return
            self.mesh_fit_job = None
            if session == self.mesh_session:
                self.model_retry_after[index] = now + (MODEL_RETRY_INTERVAL_SEC
                                                       if self.fitted_models[index] is None
                                                       else MODEL_UPGRADE_INTERVAL_SEC)
                try:
                    fit, reason, model = future.result()
                    current = self.fitted_models[index]
                    unchecked = ""
                    if fit is not None and (current is None or fit_rank(model) > fit_rank(current)):
                        # Sent only once on the part on this frame (MODEL_CHECK_FAILS).
                        fitted_in_camera = model.fitted_pose
                        here, reason = self.adopt_refit(index, model, frame, regions, motions, capture_inverse,
                                                        fitted_on)
                        if here is None:
                            fit, unchecked = None, reason
                    if fit is not None and (current is None or fit_rank(model) > fit_rank(current)):
                        T_box, extent, note = fit
                        model.fitted_pose = here
                        turn = None if current is None else refit_turn(current, model.fitted_pose)
                        upgrade = turn is not None and fit_rank(model)[0] and not fit_rank(current)[0]
                        if upgrade:
                            upgrade, why = self.preserve_refit_direction(index, current, model, frame,
                                                                         regions, motions)
                            self.say(f"object {index}: refit direction check: {why}")
                        if turn is not None and not upgrade:
                            # The fit stays, the refit in doubt: see MODEL_REFIT_TURN_DEG.
                            setattr(current, f"fitted_{turn}_doubt",
                                    np.linalg.inv(current.fitted_pose) @ model.fitted_pose)
                            print(f"[server] object {index}: refit better but turned "
                                  f"{'half round about its length' if turn == 'spin' else 'end for end'}; "
                                  f"keeping the fit, the refit sent in doubt ({note})", flush=True)
                        else:
                            if upgrade:
                                # The refined shape was checked at a different
                                # pose; rebuild its box in that same frame.
                                B = model.box[:3, :3]
                                T_box = model.fitted_pose.copy()
                                T_box[:3, :3] = model.fitted_pose[:3, :3] @ B
                                T_box[:3, 3] += model.fitted_pose[:3, :3] @ (model.fitted_linear @ model.box[:3, 3])
                                extent = fitted_axis_scale(model) * model.extent
                                self.boxes[index] = (T_box, extent)
                            else:
                                to_here = here @ np.linalg.inv(fitted_in_camera)
                                self.boxes[index] = (to_here @ T_box, extent)
                            self.fitted_models[index] = model
                            self.object_state(len(self.models))
                            self.still_motions[index] = np.asarray(motions[index], dtype=np.float64).copy()
                            self.remember_scale(model)
                            self.model_errors[index] = None
                            print(f"[server] object {index}: mesh "
                                  f"{'localized' if current is None else 'fit improved'} on retry"
                                  + (", fine over coarse in the published direction; "
                                     "the refit's other direction kept in doubt" if upgrade else "")
                                  + f" ({note})", flush=True)
                    elif current is not None:
                        print(f"[server] object {index}: "
                              + (f"refit not taken ({unchecked})" if unchecked else "refit no better")
                              + "; keeping the fit", flush=True)
                    else:
                        self.model_errors[index] = reason
                        print(f"[server] object {index}: mesh pose unlocalized ({reason}); will retry", flush=True)
                except Exception as exc:
                    self.model_errors[index] = f"mesh retry failed: {type(exc).__name__}: {exc}"
                    traceback.print_exc()

        eligible = [i for i, model in enumerate(self.models)
                    if self.model_required[i] and model is not None
                    and (self.fitted_models[i] is None or fit_rank(self.fitted_models[i]) < (True, True, True))
                    and not tracking_lost[i] and now >= self.model_retry_after[i]]
        if not eligible:
            return
        # Oldest due first, so repeated failures do not starve another object.
        index = min(eligible, key=lambda i: (self.fitted_models[i] is not None, self.model_retry_after[i]))
        self.model_retry_after[index] = now + (MODEL_RETRY_INTERVAL_SEC if self.fitted_models[index] is None
                                               else MODEL_UPGRADE_INTERVAL_SEC)
        try:
            h, w = frame.depth.shape
            masks = as_numpy(getattr(frame, "mask", None))
            if masks.shape == (len(self.models), 1, h, w):
                masks = masks[:, 0]
            if masks.shape != (len(self.models), h, w) or not np.isfinite(masks[index]).all():
                raise ValueError("no usable current segmentation mask")
            if not np.isfinite(frame.depth_factor) or frame.depth_factor <= 0:
                raise ValueError("invalid current depth factor")
            mask = self.without_robot(without_others(masks[index] > 0, index, regions)).copy()
            metres = as_numpy(frame.depth) / frame.depth_factor
            if np.count_nonzero(mask & np.isfinite(metres) & (metres > 0)) < MIN_BOX_POINTS:
                raise ValueError("insufficient current mask/depth pixels")
            motion = np.asarray(motions[index], dtype=float)
            if (motion.shape != (4, 4) or not np.isfinite(motion).all()
                    or not np.allclose(motion[3], [0, 0, 0, 1], atol=1e-6)
                    or not np.allclose(motion[:3, :3].T @ motion[:3, :3], np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(motion[:3, :3]), 1, atol=1e-5)):
                raise ValueError("no usable current tracking transform")
            capture_inverse = np.linalg.inv(motion)
            if getattr(self, "mesh_fit_executor", None) is None:
                self.mesh_fit_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mesh_pose",
                                                            initializer=no_grad_thread)
            future = self.mesh_fit_executor.submit(
                retry_mesh_fit, self.models[index], mask, metres, np.array(frame.intrinsics, copy=True),
                others_region(index, regions), self.scale_hints[index],
                getattr(self, "locate_shapes", {}).get(index), np.array(frame.rgb, copy=True),
                self.known_shape(self.models[index]))
            self.mesh_fit_job = (future, index, self.mesh_session, capture_inverse, frame.id)
            print(f"[server] object {index}: retrying mesh pose on frame {frame.id}", flush=True)
        except Exception as exc:
            reason = f"mesh retry waiting: {exc}"
            if self.fitted_models[index] is None:
                if self.model_errors[index] != reason:
                    print(f"[server] object {index}: {reason}", flush=True)
                self.model_errors[index] = reason

    def carried_regions(self, motions, frame):
        """Where the camera sees each fitted mesh in this frame, carried there
        by its tracked motion (mesh_click()'s region), one per object; None
        for one with no fit, lost, or out of view; None for all without a
        pipeline to ask. For without_others()."""
        models = getattr(self, "fitted_models", None)
        if not models:
            return None
        try:
            metres = as_numpy(frame.depth) / frame.depth_factor
            K = np.asarray(frame.intrinsics, dtype=np.float64)
            lost = [bool(getattr(obj, "lost", False)) for obj in self.pipeline.objects]
        except Exception:
            return None
        out = []
        for index, model in enumerate(models):
            region = None
            if model is not None and model.fitted_pose is not None and index < len(lost) and not lost[index]:
                try:
                    pose = np.asarray(motions[index], dtype=np.float64) @ model.fitted_pose
                    linear = getattr(model, "fitted_linear", None)
                    linear = linear if linear is not None else model.fitted_scale * np.eye(3)
                    B = model.box[:3, :3]
                    found = mesh_click(model, pose[:3, :3], pose[:3, 3],
                                       np.diag(B.T @ np.asarray(linear, np.float64) @ B), metres, K)
                    region = None if found is None else found[1]
                except Exception:
                    region = None
            out.append(region)
        return out

    def check_carried_meshes(self, motions, frame, regions=None):
        """Correct each kept mesh fit, carried into this frame by its tracked
        motion, on this frame's mask and depth (correct_carried_pose()), and
        score it there: off the part, it is withheld from this frame's reply
        (model_withheld); off it MODEL_CHECK_FAILS frames running, it is
        dropped and fitted again now. Returns the motions, each fitted
        object's carrying its corrected mesh pose."""
        motions = np.array(motions, dtype=np.float64)
        models = getattr(self, "fitted_models", None)
        if not models:
            return motions
        withheld = getattr(self, "model_withheld", None)
        if withheld is None or len(withheld) != len(models):
            withheld = self.model_withheld = [False] * len(models)
        withheld[:] = [False] * len(models)
        corrections = getattr(self, "model_corrections", None)
        if corrections is None or len(corrections) != len(models):
            corrections = self.model_corrections = [None] * len(models)
        try:
            h, w = frame.depth.shape
            masks = as_numpy(getattr(frame, "mask", None))
            if masks.shape == (len(models), 1, h, w):
                masks = masks[:, 0]
            if masks.shape != (len(models), h, w):
                return motions
            metres = as_numpy(frame.depth) / frame.depth_factor
            K = np.asarray(frame.intrinsics, dtype=np.float64)
            lost = [bool(getattr(obj, "lost", False)) for obj in self.pipeline.objects]
        except Exception:
            return motions
        fails = getattr(self, "model_check_fails", None)
        if fails is None or len(fails) != len(models):
            fails = self.model_check_fails = [0] * len(models)
        self.object_state(len(models))
        for index, model in enumerate(models):
            if model is not None and model.fitted_pose is not None and lost[index]:
                self.lost_where_held(index, model, metres, K, frame.id)
            elif hasattr(self, "lost_said"):
                self.lost_said.discard(index)
            if model is None or model.fitted_pose is None or lost[index]:
                fails[index] = 0
                corrections[index] = None
                continue
            try:
                motion = motions[index]
                if motion.shape != (4, 4) or not np.isfinite(motion).all():
                    continue
                carried = motion @ model.fitted_pose
                fit_id = getattr(model, "fitted_id", None)
                kept = corrections[index]
                start = carried @ kept[1] if kept is not None and kept[0] == fit_id else carried
                mask = self.without_robot(without_others(np.asarray(masks[index]) > 0, index, regions))
                fit = carried_view(model, mask, metres, K, others_region(index, regions))
                pose, held_here = start, False
                if fit is not None:
                    pose, cost = correct_carried_pose(fit, model, start)
                    move, turn = correction_size(model, carried, pose)
                    if move > correct_max_m(model) or turn > MODEL_CORRECT_MAX_DEG:
                        tracked, tracked_cost = correct_carried_pose(fit, model, carried)
                        if not cost < tracked_cost - MODEL_CORRECT_MIN_GAIN:
                            pose, cost = tracked, tracked_cost
                    # Where the motion and where it was last seen disagree by
                    # more than the correction moves it, the cheaper (MODEL_CHECK_FAILS).
                    still = self.still_motions[index]
                    if still is not None:
                        last = still @ model.fitted_pose
                        move, turn = correction_size(model, pose, last)
                        if (move > correct_max_m(model) or turn > MODEL_CORRECT_MAX_DEG
                                or self.holding[index]):
                            held, held_cost = correct_carried_pose(fit, model, last)
                            # Held, the motion must fit better to carry it again.
                            better = (held_cost <= cost + MODEL_CORRECT_MIN_GAIN if self.holding[index]
                                      else held_cost < cost - MODEL_CORRECT_MIN_GAIN)
                            if better and not carried_fit_rejection(model, held, mask, metres, K, fit=fit):
                                pose, cost, held_here = held, held_cost, True
                corrections[index] = (fit_id, np.linalg.inv(carried) @ pose)
                why = carried_fit_rejection(model, pose, mask, metres, K, fit=fit) if fit is not None else ""
                if held_here and not why and not self.holding[index]:
                    print(f"[server] object {index}: its tracked motion moved it where it was not on frame "
                          f"{frame.id}; held where it was last seen on the part", flush=True)
                if why:
                    # Off the part where its motion carries it: where it was
                    # last on the part, or set down (MODEL_CHECK_FAILS).
                    found = self.held_or_set_down(index, model, fit, mask, metres, K)
                    if found is not None:
                        label, pose = found
                        if not self.holding[index]:
                            print(f"[server] object {index}: the tracked motion carried its fit off the part "
                                  f"on frame {frame.id} ({why}); held {label}", flush=True)
                        self.holding[index], why = True, ""
                        corrections[index] = (fit_id, np.linalg.inv(carried) @ pose)
                elif held_here:
                    self.holding[index] = True
                elif self.holding[index]:
                    print(f"[server] object {index}: carried by its tracked motion again from frame {frame.id}",
                          flush=True)
                    self.holding[index] = False
                if why:
                    # Off the part: another way round in doubt may lie on it (Binding).
                    turned = self.other_way_on(index, model, fit, pose, mask, metres, K, why)
                    if turned is not None:
                        pose, why = turned, ""
                        carried = motion @ model.fitted_pose
                        corrections[index] = (model.fitted_id, np.linalg.inv(carried) @ pose)
                if not why and fit is not None:
                    pose = self.carried_axial(model, fit, pose, mask, metres, K)
                    corrections[index] = (model.fitted_id, np.linalg.inv(motion @ model.fitted_pose) @ pose)
                if why:
                    corrections[index] = None
                else:
                    motions[index] = pose @ np.linalg.inv(model.fitted_pose)
                    self.still_motions[index] = motions[index].copy()
            except Exception:
                traceback.print_exc()
                continue
            if why and not fails[index]:
                print(f"[server] object {index}: mesh pose withheld from frame {frame.id}: "
                      f"the carried fit is off the part ({why})", flush=True)
            fails[index] = fails[index] + 1 if why else 0
            withheld[index] = bool(why)
            if fails[index] >= MODEL_CHECK_FAILS:
                fails[index] = 0
                if not hasattr(self, "last_fitted"):
                    self.last_fitted = {}
                self.last_fitted[index] = self.fitted_models[index]
                self.fitted_models[index] = None
                self.still_motions[index] = None
                self.model_retry_after[index] = time.monotonic()
                self.model_errors[index] = (f"the carried mesh pose no longer fits the view ({why}); "
                                            "fitting it again")
                print(f"[server] object {index}: {self.model_errors[index]}", flush=True)
        return motions

    def carried_axial(self, model, fit, pose, mask, metres, K):
        """For a part round about its long box axis, the carried pose slid
        along its length onto the one end this frame's view shows, where it
        stays on the part; what the view tells of its ends kept for the reply
        (model.carried_axial). Other parts as they are. See _ModelFit.axial()."""
        scale = fitted_axis_scale(model)
        # Round as the mesh is, whatever the fit stretched it to across its length.
        if round_axis(model, 1.0) != int(np.argmax(scale * model.extent)):
            model.carried_axial = None
            return pose
        R, t = pose[:3, :3], pose[:3, 3]
        terms = fit.terms(scale, R, t, *MODEL_CHECK_POINTS)
        if terms is None:
            return pose
        quality = ("fit" if not model_pose_rejection(terms)
                   else "coarse" if not model_pose_rejection(terms, coarse=True) else None)
        _, _, t2, _, report = fit.axial(terms, R, t, scale, quality)
        slid = pose.copy()
        slid[:3, 3] = t2
        if report["slid_m"] and carried_fit_rejection(model, slid, mask, metres, K, fit=fit):
            report["slid_m"] = 0.0
            slid = pose
        model.carried_axial = report
        return slid

    def lost_where_held(self, index, model, metres, K, frame_id):
        """Diagnostic, once a loss: whether the depth still shows a part the
        tracker lost where it was last on the part -- a part set down that
        the tracker does not find again (PP-S3-1, 2026-11-05: lost=[True]
        after the set-down, to the end). The reply is not changed by it."""
        said = getattr(self, "lost_said", None)
        if said is None:
            said = self.lost_said = set()
        still = self.still_motions[index] if index < len(self.still_motions) else None
        if index in said or still is None:
            return
        said.add(index)
        try:
            pose = still @ model.fitted_pose
            found = mesh_click(model, pose[:3, :3], pose[:3, 3], fitted_axis_scale(model), metres, K)
            if found is None:
                print(f"[server] object {index}: tracking lost on frame {frame_id}; where it was last on the "
                      "part is out of view or hidden", flush=True)
                return
            fit = carried_view(model, found[1], metres, K)
            why = "too little depth there" if fit is None else carried_fit_rejection(model, pose, found[1], metres,
                                                                                       K, fit=fit)
            print(f"[server] object {index}: tracking lost on frame {frame_id}; the depth "
                  + (f"no longer shows it where it was last on the part ({why})" if why
                     else "still shows it where it was last on the part") + " -- diagnostic only", flush=True)
        except Exception:
            traceback.print_exc()

    def object_state(self, count):
        """The per-object state of MODEL_CHECK_FAILS's holds, made where missing."""
        for name, empty in (("still_motions", None), ("holding", False)):
            kept = getattr(self, name, None)
            if kept is None or len(kept) != count:
                setattr(self, name, [empty] * count)
        if getattr(self, "pose_priors", None) is None:
            self.pose_priors = {}

    def frame_view(self, index, frame, regions, model, prior=None):
        """(fit, mask, metres, K): object `index`'s view on this frame for
        `model` -- its mask less the others' regions and the robot, as
        check_carried_meshes() takes it -- or None where it shows too little.
        Where the tracker has too little of it and the robot says where it
        is (`prior`, its mesh pose in this frame's camera: a part in the
        hand, lost to the tracker there), the view is taken where the mesh
        lies so (prior_region())."""
        h, w = frame.depth.shape
        masks = as_numpy(getattr(frame, "mask", None))
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        tracked = masks.ndim == 3 and masks.shape[1:] == (h, w) and index < len(masks)
        if not tracked and prior is None:
            return None
        metres = as_numpy(frame.depth) / frame.depth_factor
        K = np.asarray(frame.intrinsics, dtype=np.float64)
        mask = (self.without_robot(without_others(np.asarray(masks[index]) > 0, index, regions)) if tracked
                else np.zeros((h, w), bool))
        if prior is not None and mask.sum() < MIN_BOX_POINTS:
            mask = self.without_robot(without_others(prior_region(model, prior, metres, K), index, regions))
        fit = carried_view(model, mask, metres, K, others_region(index, regions))
        return None if fit is None else (fit, mask, metres, K)

    def best_on_view(self, model, view, candidates):
        """Of `candidates` ((label, pose in this frame's camera)), each
        corrected on the frame's `view` (frame_view()) as a carried pose is,
        the cheapest on the part: (label, pose, cost); and why each not
        ({label: why})."""
        fit, mask, metres, K = view
        best, why = None, {}
        for label, pose in candidates:
            if pose is None or not np.isfinite(pose).all():
                continue
            posed, cost = correct_carried_pose(fit, model, pose)
            off = carried_fit_rejection(model, posed, mask, metres, K, fit=fit)
            if off:
                why[label] = off
            elif best is None or cost < best[2]:
                best = (label, posed, cost)
        return best, why

    def held_or_set_down(self, index, model, fit, mask, metres, K):
        """(label, pose) where the fit is on the part on this frame (`fit` its
        carried_view()) though its motion carries it off: where it was last on
        it (still_motions), or at the robot's pose prior; None if neither. See
        MODEL_CHECK_FAILS."""
        candidates = []
        still = self.still_motions[index] if index < len(self.still_motions) else None
        if still is not None:
            candidates.append(("where it was last seen on the part", still @ model.fitted_pose))
        prior = (getattr(self, "pose_priors", None) or {}).get(index)
        if prior is not None and prior.get("status") in ("adopted", "agrees"):
            candidates.append((f"where the robot set it down ({prior['source']})", prior["pose"]))
        best, _ = self.best_on_view(model, (fit, mask, metres, K), candidates)
        return None if best is None else best[:2]

    def preserve_refit_direction(self, index, current, model, frame, regions, motions):
        """Validate a finer shape in the old direction before publishing it.

        Both stored poses use the tracker's reference frame. Recheck on the
        current camera frame, including motion, and allow only ordinary
        carried-pose corrections. On failure neither model is changed.
        """
        view = self.frame_view(index, frame, regions, model)
        if view is None:
            return False, "no current view; keeping the published direction"
        fit, _, _, _ = view
        motion = np.asarray(motions[index], dtype=np.float64)
        start = motion @ current.fitted_pose
        posed, _ = correct_carried_pose(fit, model, start)
        move, angle = correction_size(model, start, posed)
        if move > correct_max_m(model) or angle > MODEL_CORRECT_MAX_DEG:
            return False, "refined shape needs a large correction; keeping the published direction"
        terms = fit.terms(fitted_axis_scale(model), posed[:3, :3], posed[:3, 3], *MODEL_CHECK_POINTS)
        why = "no fit" if terms is None else model_pose_rejection(terms)
        if why:
            return False, f"published direction is not a fine fit ({why}); other direction left in doubt"
        old_pose = model.fitted_pose.copy()
        new_pose = np.linalg.inv(motion) @ posed
        base = np.linalg.inv(new_pose) @ old_pose
        # Re-express every candidate in the new published frame; include the
        # refit's original pose, which was not a doubt relative to itself.
        doubts = [base @ d for d in [np.eye(4)] + model_doubts(model)]
        doubts = [d for d in doubts if not same_way(model, np.eye(3), d[:3, :3])]
        model.fitted_unlikely = [base @ d for d in getattr(model, "fitted_unlikely", [])]
        model.fitted_pose = new_pose
        model.fitted_doubts = doubts
        model.fitted_spin_doubt = next((d for d in doubts if turn_kind(model, d[:3, :3]) == "spin"), None)
        model.fitted_ends_doubt = next((d for d in doubts if turn_kind(model, d[:3, :3]) == "ends"), None)
        # Measurement claims and axis diagnostics belonged to the other pose.
        model.fitted_measured = bool(terms.get("measured", False))
        model.fitted_axis_source = None
        model.fitted_axial = None
        return True, "fine fit verified in the published direction; other direction left in doubt"

    def adopt_refit(self, index, model, frame, regions, motions, capture_inverse, fitted_on):
        """The pose to send a background refit (`model`, its fitted_pose in
        the camera of frame `fitted_on`) with on this frame, in the tracker's
        reference, or (None, why): carried here by the tracked motion, where it
        was fitted, or at the robot's pose prior -- the cheapest of them on
        the part here. See MODEL_CHECK_FAILS."""
        view = self.frame_view(index, frame, regions, model)
        motion = np.asarray(motions[index], dtype=np.float64)
        if view is None or motion.shape != (4, 4) or not np.isfinite(motion).all():
            return None, f"too little of it in view on frame {frame.id} to check the refit on"
        T = model.fitted_pose
        candidates = [("where it was fitted", T)]
        if capture_inverse is not None:
            candidates.insert(0, ("carried by its motion", motion @ capture_inverse @ T))
        prior = (getattr(self, "pose_priors", None) or {}).get(index)
        if prior is not None:
            candidates.append((f"where the robot set it down ({prior['source']})", prior["pose"]))
        best, why = self.best_on_view(model, view, candidates)
        if best is None:
            return None, (f"the refit on frame {fitted_on} is off the part on frame {frame.id}: "
                          + "; ".join(f"{label}: {off}" for label, off in why.items()))
        label, pose, _ = best
        print(f"[server] object {index}: the refit on frame {fitted_on} is on the part on frame {frame.id} "
              f"{label}", flush=True)
        return np.linalg.inv(motion) @ pose, ""

    def take_priors(self, meta, frame_id):
        """The robot's poses of parts it set down (meta["priors"]: [{"object":
        index, "pose": 4x4, the mesh's pose as model_pose gives it, in this
        frame's camera, "source": what says so}]), kept to be tried
        (apply_priors()). See MODEL_CHECK_FAILS."""
        self.object_state(len(getattr(self, "models", None) or []))
        for prior in meta.get("priors") or []:
            try:
                index = int(prior["object"])
                pose = np.asarray(prior["pose"], dtype=np.float64).reshape(4, 4)
                if (not 0 <= index < len(self.models) or not np.isfinite(pose).all()
                        or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
                        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
                        or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4)):
                    raise ValueError("not an object's rigid pose")
                self.pose_priors[index] = {"pose": pose, "source": str(prior.get("source") or "client"),
                                           "frame": frame_id, "status": "pending", "why": ""}
                print(f"[server] object {index}: pose prior from {self.pose_priors[index]['source']} "
                      f"on frame {frame_id}", flush=True)
            except (KeyError, TypeError, ValueError) as exc:
                print(f"[server] unusable pose prior {prior!r}: {exc}", flush=True)

    def apply_priors(self, motions, frame, regions=None):
        """Try each pending pose prior (take_priors()) on this frame, as one
        more way the part may lie, scored with the fit's: corrected on the
        frame and off the part, it waits for a frame it is on (the gripper
        may still be on it); on it, it is taken where there is no fit or the
        fit is off the part here, agrees where it is the fit's way round,
        and is taken over the fit's -- which stays in doubt (Binding) --
        where the view cannot tell them apart (settle_turns()'s margin),
        refused where the fit's fits better by more. Where it agrees and fits
        this frame better than the carried pose by MODEL_CORRECT_MIN_GAIN, the
        pose is sent there. Returns the motions. See MODEL_CHECK_FAILS."""
        motions = np.array(motions, dtype=np.float64)
        priors = getattr(self, "pose_priors", None) or {}
        for index, prior in priors.items():
            if prior["status"] != "pending":
                continue
            try:
                current = self.fitted_models[index]
                model = current
                if model is None:
                    # Relocating: the mesh at the shape it was last fitted at, else measured.
                    base = getattr(self, "last_fitted", {}).get(index) or self.models[index]
                    if base is None:
                        prior.update(status="refused", why="no mesh for it")
                        continue
                    model = copy.copy(base)
                    if getattr(model, "fitted_linear", None) is None:
                        shape = self.known_shape(model)
                        shape = np.ones(3) * (model.known_scale or model.unit) if shape is None else shape
                        B = model.box[:3, :3]
                        model.fitted_linear, model.fitted_scale = (B * shape) @ B.T, _overall(shape)
                # A part in the hand, lost to the tracker there, is looked for
                # where the robot holds it (HOLD_PRIOR_SOURCE, prior_region()).
                held = prior["pose"] if prior["source"] == HOLD_PRIOR_SOURCE else None
                view = self.frame_view(index, frame, regions, model, prior=held)
                if view is None:
                    prior["why"] = f"too little of it in view on frame {frame.id}"
                    continue
                if held is not None:
                    # Fitted afresh on what the camera sees of it in the hand,
                    # pose and size, and taken only where that fits it finely.
                    self.fit_in_hand(index, view, motions[index], frame, prior)
                    print(f"[server] object {index}: pose prior {prior['status']} ({prior['why']})", flush=True)
                    continue
                best, why = self.best_on_view(model, view, [("prior", prior["pose"])])
                if best is None:
                    prior["why"] = f"off the part on frame {frame.id}: {why.get('prior', '')}"
                    continue
                _, pose_p, _ = best
                motion = np.asarray(motions[index], dtype=np.float64)
                if current is None or self.model_withheld[index]:
                    self.take_prior_as_fit(index, model, view, pose_p, motion, prior)
                    motions[index] = motion
                    continue
                pose_c = motion @ current.fitted_pose
                if same_way(current, pose_c[:3, :3], pose_p[:3, :3]):
                    prior.update(status="agrees", why="the fit's way round")
                    scale = fitted_axis_scale(current)
                    tc = view[0].terms(scale, pose_c[:3, :3], pose_c[:3, 3], *MODEL_CHECK_POINTS)
                    tp = view[0].terms(scale, pose_p[:3, :3], pose_p[:3, 3], *MODEL_CHECK_POINTS)
                    if tc is None or tp["cost"] < tc["cost"] - MODEL_CORRECT_MIN_GAIN:
                        motions[index] = pose_p @ np.linalg.inv(current.fitted_pose)
                else:
                    fit = view[0]
                    scale = fitted_axis_scale(current)
                    tc = fit.terms(scale, pose_c[:3, :3], pose_c[:3, 3], *MODEL_CHECK_POINTS)
                    tp = fit.terms(scale, pose_p[:3, :3], pose_p[:3, 3], *MODEL_CHECK_POINTS)
                    margin = turn_margin(tc, tp)
                    costs = f"cost {tp['cost']:.3f} against the fit's {tc['cost']:.3f}, margin {margin:.3f}"
                    if tp["cost"] > tc["cost"] + margin:
                        prior.update(status="refused", why=f"the fit's way round fits the view better: {costs}")
                    else:
                        binding = self.bindings[index]
                        if binding is None or binding.fit_id != current.fitted_id:
                            binding = self.bindings[index] = Binding(current)
                        turn = np.linalg.inv(pose_c) @ pose_p
                        k = binding.add(turn)
                        if k is None:
                            k = next(j for j, T in enumerate(binding.turns)
                                     if same_way(current, T[:3, :3], turn[:3, :3]))
                        if not binding.alive[k]:
                            binding.alive[k], binding.ruled[k] = True, None
                        if k:
                            self.turn_binding(index, current, binding, k)
                        prior.update(status="adopted", why=f"the view cannot tell it from the fit's way round: {costs}")
                print(f"[server] object {index}: pose prior {prior['status']} ({prior['why']})", flush=True)
                fitted = self.fitted_models[index]
                self.still_motions[index] = pose_p @ np.linalg.inv(fitted.fitted_pose) if fitted is not None else None
                if fitted is not None and prior["status"] == "adopted":
                    motions[index] = pose_p @ np.linalg.inv(fitted.fitted_pose)
            except Exception:
                traceback.print_exc()
        return motions

    def fit_in_hand(self, index, view, motion, frame, prior):
        """A part the robot holds (a HOLD_PRIOR_SOURCE prior), seen where the
        robot holds it (`view`, frame_view() on prior_region()): its mesh
        fitted afresh there, pose and size as an init fits them, and taken --
        a new fit, the part not lost on this frame (hold_seen) -- only where
        that fits it finely. A coarse fit in the hand is no better than none:
        RI-S1-1's ring, its pose corrected at the robot's (2026-11-20), came
        out coarse, in and out of view, and in doubt which way round; the
        robot planned on that, and failed, where it had planned on its hold.
        The prior is spent either way: the robot sends the next."""
        _, mask, metres, K = view
        model = copy.copy(self.models[index])
        seen = observe(mask, metres, K)
        fit, reason = (fit_model_box(model, seen, K, known_shape=self.known_shape(model))
                       if seen is not None else (None, "too little depth where it is held"))
        if fit is None or model.fitted_quality != "fit":
            prior.update(status="refused", why=f"in the hand on frame {frame.id}, "
                                               + (f"its fit coarse: {fit[2]}" if fit is not None else reason))
            return
        motion = np.asarray(motion, dtype=np.float64)
        model.fitted_pose = np.linalg.inv(motion) @ model.fitted_pose
        model.fitted_id = model.fitted_id or uuid.uuid4().hex
        B, scale = model.box[:3, :3], fitted_axis_scale(model)
        T = np.eye(4)
        T[:3, :3] = model.fitted_pose[:3, :3] @ B
        T[:3, 3] = model.fitted_pose[:3, :3] @ (model.fitted_linear @ model.box[:3, 3]) + model.fitted_pose[:3, 3]
        self.boxes[index] = (T, scale * model.extent)
        self.fitted_models[index] = model
        self.still_motions[index] = motion.copy()
        self.model_errors[index] = None
        self.model_withheld[index] = False
        self.model_check_fails[index] = 0
        self.model_corrections[index] = None
        self.__dict__.setdefault("hold_seen", {})[index] = frame.id
        prior.update(status="adopted", why=f"fitted in the hand on frame {frame.id}: {fit[2]}")

    def take_prior_as_fit(self, index, model, view, pose, motion, prior):
        """A pose prior on the part where the object has no fit on it: the
        mesh fitted there, a new fit (see apply_priors())."""
        fit = view[0]
        model = copy.copy(model)
        scale = fitted_axis_scale(model)
        terms = fit.terms(scale, pose[:3, :3], pose[:3, 3], MODEL_SAMPLES, MODEL_FINE_POINTS)
        model.fitted_pose = np.linalg.inv(motion) @ pose
        model.fitted_quality = "fit" if not model_pose_rejection(terms) else "coarse"
        model.fitted_measured = bool(getattr(model, "fitted_measured", False))
        model.fitted_doubts, model.fitted_spin_doubt, model.fitted_ends_doubt = [], None, None
        model.fitted_id = uuid.uuid4().hex
        B = model.box[:3, :3]
        T = np.eye(4)
        T[:3, :3] = model.fitted_pose[:3, :3] @ B
        T[:3, 3] = model.fitted_pose[:3, :3] @ (model.fitted_linear @ model.box[:3, 3]) + model.fitted_pose[:3, 3]
        self.boxes[index] = (T, scale * model.extent)
        self.fitted_models[index] = model
        self.still_motions[index] = np.asarray(motion, dtype=np.float64).copy()
        self.model_errors[index] = None
        self.model_withheld[index] = False
        self.model_check_fails[index] = 0
        self.model_corrections[index] = None
        prior.update(status="adopted", why=f"on the part ({model.fitted_quality}, cost {terms['cost']:.3f}) "
                                           "where the server had no fit on it")

    def update_bindings(self, motions, frame, regions=None, robot=None, observe=True, rank=False):
        """Each fitted mesh's binding (Binding): made for a new fit, carrying
        the evidence of the one before; the ways round a refit held back
        added; the person's choice (meta["bind"]); with `observe`, this
        frame's evidence; and what that rules out applied -- the published
        way round turned to the one left, the doubts those still living.
        `robot`: the robot's pixels, where the part may not be judged at rest.
        `rank`, on the init's view: a way round the ends' silhouettes vote for
        over the published one is published in its place, the other kept in
        doubt (ENDS_SILHOUETTE_SIGMAS). See BINDING_CONFIDENCE."""
        models = getattr(self, "fitted_models", None)
        if not models:
            return
        bindings = getattr(self, "bindings", None)
        if bindings is None or len(bindings) != len(models):
            bindings = self.bindings = [None] * len(models)
        try:
            h, w = frame.depth.shape
            masks = as_numpy(getattr(frame, "mask", None))
            if masks.shape == (len(models), 1, h, w):
                masks = masks[:, 0]
            if masks.shape != (len(models), h, w):
                observe = False
            metres = as_numpy(frame.depth) / frame.depth_factor
            K = np.asarray(frame.intrinsics, dtype=np.float64)
            lost = [bool(getattr(obj, "lost", False)) for obj in self.pipeline.objects]
        except Exception:
            return
        withheld = getattr(self, "model_withheld", None) or [False] * len(models)
        pending = getattr(self, "pending_binds", None) or {}
        for index, model in enumerate(models):
            if model is None or model.fitted_pose is None or getattr(model, "fitted_id", None) is None:
                continue
            try:
                binding = bindings[index]
                if binding is None or binding.fit_id != model.fitted_id:
                    before = (binding if binding is not None and binding.model.model_id == model.model_id
                              else None)
                    binding = bindings[index] = Binding(model, before)
                else:
                    for d in model_doubts(model):
                        binding.add(d)
                chosen = pending.pop(index, None)
                if chosen is not None:
                    ident, by = chosen if isinstance(chosen, tuple) else (chosen, "person")
                    if binding.choose(ident, by):
                        self.say(f"object {index}: way round {ident} chosen by the {by}")
                        if by == "camera" and binding.ids.index(ident):
                            self.turn_binding(index, model, binding, binding.ids.index(ident))
                    else:
                        self.say(f"object {index}: way round {ident} chosen by the {by}, but it is none in doubt; "
                                 "ignored")
                motion = np.asarray(motions[index], dtype=np.float64)
                if (observe and len(binding.living()) > 1 and index < len(lost) and not lost[index]
                        and not withheld[index] and np.isfinite(motion).all()):
                    mask = self.without_robot(without_others(np.asarray(masks[index]) > 0, index, regions))
                    fit = carried_view(model, mask, metres, K, others_region(index, regions))
                    if fit is not None:
                        at_rest = robot is None or robot.shape != fit.grown.shape or not (fit.grown & robot).any()
                        if binding.observe(fit, motion @ model.fitted_pose, at_rest):
                            self.say_ends(index, binding, rank)
                # Not a kept fit's (carry_kept_fits()): its ways round were
                # weighed on the views before, this one among them.
                if rank and len(binding.living()) > 1 and index not in (getattr(self, "kept_fits", None) or ()):
                    k = binding.ends_favoured()
                    if k is not None:
                        self.say(f"object {index}: the init's view favours {binding.ids[k]} end for end over the "
                                 f"published {binding.ids[0]}; publishing it, {binding.ids[0]} kept in doubt")
                        self.turn_binding(index, model, binding, k)
                living_before = len(binding.living())
                kept, changed = binding.decide()
                if kept:
                    self.turn_binding(index, model, binding, kept)
                living = binding.living()
                model.fitted_doubts = [binding.turns[k] for k in binding.in_doubt()]
                model.fitted_unlikely = [binding.turns[k] for k in living if k and binding.unlikely[k]]
                model.fitted_spin_doubt = next((d for d in model.fitted_doubts
                                                if turn_kind(model, d[:3, :3]) == "spin"), None)
                model.fitted_ends_doubt = next((d for d in model.fitted_doubts
                                                if turn_kind(model, d[:3, :3]) == "ends"), None)
                if changed or chosen is not None:
                    post = binding.posterior()
                    left = ", ".join(f"{binding.ids[k]} {post.get(k, 1.0):.2f}" for k in living)
                    self.say(f"object {index}: binding {binding.state()} by {binding.by} after "
                             f"{binding.frames} frame(s) of evidence; {living_before} way(s) round "
                             f"then {len(living)}: {left}"
                             + "".join(f"; {binding.ids[k]} out: {binding.ruled[k][1]}"
                                       for k in range(len(binding.turns)) if not binding.alive[k]))
            except Exception:
                traceback.print_exc()

    def say_ends(self, index, binding, always=False):
        """The last frame's evidence on the ways round turned end for end
        (Binding.vote_ends()), silhouettes and ends: each pair's, where it
        voted or `always`."""
        for (j, k), both in binding.last_ends.items():
            for kind, r in both.items():
                if r["winner"] is None and not always:
                    continue
                if kind == "silhouette":
                    told = (f"{r['for_j']} px for {binding.ids[j]}, {r['for_k']} for {binding.ids[k]}, of {r['told']} "
                            f"told ({r['told'] / max(r['area'], 1):.1%} of the silhouettes' {r['area']})")
                else:
                    told = (f"{binding.ids[j]} {r['off_j'] * 1000:.1f} mm off where the part ends, {binding.ids[k]} "
                            f"{r['off_k'] * 1000:.1f} mm, on {r['ends']} end(s): {r['share']:.1%} of its length apart")
                verdict = (f"a vote for {binding.ids[j] if r['winner'] == 'j' else binding.ids[k]}" if r["winner"]
                           else f"no vote: {r['why']}")
                self.say(f"object {index}: ends' {kind}s {binding.ids[j]} / {binding.ids[k]}: {told}; {verdict}")

    def other_way_on(self, index, model, fit, pose, mask, metres, K, why):
        """Where the published way round, corrected to `pose`, is off the part
        on this frame (`why`), the way round in doubt that lies on it --
        corrected on `fit` as the published one was, the cheapest of those
        carried_fit_rejection() passes -- published in its place; its pose,
        or None where none does. The other stays in doubt: one frame is one
        view, and one view rules nothing out (BINDING_CONFIDENCE)."""
        bindings = getattr(self, "bindings", None) or []
        binding = bindings[index] if index < len(bindings) else None
        if binding is None or binding.fit_id != model.fitted_id or len(binding.living()) < 2:
            return None
        best = None
        for k in binding.living():
            if not k:
                continue
            posed, cost = correct_carried_pose(fit, model, pose @ binding.turns[k], binding.scales[k])
            if (not carried_fit_rejection(model, posed, mask, metres, K, fit=fit, scale=binding.scales[k])
                    and (best is None or cost < best[2])):
                best = (k, posed, cost)
        if best is None:
            return None
        k, posed, _ = best
        self.say(f"object {index}: the published way round {binding.ids[0]} left the part ({why}) where "
                 f"{binding.ids[k]} lay on it; {binding.ids[k]} published, the other still in doubt")
        self.turn_binding(index, model, binding, k)
        return posed

    def turn_binding(self, index, model, binding, k):
        """Publish way round k of `binding` in place of the published one: the
        fit turned to it, with a new id, its box with it."""
        kind = turn_kind(model, binding.turns[k][:3, :3]) or "about its round axis"
        model.fitted_pose = model.fitted_pose @ binding.turns[k]
        binding.rebase(k)
        B, scale = model.box[:3, :3], binding.scale
        model.fitted_linear = (B * scale) @ B.T
        model.fitted_id = binding.fit_id = uuid.uuid4().hex
        boxes = getattr(self, "boxes", None)
        if boxes and index < len(boxes) and boxes[index] is not None:
            T = np.eye(4)
            T[:3, :3] = model.fitted_pose[:3, :3] @ B
            T[:3, 3] = model.fitted_pose[:3, :3] @ (model.fitted_linear @ model.box[:3, 3]) + model.fitted_pose[:3, 3]
            boxes[index] = (T, scale * model.extent)
        for name in ("model_corrections", "sent_model_poses"):
            kept = getattr(self, name, None)
            if kept and index < len(kept):
                kept[index] = None
        self.say(f"object {index}: published way round turned to {binding.ids[0]} "
                 f"({kind} of the one published before)")

    def load_archive(self, models):
        """Each mesh's archive (see the comment after BINDING_FEATURE_PX),
        where kept: its shapes among the known ones, the first time this run
        sees the mesh."""
        if not getattr(self, "archive_dir", None):
            return
        for index, model in enumerate(models):
            if model is None:
                continue
            path = os.path.join(self.archive_dir, f"{model.model_id}.npz")
            if not os.path.exists(path):
                continue
            try:
                with np.load(path) as data:
                    points = len(data["points"])
                    shapes = np.asarray(data["shapes"], np.float64).reshape(-1, 3).tolist()
                known = getattr(self, "known_scales", None)
                if known is None:
                    known = self.known_scales = {}
                if shapes and model.model_id not in known:
                    known[model.model_id] = shapes[-LOCATE_KNOWN_SCALES:]
                self.say(f"object {index}: archive of {len(shapes)} shapes, {points} surface points")
            except Exception as exc:
                print(f"[server] object {index}: unusable archive {path} ({exc})", flush=True)

    def archive_session(self):
        """File, where an archive is kept, each mesh's measured shapes, and the
        surface seen of its part where its binding was settled by more than
        the init view; see BINDING_FEATURE_PX."""
        if not getattr(self, "archive_dir", None):
            return
        models = getattr(self, "models", None) or []
        fitted = getattr(self, "fitted_models", None) or []
        bindings = getattr(self, "bindings", None) or []
        surfaces = getattr(self, "surfaces", None) or []
        for index, model in enumerate(models):
            if model is None:
                continue
            try:
                path = os.path.join(self.archive_dir, f"{model.model_id}.npz")
                points, runs, shapes = np.zeros((0, 3)), np.zeros(0, int), []
                if os.path.exists(path):
                    with np.load(path) as data:
                        points = np.asarray(data["points"], np.float64).reshape(-1, 3)
                        runs = np.asarray(data["runs"]) if "runs" in data else np.ones(len(points), int)
                        shapes = np.asarray(data["shapes"], np.float64).reshape(-1, 3).tolist()
                shapes = (getattr(self, "known_scales", None) or {}).get(model.model_id, shapes)
                added = 0
                fit = fitted[index] if index < len(fitted) else None
                binding = bindings[index] if index < len(bindings) else None
                if (fit is not None and binding is not None and binding.fit_id == fit.fitted_id
                        and binding.state() == "locked" and fit.fitted_quality == "fit"
                        and index < len(surfaces)):
                    seen = surfaces[index].points()
                    linear = np.asarray(fit.fitted_linear if fit.fitted_linear is not None
                                        else fit.fitted_scale * np.eye(3), np.float64)
                    back = np.linalg.inv(fit.fitted_pose)
                    scaled = seen @ back[:3, :3].T + back[:3, 3]
                    B = fit.box[:3, :3]
                    half = fitted_axis_scale(fit) * fit.extent / 2 + 2 * binding.tol
                    inside = np.all(np.abs((scaled - linear @ fit.box[:3, 3]) @ B) <= half, axis=1)
                    new = scaled[inside] @ np.linalg.inv(linear).T
                    added = len(new)
                    voxel = float(np.linalg.norm(fit.extent)) / SURFACE_VOXELS_ACROSS
                    every = np.vstack([points, new])
                    keys, inverse = np.unique(np.floor(every / voxel).astype(np.int64), axis=0, return_inverse=True)
                    inverse = inverse.reshape(-1)
                    sums = np.zeros((len(keys), 3))
                    np.add.at(sums, inverse, every)
                    # A voxel's runs: the most of the old points in it, and one for this run.
                    counted = np.zeros(len(keys), int)
                    np.maximum.at(counted, inverse[:len(points)], runs.astype(int))
                    counted[np.unique(inverse[len(points):])] += 1
                    order = np.argsort(-counted, kind="stable")[:SURFACE_MAX_VOXELS]
                    points = (sums / np.bincount(inverse, minlength=len(keys))[:, None])[order]
                    runs = counted[order]
                if not added and not shapes:
                    continue
                os.makedirs(self.archive_dir, exist_ok=True)
                partial = path + ".partial.npz"
                np.savez(partial, points=np.asarray(points, np.float32), runs=np.asarray(runs, np.int32),
                         shapes=np.asarray(shapes, np.float64))
                os.replace(partial, path)
                print(f"[server] object {index}: archived {added} surface points "
                      f"({len(points)} in all), {len(shapes)} shapes", flush=True)
            except Exception:
                traceback.print_exc()

    def frontend_gates(self, index, metres, region, K):
        """Set object `index`'s front-end gates from its pixels `region` on
        this frame (see TRACK_ERROR_PX)."""
        try:
            obj = self.pipeline.objects[index]
            z = metres[region & np.isfinite(metres) & (metres > 0)]
            if len(z) < MIN_BOX_POINTS:
                return
            sigma = image_noise(metres, region) or 0.0
            across = TRACK_ERROR_PX * float(np.median(z)) / float(K[0, 0])
            gate = TOL_SIGMAS * float(np.hypot(sigma, across))
            obj.reg_residual_thres = gate
            obj.map_growth_max_mean_residual = gate
            obj.pending_obj_spread_thres = gate
        except Exception:
            pass

    def record_surfaces(self, motions, frame, regions=None):
        """Add this frame's view of each tracked object to its SurfaceMemory."""
        try:
            h, w = frame.depth.shape
            masks = as_numpy(getattr(frame, "mask", None))
            if masks.shape == (len(self.surfaces), 1, h, w):
                masks = masks[:, 0]
            if masks.shape != (len(self.surfaces), h, w):
                return
            metres = as_numpy(frame.depth) / frame.depth_factor
            K = np.asarray(frame.intrinsics, dtype=np.float64)
            lost = [bool(getattr(obj, "lost", False)) for obj in self.pipeline.objects]
        except Exception:
            return
        rng = np.random.default_rng(frame.id)
        for index, memory in enumerate(self.surfaces):
            try:
                motion = np.asarray(motions[index], dtype=np.float64)
                if lost[index] or motion.shape != (4, 4) or not np.isfinite(motion).all():
                    continue
                view = observe(self.without_robot(without_others(np.asarray(masks[index]) > 0, index, regions)),
                               metres, K, rng)
                if view is None:
                    continue
                cam, _, keep, _, _ = view
                self.frontend_gates(index, metres, keep, K)
                points = cam[keep]
                points = points[rng.permutation(len(points))[:SURFACE_PIXELS]]
                back = np.linalg.inv(motion)
                memory.add(points @ back[:3, :3].T + back[:3, 3], frame.id, camera=back[:3, 3])
            except Exception:
                continue

    def bboxes(self, motions, frame):
        """Per-object boxes; a failure yields None for that box, never a failed reply.

        recognition_lost() rejects mesh objects missing a box while keeping
        the other objects in the reply usable.
        """
        out = []
        for index, motion in enumerate(motions):
            box = self.boxes[index] if self.boxes and index < len(self.boxes) else None
            if box is None:
                out.append(None)
                continue
            try:
                result = box_in_camera(motion, *box, frame.intrinsics, frame.rgb.shape[:2])
                prior = (getattr(self, "pose_priors", None) or {}).get(index)
                if prior is not None:
                    # What came of the robot's pose of it (MODEL_CHECK_FAILS).
                    result["model_prior"] = {k: prior[k] for k in ("source", "status", "why", "frame")}
                models = getattr(self, "fitted_models", None)
                model = models[index] if models and index < len(models) else None
                withheld = getattr(self, "model_withheld", None)
                if withheld and index < len(withheld) and withheld[index]:
                    model = None                 # off the part on this frame; see check_carried_meshes()
                if model is not None and model.fitted_pose is not None:
                    linear = getattr(model, "fitted_linear", None)
                    pose = motion @ model.fitted_pose
                    held, axis = self.held_turn(index, model, pose)
                    result.update(model_pose=(pose @ held).tolist(),
                                  model_scale=model.fitted_scale, model_id=model.model_id,
                                  model_linear=(np.asarray(linear) if linear is not None
                                                else model.fitted_scale * np.eye(3)).tolist(),
                                  model_quality=getattr(model, "fitted_quality", None) or "fit",
                                  model_measured=bool(getattr(model, "fitted_measured", True)),
                                  model_fit_id=getattr(model, "fitted_id", None),
                                  # What the mesh itself leaves open, in its own
                                  # frame: the axis it is round about, and
                                  # whether its ends are alike.
                                  model_symmetry={"axis": axis,
                                                  "ends": bool(getattr(model, "ends_alike", False))})
                    axial = getattr(model, "carried_axial", None) or getattr(model, "fitted_axial", None)
                    if axial is not None:
                        # Which ends of its length the view showed (_ModelFit.axial()):
                        # this frame's for a bar, else the fit's.
                        result["model_axial"] = axial
                    sources = getattr(model, "fitted_axis_source", None)
                    if sources is not None:
                        # Per box axis of the mesh (its frame): the scale along it, the
                        # part's size there, and where that came from (MODEL_SUPPORT_TOP_PERCENTILE).
                        axis_scale = fitted_axis_scale(model)
                        result["model_axes"] = [{"axis": model.box[:3, j].tolist(), "scale": float(axis_scale[j]),
                                                 "extent_m": float(axis_scale[j] * model.extent[j]),
                                                 "source": sources[j]} for j in range(3)]
                    kept = getattr(model, "kept_vertices", None)
                    if kept is not None:
                        # The box of the mesh less its loose fragments (MODEL_FRAGMENT_GAP).
                        scaled = kept @ np.asarray(result["model_linear"], np.float64).T
                        result["model_bounds"] = [scaled.min(0).tolist(), scaled.max(0).tolist()]
                    doubt = getattr(model, "fitted_spin_doubt", None)
                    if doubt is not None:
                        # Relative to the fit, so carried by the motion with it.
                        result["model_spin_doubt"] = (np.linalg.inv(held) @ doubt @ held).tolist()
                    ends = getattr(model, "fitted_ends_doubt", None)
                    if ends is not None:
                        result["model_ends_doubt"] = (np.linalg.inv(held) @ ends @ held).tolist()
                    # Every way round in doubt (MODEL_TURN_COST_SLACK), with a
                    # refit's held back (MODEL_REFIT_TURN_DEG).
                    doubts = list(getattr(model, "fitted_doubts", []))
                    doubts += [d for d in (doubt, ends)
                               if d is not None and not any(np.allclose(d, k) for k in doubts)]
                    if doubts:
                        sent = [np.linalg.inv(held) @ d @ held for d in doubts]
                        result["model_doubts"] = [d.tolist() for d in sent]
                        # Each one's kind (turn_kind(): "spin" about the long box
                        # axis, "ends" about another), the axis it turns about in
                        # the mesh's frame and by how much; and every "ends" one,
                        # where model_ends_doubt has the first.
                        kinds = []
                        for d, d_sent in zip(doubts, sent):
                            axis = np.linalg.eigh(d_sent[:3, :3] + d_sent[:3, :3].T)[1][:, -1]
                            angle = np.degrees(np.arccos(np.clip((np.trace(d_sent[:3, :3]) - 1) / 2, -1, 1)))
                            kinds.append({"kind": turn_kind(model, d[:3, :3]) or "round",
                                          "axis": (axis * np.sign(axis[np.argmax(np.abs(axis))])).tolist(),
                                          "deg": round(float(angle), 1)})
                        result["model_doubt_kinds"] = kinds
                        result["model_ends_doubts"] = [d.tolist() for d, k in zip(sent, kinds) if k["kind"] == "ends"]
                    # Whether the pose is to be taken as it is: a fine fit, no other
                    # way round in doubt, and every length the view showed both
                    # ends of. Else why not -- for the Agent, one field to read
                    # before it acts on a pose as exact.
                    why = []
                    if result.get("model_quality") != "fit":
                        why.append("coarse fit")
                    if result.get("model_doubts"):
                        why.append("ways round in doubt")
                    if (result.get("model_axial") or {}).get("length_range_m") is not None:
                        why.append("length not seen end to end")
                    if getattr(model, "fitted_search_cut", False):
                        why.append("search cut short")
                    result["model_certain"] = not why
                    result["model_uncertain_why"] = why
                    # How far its binding is, and how to tell its ways round (BINDING_CONFIDENCE).
                    bindings = getattr(self, "bindings", None) or []
                    binding = bindings[index] if index < len(bindings) else None
                    if binding is not None and binding.fit_id == getattr(model, "fitted_id", None):
                        result["model_binding"] = binding.reply(held, pose, np.asarray(frame.intrinsics, np.float64),
                                                                frame.rgb.shape[:2])
                surfaces = getattr(self, "surfaces", None)
                if (surfaces and index < len(surfaces)
                        and (frame.id == 0 or frame.id % SURFACE_PUBLISH_EVERY == 0)):
                    # In the frame of the pose published: the mesh's, else the box's.
                    reference = (model.fitted_pose if model is not None and model.fitted_pose is not None
                                 else box[0])
                    points, views = surfaces[index].points(views=True)
                    if len(points):
                        back = np.linalg.inv(reference)
                        result.update(surface=encode_points(points @ back[:3, :3].T + back[:3, 3]),
                                      surface_views=encode_points(views @ back[:3, :3].T),
                                      surface_frame="model" if reference is not box[0] else "box")
                out.append(result)
            except Exception:
                if not self.bbox_warned:
                    self.bbox_warned = True
                    print("[server] bbox failed; sending None (logged once per session)",
                          flush=True)
                    traceback.print_exc()
                out.append(None)
        return out

    def remember_scale(self, model):
        """Keep a fine fit's shape, its scales along its mesh's box axes, where
        the view measured it: with those measured before, or in their place
        where it fitted the view better than they did; not where theirs was
        kept. See LOCATE_KNOWN_SCALES."""
        if model is not None and model.fitted_quality == "fit" and model.fitted_measured:
            # Its proportions' bound (MODEL_STRETCH_MIN_GAIN's comment).
            shape = np.asarray(fitted_axis_scale(model), np.float64)
            off = float(np.max(np.maximum(shape / _overall(shape), _overall(shape) / shape)))
            bound = model.stretch_max
            # (Not on a stretch the bound stopped alone: that came on every fit of
            # PP-S1-1's sample mesh measuring 1.32, drawing out an axis the view
            # did not show, and its bound grew 1.52, 1.75, 2.01 -- 2026-11-18. A
            # part past the bound shows near it, its proportions descended freely
            # up to it, and widens it a step at a time.)
            wider = off * MODEL_STRETCH_STEP if off * MODEL_STRETCH_STEP > bound else None
            if wider is not None:
                stretch = getattr(self, "learnt_stretch", None)
                if stretch is None:
                    stretch = self.learnt_stretch = {}
                stretch[model.model_id] = min(4.0, max(stretch.get(model.model_id, 0.0), wider))
                print(f"[server] mesh {model.model_id[:6]}: proportions {off:.2f} off the mesh's"
                      + (", wanting past its bound" if getattr(model, "fitted_stretch_wanted", False) else "")
                      + f"; would want {stretch[model.model_id]:.2f}, held at the fixed bound", flush=True)
        error = getattr(model, "fitted_mesh_error", None)
        if model is not None and error is not None:
            learnt = getattr(self, "learnt_errors", None)
            if learnt is None:
                learnt = self.learnt_errors = {}
            learnt[model.model_id] = (learnt.get(model.model_id, []) + [float(error)])[-LOCATE_KNOWN_SCALES:]
            print(f"[server] mesh {model.model_id[:6]}: {error:.2%} off its part at the 95th percentile on this "
                  f"fit; {np.median(learnt[model.model_id]):.2%} on its last {len(learnt[model.model_id])}", flush=True)
        how = getattr(model, "fitted_known_shape", None)
        if model is None or model.fitted_quality != "fit" or not model.fitted_measured or how == "kept":
            return
        known = getattr(self, "known_scales", None)
        if known is None:
            known = self.known_scales = {}
        before = [] if how == "beaten" else known.get(model.model_id, [])
        known[model.model_id] = (before + [fitted_axis_scale(model).tolist()])[-LOCATE_KNOWN_SCALES:]

    def known_shape(self, model):
        """The median, axis by axis, of a mesh's measured shapes, or None."""
        shapes = (getattr(self, "known_scales", None) or {}).get(getattr(model, "model_id", None))
        return np.median(np.asarray(shapes, np.float64).reshape(-1, 3), axis=0) if shapes else None

    def known_scale(self, model):
        """The overall size of a mesh's measured shape, or None."""
        shape = self.known_shape(model)
        return None if shape is None else _overall(shape)

    def held_turn(self, index, model, pose):
        """(turn, axis): the turn of the fitted mesh at `pose` about the axis
        it is round about (round_axis()), in its own frame about its box
        centre, that brings it nearest the pose last sent for object `index`
        -- with none sent, another of its box axes as near the image's up as
        it goes (toward the camera where the axis is the up) -- and that axis
        in the mesh's frame; identity and None for a mesh round about none.
        The pose so turned is remembered as sent. See MODEL_SYMMETRY_STEP_DEG."""
        sent = getattr(self, "sent_model_poses", None) or []
        sent += [None] * (index + 1 - len(sent))
        self.sent_model_poses = sent
        turn = np.eye(4)
        scale = fitted_axis_scale(model)
        k = round_axis(model, scale)
        if k is not None:
            B = model.box[:3, :3]
            a = B[:, k]
            R = pose[:3, :3]
            last = sent[index]
            if last is not None:
                D = last[:3, :3].T @ R
                angle = np.arctan2(np.trace(D @ _skew(a)), np.trace(D) - a @ D @ a)
            else:
                e = B[:, min(j for j in range(3) if j != k)]
                e = e * np.sign(e[int(np.argmax(np.abs(e)))])
                toward = np.array([0.0, -1.0, 0.0])
                if abs((R @ a) @ toward) > 0.9:
                    toward = np.array([0.0, 0.0, -1.0])
                angle = np.arctan2((R @ np.cross(a, e)) @ toward, (R @ e) @ toward)
            Q = _rotation(angle * a)
            linear = getattr(model, "fitted_linear", None)
            linear = linear if linear is not None else model.fitted_scale * np.eye(3)
            centre = np.asarray(linear, np.float64) @ model.box[:3, 3]
            turn[:3, :3], turn[:3, 3] = Q, centre - Q @ centre
        sent[index] = pose @ turn
        return turn, None if k is None else model.box[:3, k].tolist()

    def recognition_lost(self, tracking_lost, bboxes):
        """Lost: its tracks are, or it has no box at all. An object whose mesh
        has not fitted yet is not lost: it has the box of its points, and its
        reply says it has no mesh pose."""
        return [bool(lost or bboxes[index] is None) for index, lost in enumerate(tracking_lost)]

    def note_last_fits(self, motions, lost):
        """Each mesh's fit as last seen on its part -- not lost, nor withheld
        off it -- with its pose in the camera then: what an init sent again
        keeps for the part (meta["keep"], carry_kept_fits())."""
        last = getattr(self, "last_fits", None)
        if last is None:
            last = self.last_fits = {}
        withheld = getattr(self, "model_withheld", None) or []
        bindings = getattr(self, "bindings", None) or []
        sent = getattr(self, "sent_model_poses", None) or []
        for index, model in enumerate(getattr(self, "fitted_models", None) or []):
            if (model is None or model.fitted_pose is None or index >= len(lost) or lost[index]
                    or (index < len(withheld) and withheld[index])):
                continue
            motion = np.asarray(motions[index], dtype=np.float64)
            if not np.isfinite(motion).all():
                continue
            last[index] = {"model": model, "pose": motion @ model.fitted_pose, "fit_id": model.fitted_id,
                           "binding": bindings[index] if index < len(bindings) else None,
                           "sent": sent[index] if index < len(sent) else None, "frame": self.frame_id - 1,
                           "session": getattr(self, "mesh_session", 0)}

    def carry_kept_fits(self, meta, models, metres, K):
        """The fits an init sent again keeps ({index: note_last_fits()'s}),
        and the region each is seen in on this view ({index: region}).

        An init sent again for a part lost (pose_client's relocate()) fitted
        every part afresh, those still tracked too: PP-S3-2's sample, tracked
        and untouched, came out of it at another size, not counted as
        measured -- an end hidden under the arm -- and turned end for end by
        the new view's silhouettes, and its ways round were asked of the
        person again (batch 20261113_bcd_clean). A part the client sends in
        meta["keep"] keeps its fit as last seen: the same fit, size, ways
        round and the person's answers (its Binding), where it is the same
        mesh's and its fit has not changed since; clicked where the camera
        sees it so, where the client sent no click. fit_boxes() then fits it
        afresh only where this view says it no longer lies so
        (carried_fit_rejection())."""
        last = getattr(self, "last_fits", None) or {}
        carried, regions = {}, {}
        for index in meta.get("keep") or []:
            try:
                index = int(index)
            except (TypeError, ValueError):
                continue
            fit = last.get(index)
            if fit is not None and fit["session"] != getattr(self, "mesh_session", 0):
                fit = None                       # an earlier session's, not the one tracked since
            if not 0 <= index < len(models) or models[index] is None or fit is None:
                self.say(f"object {index}: no fit of it seen to keep; fitted afresh")
                continue
            model = fit["model"]
            if (model.model_id != models[index].model_id or model.fitted_id != fit["fit_id"]
                    or len(model.points) != len(models[index].points)):
                self.say(f"object {index}: its fit has changed since it was last seen; fitted afresh")
                continue
            pose = fit["pose"]
            click = mesh_click(model, pose[:3, :3], pose[:3, 3], fitted_axis_scale(model), metres, K)
            if click is None:
                self.say(f"object {index}: its fit as last seen (frame {fit['frame']}) is not seen on this view; "
                         "fitted afresh")
                continue
            if not len(meta["points"][index]):
                meta["points"][index], meta["labels"][index] = click[0], [1] * len(click[0])
            carried[index], regions[index] = fit, click[1]
        return carried, regions

    def handle_init(self, meta, parts):
        started = time.time()
        self.init_report = []
        # The init's SAM2 previews by their inputs (preview_masks()).
        self.init_previews = {}
        try:
            return self.init_session(meta, parts, started)
        finally:
            self.init_previews = None

    def preview_masks(self, rgb, points, labels):
        """The pipeline's preview_user_masks(), made once in an init for each
        set of inputs: separating the parts, finding obstacles and fitting
        the boxes ask for the same (~0.55 s each, 2026-11-12). A preview
        starts the segmenter afresh (load_first_frame()), so it is the same
        whenever it is made."""
        previews = getattr(self, "init_previews", None)
        if previews is None:
            return self.pipeline.preview_user_masks(rgb, points, labels)
        key = _digest(rgb, points, labels)
        if key not in previews:
            previews[key] = self.pipeline.preview_user_masks(rgb, points, labels)
        return previews[key]

    def init_session(self, meta, parts, started):
        self.build_pipeline()
        rgb = decode_rgb(parts[0])
        depth = decode_depth(parts[1], meta).astype(np.float32)
        K = np.asarray(meta["K"], dtype=np.float64).reshape(3, 3)
        depth_factor = float(meta["depth_factor"])
        models = decode_models(meta, parts[2:], len(meta["points"]))
        self.load_archive(models)
        # Each mesh's prior from its own fits, where the client said none (FIT_TOL_SIZE).
        for index, model in enumerate(models):
            errors = (getattr(self, "learnt_errors", None) or {}).get(getattr(model, "model_id", None))
            if model is not None and errors and not getattr(model, "tol_declared", False):
                learnt = float(np.median(errors))
                if learnt > MESH_ERROR_P95:
                    # Said, not fitted at: see the comment after FIT_TOL_SIZE.
                    self.say(f"object {index}: its mesh is {learnt:.1%} off its part at the 95th percentile "
                             f"on {len(errors)} fit(s) before, rougher than the {MESH_ERROR_P95:.2%} it is "
                             f"fitted at: expect coarse fits and refusals; a truer mesh, or its error sent "
                             f"as mesh_error, is what helps")
        # Each mesh's proportion bound from its own fits, where the client said none.
        for index, model in enumerate(models):
            learnt = (getattr(self, "learnt_stretch", None) or {}).get(getattr(model, "model_id", None))
            if False and model is not None and learnt:     # fixed bound (MODEL_STRETCH_MAX, 2026-11-19)
                model.set_stretch_max(learnt)
                self.say(f"object {index}: its mesh's proportions allowed {learnt:.2f} off, as its fits before showed")
        # The jump guard's speeds where the client gives them (pipeline_test2.yaml).
        frontend = getattr(self.pipeline, "frontend", None)
        if frontend is not None:
            if meta.get("max_speed_mps") is not None:
                frontend.pose_jump_guard_max_speed = float(meta["max_speed_mps"])
            if meta.get("max_turn_dps") is not None:
                frontend.pose_jump_guard_max_rot_speed_deg = float(meta["max_turn_dps"])
        robot = decode_robot_mask(meta, depth.shape)
        self.same_scale = bool(meta.get("same_scale_meshes"))    # see LOCATE_SCALE_AGREE
        self.init_proposals = None
        self.kept_fits = set()
        carried, kept_regions = self.carry_kept_fits(meta, models, depth / depth_factor, K)
        meta["points"], meta["labels"], located = self.locate_unclicked(
            rgb, depth / depth_factor, K, meta["points"], meta["labels"], models, robot)
        for index, region in kept_regions.items():
            located.setdefault(index, region)
        self.init_points = meta["points"]
        meta["points"], meta["labels"] = self.separate_parts(rgb, meta["points"], meta["labels"])
        # Followed apart from the parts, by the depth (OBSTACLE_FOLLOW_PX).
        self.obstacles = self.discover_obstacles(rgb, depth / depth_factor, K, meta["points"], meta["labels"],
                                                 located, robot, int(meta.get("discover") or 0))
        self.init_proposals = None
        self.anchors = np.stack([
            anchor_from_points(points, labels, depth, K, depth_factor)
            for points, labels in zip(meta["points"], meta["labels"])
        ])
        specs = meta.get("models") or []
        required = [index < len(specs) and specs[index] is not None
                    for index in range(len(meta["points"]))]
        frame = Frame(
            id=self.frame_id,
            rgb=rgb,
            depth=depth,
            intrinsics=K,
            depth_factor=depth_factor,
            timestamp=meta.get("timestamp", time.time()),
        )
        stepped = []

        def first_step():
            # This session's fits from here on (note_last_fits()); those before
            # are kept until now, for an init asking for clicks sent again.
            self.last_fits = {}
            self.pipeline.add_user_points(meta["points"], meta["labels"])
            stepped.append(np.asarray(self.pipeline.step(frame), dtype=np.float64))

        # The first frame tracked while the meshes are fitted (INIT_STEP_WITH_FITS).
        self.boxes = self.fit_boxes(rgb, depth / depth_factor, K, meta["points"], meta["labels"],
                                    models, required, located, robot, carried,
                                    meanwhile=first_step if INIT_STEP_WITH_FITS else None)
        # A clicked part whose points took in what is not it (points_stray())
        # is clicked again -- by the Agent, else the person (NeedClick) --
        # rather than tracked on them: its size and pose come out of what is
        # not it. Once only: asked in the init before this one, it is tracked
        # as before (its box, the mesh retried on fresh frames).
        asked = getattr(self, "strays_asked", set())
        strays = {index: why for index, why in self.init_strays.items() if index not in asked}
        self.strays_asked = set(strays)
        if strays:
            # Where to click it again, offered: where the mesh came to on its
            # points, the camera seeing it there at its depth (mesh_click()) --
            # what touches the part shows nearer, and is left out. Without it
            # the Agent's VLM was shown only the unnamed boxes in view, the
            # part not among them, and every click fell to the person
            # (RR-S1-2, 2026-11-23).
            offered = {}
            for index in strays:
                pose = getattr(strays[index], "pose", None)
                try:
                    found = (mesh_click(models[index], *pose, depth / depth_factor, K)
                             if pose is not None and models[index] is not None else None)
                except Exception:
                    traceback.print_exc()
                    found = None
                offered[index] = [] if found is None else [np.asarray(found[0]).tolist()]
            for index in sorted(strays):
                self.say(f"object {index}: its click's segment takes in what is not it "
                         f"({strays[index].split('; ')[-1]}); asked to be clicked again"
                         + (f", where its mesh came to offered at {offered[index][0]}" if offered[index] else ""))
            raise NeedClick([{"object": index, "why": "its segment took in what is not it: click it again, "
                                                      "away from whatever touches it", "candidates": offered[index]}
                             for index in sorted(strays)],
                            [[] if index in strays else np.asarray(p).tolist()
                             for index, p in enumerate(self.init_points)])
        if not stepped:
            first_step()
        motions = stepped[0]
        self.surfaces = [SurfaceMemory() for _ in meta["points"]]
        regions = self.carried_regions(motions, frame)
        self.record_surfaces(motions, frame, regions)
        # Where each fit was seen on the part: the init's frame (MODEL_CHECK_FAILS).
        self.object_state(len(motions))
        for index, model in enumerate(self.fitted_models or []):
            if model is not None and model.fitted_pose is not None:
                self.still_motions[index] = np.asarray(motions[index], dtype=np.float64).copy()
        # The init view's own evidence on the ways round it left in doubt.
        self.update_bindings(motions, frame, regions, robot, rank=True)
        self.smoother.reset(motions)
        self.frame_id += 1
        self.say(f"init done in {time.time() - started:.1f} s")
        return motions @ self.anchors, frame, motions

    def say(self, text):
        """A line of the server's log, kept for the init's reply while an init
        runs (init_report): what the client sees of how its objects were
        located and fitted, and how long each took."""
        print(f"[server] {text}", flush=True)
        report = getattr(self, "init_report", None)
        if report is not None:
            report.append(text)

    def proposals(self, rgb, metres, K=None):
        """(proposals, sources, (SAM2's, those taken as their depth region, the
        depth's)) of the init frame: SAM2's masks, each with LOCATE_DEPTH_SAME_IOU
        with a depth region that region instead, then the depth's other regions
        (depth_proposals()); (None, None, None) if the segmenter offers none.
        Made once an init: locating and discovering obstacles share them."""
        made = getattr(self, "init_proposals", None)
        if made is not None:
            return made
        sam = self.pipeline.propose_masks(rgb)
        if sam is None:
            return None, None, None
        sam = list(sam)
        regions = depth_proposals(metres, K)
        proposals, sources, as_region = [], [], 0
        for which, mask in enumerate(sam):
            same = [k for k, region in enumerate(regions)
                    if (mask & region).sum() > LOCATE_DEPTH_SAME_IOU * (mask | region).sum()]
            if same:
                proposals.append(regions.pop(same[0]))
                sources.append(f"SAM2 proposal {which} as its depth region")
                as_region += 1
            else:
                proposals.append(mask)
                sources.append(f"SAM2 proposal {which}")
        proposals += regions
        sources += [f"depth region {k}" for k in range(len(regions))]
        self.init_proposals = (proposals, sources, (len(sam), as_region, len(regions)))
        return self.init_proposals

    def locate_unclicked(self, rgb, metres, K, points, labels, models, robot=None):
        """Each object's click, made from its mesh where none was sent, and
        the region each was located in ({index: region}).

        An object sent with an empty point list and a mesh is located in the
        view by the mesh (locate_with_mesh()); clicked objects are kept as
        they are. No object is located on another's -- a clicked object's
        segment, an object located before -- nor on `robot`, the robot's
        pixels as the client sees them. One not located is tried again once
        others are, without their pixels: RR-S1-2's ring, the bar through its
        hole in its segment, fitted none, and was located once the bar was.
        """
        self.locate_scales, self.locate_shapes = {}, {}
        if all(len(p) for p in points):
            return points, labels, {}
        started = time.time()
        proposals, sources, (sam, as_region, regions) = self.proposals(rgb, metres, K)
        # The fits made, by their inputs: each round looks at every proposal again.
        fits_made = types.SimpleNamespace(done={}, reused=0)
        if proposals is None:
            raise RuntimeError("the segmenter offers no mask proposals; click every object")

        clicked = [p for pts, lbls in zip(points, labels)
                   for p, label in zip(pts, lbls) if label]
        taken = np.zeros(metres.shape, bool)
        # A clicked object's segment is that object's: the obstacle box's
        # round latch, in it, took the hidden ring's place in RI-S4-2.
        by_click = [index for index, pts in enumerate(points) if len(pts)]
        if by_click:
            try:
                logits = self.pipeline.preview_user_masks(rgb, [points[i] for i in by_click],
                                                          [labels[i] for i in by_click])
                taken |= np.any((as_numpy(logits) > 0).reshape(len(by_click), *metres.shape), axis=0)
            except Exception as exc:
                self.say(f"no segments of the clicked objects ({type(exc).__name__}: {exc}); "
                         "proposals on them are fitted too")
        points, labels = list(points), list(labels)
        self.say(f"{sam} SAM2 proposals ({as_region} taken as their depth region), "
                 f"{regions} more from depth; clicked objects on {int(taken.sum())} px; "
                 + (f"robot on {int(robot.sum())} px" if robot is not None else "robot not sent"))
        pending = [index for index, pts in enumerate(points) if not len(pts)]
        for index in pending:
            if models[index] is None:
                raise ValueError(f"object {index}: no click and no usable mesh to locate it by")
        clicked_taken = taken.copy()
        grown = taken.copy()             # the located parts in it grown (LOCATE_REGION_GROW_PX)
        located, notes, fitted, round_, rest = {}, {}, {}, 0, False
        fitted_rivals = {}             # each located part's fit, with the other places it fits (fit[3])
        shapes = {}               # each located part's scale along its box axes
        while pending:
            round_ += 1
            progress = False
            for index in list(pending):
                why = []
                click, region, note, fit = locate_with_mesh(models[index], proposals, metres, K,
                                                            clicked, taken, robot, why, rest, grown=grown,
                                                            image=rgb, kept=fits_made)
                for which, said in sorted(why):
                    self.say(f"  object {index}{f' (round {round_})' if round_ > 1 else ''}"
                             f"{', what is left of proposals' if rest else ''}, {sources[which]} "
                             f"({describe_proposal(proposals[which])}): {said}")
                notes[index] = note
                if click is None:
                    continue
                taken |= region
                grown |= _grown(region)
                located[index] = region
                fitted[index] = fit[:2] + (rest,)
                shapes[index] = fit[2]
                fitted_rivals[index] = fit
                pending.remove(index)
                progress = True
                points[index], labels[index] = click, [1] * len(click)
                self.say(f"object {index}: located by its mesh at {click} ({note})")
            if not progress:
                if rest or not any((proposal & taken).any() for proposal in proposals):
                    break
                # Last, what is left of the proposals other objects took.
                rest = True
        # Not located: the person is asked (NeedClick), once the rest are.
        unsure = {index: (f"not located by its mesh ({notes[index]})", []) for index in pending}
        rivals = {index: fit[3] for index, fit in fitted_rivals.items()}
        # The one whose scale is furthest off every other's (LOCATE_SCALE_AGREE),
        # the lower scored of two as far off, is located again at theirs. One
        # kept at its own, none fitting at theirs, is the one in doubt: it sets
        # no scale for the others. RR-S1-1's ring, located on the hand at 1.55,
        # was kept so; then the bar, fitted right at 0.59, was located again at
        # the ring's, on the hand at the image's edge (2026-10-31).
        again, kept = set(), set()
        while self.same_scale:
            off = []
            for index, (score, scale, _) in fitted.items():
                others = [fitted[j][1] for j in fitted if j != index and j not in kept]
                if index in again or not others:
                    continue
                # The nearest other part's: a scale is off only where no other
                # part's agrees. HS-S1-1's four meshes are of two reconstructions,
                # ring and bar at 0.64-0.85, tester and sample at 2.2-2.4: off the
                # median of the others, every one was (2026-11-17).
                at = float(min(others, key=lambda o: abs(np.log(scale / o))))
                ratio = abs(np.log(scale / at))
                if ratio > np.log(models[index].stretch_max):   # LOCATE_SCALE_AGREE, the mesh's own
                    off.append((round(ratio, 6), -score, index, at))
            if not off:
                break
            _, _, index, at = max(off)
            again.add(index)
            why = []
            others = clicked_taken | np.any([located[j] for j in located if j != index], axis=0)
            others_grown = clicked_taken | np.any([_grown(located[j]) for j in located if j != index], axis=0)
            # What is left of the others' proposals is proposed too: RR-S1-1's
            # ring, with the bar in one proposal, was first located on the hand
            # holding the bar, and is what is left of it once the bar is located.
            click, region, note, fit = locate_with_mesh(
                models[index], proposals, metres, K, clicked, others, robot, why, True,
                (at / models[index].stretch_max, at * models[index].stretch_max), grown=others_grown, image=rgb,
                kept=fits_made)
            for which, said in sorted(why):
                self.say(f"  object {index}, again at the others' scale {at:.2f}, {sources[which]} "
                         f"({describe_proposal(proposals[which])}): {said}")
            if click is None:
                self.say(f"object {index}: kept at scale {fitted[index][1]:.2f}, off the others' "
                         f"{at:.2f}; none fits at theirs ({note})")
                kept.add(index)
                continue
            located[index] = region
            fitted[index] = fit[:2] + (fitted[index][2],)
            shapes[index] = fit[2]
            rivals[index] = fit[3]
            points[index], labels[index] = click, [1] * len(click)
            self.say(f"object {index}: located again by its mesh at {click}, at the others' scale "
                     f"({note})")
        # One located weakly is looked for again once the others are, what is
        # left of their proposals proposed too, at their scale if they share
        # one; where it fits
        # better by LOCATE_BETTER, it is there: RR-S1-1's ring, located first on
        # the hand holding the bar (0.68) -- the bar and the ring on it one
        # proposal -- is what is left of that once the bar is located.
        for index in sorted(fitted, key=lambda i: fitted[i][0]):
            score = fitted[index][0]
            others = [fitted[j][1] for j in fitted if j != index and j not in kept]
            if score >= LOCATE_WEAK or not others:
                continue
            at = float(np.exp(np.median(np.log(others))))
            why = []
            taken_now = clicked_taken | np.any([located[j] for j in located if j != index], axis=0)
            grown_now = clicked_taken | np.any([_grown(located[j]) for j in located if j != index], axis=0)
            click, region, note, fit = locate_with_mesh(
                models[index], proposals, metres, K, clicked, taken_now, robot, why, True,
                (at / models[index].stretch_max, at * models[index].stretch_max) if self.same_scale else None,
                grown=grown_now,
                image=rgb, kept=fits_made)
            if click is None or fit[0] < score + LOCATE_BETTER:
                continue
            for which, said in sorted(why):
                self.say(f"  object {index}, looked for again ({score:.2f}), {sources[which]} "
                         f"({describe_proposal(proposals[which])}): {said}")
            located[index] = region
            fitted[index] = fit[:2] + (True,)
            shapes[index] = fit[2]
            rivals[index] = fit[3]
            kept.discard(index)             # at the others' scale now
            points[index], labels[index] = click, [1] * len(click)
            self.say(f"object {index}: located again by its mesh at {click}, fitting better than at "
                     f"{score:.2f} ({note})")
        self.locate_scales = {index: fit[1] for index, fit in fitted.items()}
        self.locate_shapes = shapes
        self.say(f"located by mesh in {time.time() - started:.1f} s "
                 f"({len(fits_made.done)} fits made, {fits_made.reused} reused)")
        # Located, but not for sure: weakly, or as well somewhere else.
        for index, (score, scale, _) in fitted.items():
            here = [points[index]]
            if score < LOCATE_ASK_BELOW:
                unsure[index] = (f"fits its place only weakly ({score:.2f})", here)
            elif rivals.get(index):
                said = ", ".join(f"{other:.2f}" for _, other in rivals[index])
                unsure[index] = (f"fits {len(rivals[index]) + 1} places about as well ({score:.2f}; {said})",
                                 here + [click for click, _ in rivals[index]])
            elif index in kept:
                unsure[index] = (f"located at scale {scale:.2f}, off the other parts'", here)
        if unsure:
            for index, (why, _) in sorted(unsure.items()):
                self.say(f"object {index}: unsure -- {why}; the person is asked to click it")
            raise NeedClick([{"object": index, "why": why, "candidates": [np.asarray(c).tolist() for c in found]}
                             for index, (why, found) in sorted(unsure.items())],
                            [[] if index in unsure else np.asarray(p).tolist() for index, p in enumerate(points)])
        return points, labels, located

    def separate_parts(self, rgb, points, labels):
        """Each part's points and labels, the clicks of the other parts its
        segment holds added as negative points; see LOCATE_REGION_GROW_PX."""
        if len(points) < 2:
            return points, labels
        try:
            h, w = rgb.shape[:2]
            masks = (as_numpy(self.preview_masks(rgb, points, labels)) > 0).reshape(len(points), h, w)
        except Exception as exc:
            self.say(f"parts not checked for one segment holding two ({type(exc).__name__}: {exc})")
            return points, labels
        out_points, out_labels = [], []
        for index in range(len(points)):
            held = [click for other in range(len(points)) if other != index
                    for click, label in zip(points[other], labels[other])
                    if label and masks[index][min(max(int(round(click[1])), 0), h - 1),
                                              min(max(int(round(click[0])), 0), w - 1)]]
            out_points.append(list(points[index]) + [list(click) for click in held])
            out_labels.append(list(labels[index]) + [0] * len(held))
            if held:
                self.say(f"object {index}: its segment holds {len(held)} click(s) of other parts, "
                         "given it as negative points")
        return out_points, out_labels

    def discover_obstacles(self, rgb, metres, K, points, labels, located, robot, count):
        """Up to `count` obstacles in the init frame, each {"region": its pixels,
        "cuts": the planes (normal, point) cut from it, "box": (T_cam_box,
        extent), "missed": 0}, as follow_obstacles() follows them; see
        OBSTACLE_MIN_PX. `points` and `labels`, the parts' clicks, and
        `located`, {index: region} of those located by mesh, are the parts'
        pixels, which no obstacle takes."""
        if count <= 0:
            return []
        started = time.time()
        proposals, sources, _ = self.proposals(rgb, metres, K)
        if proposals is None:
            self.say("no obstacles looked for: the segmenter offers no mask proposals")
            return []
        cam, valid = backproject(metres, K)
        depth = cam[..., 2]
        taken = np.zeros(metres.shape, bool)
        for region in located.values():
            taken |= region
        try:
            logits = self.preview_masks(rgb, points, labels)
            taken |= np.any((as_numpy(logits) > 0).reshape(len(points), *metres.shape), axis=0)
        except Exception as exc:
            self.say(f"obstacles: no segments of the parts ({type(exc).__name__}: {exc}); "
                     "only their located regions are left out")
        if robot is not None:
            taken |= robot
        rng = np.random.default_rng(0)
        found, kept, left = [], np.zeros(metres.shape, bool), {}
        for which in sorted(range(len(proposals)), key=lambda k: -int(proposals[k].sum())):
            mask = proposals[which] & ~taken & valid
            if mask.sum() < image_area_px(OBSTACLE_MIN_PX, mask.shape):
                continue
            view = observe(mask, metres, K)
            if view is None:
                continue
            _, _, keep, plane, note = view
            if note.startswith("flat"):
                continue                          # lies flat on its support: no obstacle
            cuts = [plane] if note == "support surface removed" else []
            # A surface within it -- the bench a box stands on, depth-joined
            # to it -- is no obstacle, what stands on it may be: in front of
            # the plane its pixels hold most of, where that is a surface
            # (OBSTACLE_FLAT_M), alone.
            pts = cam[keep]
            tol = max(noise_tol(cam[..., 2], keep, (1,)) or 0.0, cam_pixel_tol(cam, keep))
            surface = _ring_plane(pts[rng.permutation(len(pts))[:LOCATE_BOX_POINTS]], rng, tol)
            if surface is not None:
                n, p = surface
                on = cam[keep & (np.abs((cam - p) @ n) < PLANE_TOLS * tol)]
                on = on[rng.permutation(len(on))[:LOCATE_BOX_POINTS]] - p
                if (len(on) >= MIN_BOX_POINTS
                        and np.ptp(on @ np.linalg.svd(on, full_matrices=False)[2][0]) >= OBSTACLE_FLAT_M):
                    keep = keep & ((cam - p) @ n < -PLANE_TOLS * tol)
                    cuts.append((n, p, tol))
            for piece in depth_pieces(keep, depth, image_area_px(OBSTACLE_MIN_PX, keep.shape), K):
                if (piece & kept).sum() > OBSTACLE_OVERLAP * piece.sum():
                    continue
                pts = cam[piece]
                sample = pts[rng.permutation(len(pts))[:LOCATE_BOX_POINTS]]
                R, centre, extent = min_obb(sample)
                size = " x ".join(f"{e * 1000:.0f}" for e in sorted(extent, reverse=True))
                said = f"{sources[which]} ({describe_proposal(piece)}), {size} mm"
                why = None
                if not OBSTACLE_EXTENT_M[0] <= extent.max() <= OBSTACLE_EXTENT_M[1]:
                    why = "not an obstacle's size"
                elif np.median(pts[:, 2]) > OBSTACLE_MAX_DEPTH_M:
                    why = "too far"
                elif plane is not None and -np.min((pts - plane[1]) @ plane[0]) < OBSTACLE_MIN_HEIGHT_M:
                    why = "no higher than its support"
                elif extent.max() >= OBSTACLE_FLAT_M:
                    mid = sample.mean(0)
                    normal = np.linalg.svd(sample - mid, full_matrices=False)[2][2]
                    if np.mean(np.abs((sample - mid) @ normal) < PLANE_TOLS * tol) >= OBSTACLE_FLAT_SHARE:
                        why = "a surface"
                if why is not None:
                    left[why] = left.get(why, 0) + 1
                    continue
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = R, centre
                found.append({"region": piece, "cuts": cuts, "box": (T, extent), "missed": 0})
                kept |= piece
                self.say(f"obstacle {len(found)}: {said}")
                if len(found) >= count:
                    break
            if len(found) >= count:
                break
        self.say(f"{len(found)} obstacle(s) found in {time.time() - started:.1f} s"
                 + (f"; left out: {', '.join(f'{n} {why}' for why, n in left.items())}" if left else ""))
        return found

    def follow_obstacles(self, frame, regions=None):
        """Each obstacle found at init, found again on this frame by the depth
        near where it was last; see OBSTACLE_FOLLOW_PX. `regions`, the fitted
        meshes' (carried_regions()), are the parts' too."""
        obstacles = getattr(self, "obstacles", None)
        if not obstacles:
            return
        try:
            h, w = frame.depth.shape
            metres = as_numpy(frame.depth) / frame.depth_factor
            K = np.asarray(frame.intrinsics, dtype=np.float64)
            masks = as_numpy(getattr(frame, "mask", None))
            if masks.ndim == 4:
                masks = masks[:, 0]
        except Exception:
            return
        taken = np.zeros((h, w), bool)
        if masks.ndim == 3 and masks.shape[1:] == (h, w):
            taken |= np.any(masks > 0, axis=0)
        for part in [*(regions or []), getattr(self, "track_robot", None)]:
            if part is not None and part.shape == (h, w):
                taken |= part
        cam, valid = backproject(metres, K)
        depth = cam[..., 2]
        follow = image_px(OBSTACLE_FOLLOW_PX, (h, w))
        grow = np.ones((2 * follow + 1,) * 2, np.uint8)
        # Each pixel is the nearest obstacle's to look at (OBSTACLE_SMOOTH_FULL_DEG).
        owner = None
        if len(obstacles) > 1:
            owner = np.argmin(np.stack([cv2.distanceTransform((~obstacle["region"]).astype(np.uint8), cv2.DIST_L2, 3)
                                        for obstacle in obstacles]), axis=0)
        for number, obstacle in enumerate(obstacles):
            try:
                last = obstacle["region"]
                v, u = np.nonzero(last)
                near = np.zeros((h, w), bool)
                near[max(v.min() - follow, 0):v.max() + follow + 1,
                     max(u.min() - follow, 0):u.max() + follow + 1] = True
                near &= valid & ~taken
                if owner is not None:
                    near &= owner == number
                for n, p, *cut_tol in obstacle["cuts"]:
                    near[near] = (cam[near] - p) @ n < -PLANE_TOLS * (cut_tol[0] if cut_tol else max(noise_tol(cam[..., 2], near, (1,)) or 0.0, cam_pixel_tol(cam, near)))
                was = cv2.dilate(last.astype(np.uint8), grow) > 0
                pieces = depth_pieces(near, depth, image_area_px(OBSTACLE_MIN_PX, (h, w)), K)
                # On where it was (intersection over union), else -- moved off
                # it -- within its grown region and as big within
                # OBSTACLE_FOLLOW_SIZE: the bench beside an arm is mostly in
                # that region too, three times its size.
                shares = [((piece & last).sum() / max(piece.sum(), last.sum()), k, piece)
                          for k, piece in enumerate(pieces)]
                if not any(share >= OBSTACLE_FOLLOW_SHARE for share, _, _ in shares):
                    shares = [((piece & was).sum() / piece.sum(), k, piece) for k, piece in enumerate(pieces)
                              if 1 / OBSTACLE_FOLLOW_SIZE <= piece.sum() / last.sum() <= OBSTACLE_FOLLOW_SIZE]
                found = None
                # The most on where it was, of an obstacle's size: joined in
                # the depth to the bench, an arm is not that.
                for share, _, piece in sorted(shares, key=lambda item: -item[0]):
                    if share < OBSTACLE_FOLLOW_SHARE:
                        break
                    pts = cam[piece]
                    R, centre, extent = min_obb(pts[::max(1, len(pts) // OBSTACLE_BOX_POINTS)])
                    if OBSTACLE_EXTENT_M[0] <= extent.max() <= OBSTACLE_EXTENT_M[1]:
                        found = piece
                        break
                if found is None:
                    obstacle["missed"] += 1
                    continue
                piece = found
                # The last box's axes, and so much of the turn (OBSTACLE_SMOOTH_FULL_DEG).
                before = obstacle["box"][0][:3, :3]
                P = max(_signed_permutations(), key=lambda P: np.trace(before.T @ R @ P))
                R, extent = R @ P, np.abs(P).T @ extent
                turn = cv2.Rodrigues(before.T @ R)[0].ravel()
                share = np.clip(np.degrees(np.linalg.norm(turn)) / OBSTACLE_SMOOTH_FULL_DEG,
                                OBSTACLE_SMOOTH_MIN_ALPHA, 1.0)
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = before @ _rotation(share * turn), centre
                obstacle.update(region=piece, box=(T, extent), missed=0)
                taken |= piece
            except Exception:
                traceback.print_exc()
                obstacle["missed"] += 1

    def obstacle_replies(self, frame):
        """The obstacles as the reply carries them: {"id", "lost", "bbox"},
        the box as box_in_camera() gives a part's."""
        K = np.asarray(frame.intrinsics, dtype=np.float64)
        out = []
        for number, obstacle in enumerate(getattr(self, "obstacles", None) or [], 1):
            T, extent = obstacle["box"]
            out.append({"id": number, "lost": obstacle["missed"] >= OBSTACLE_LOST_FRAMES,
                        "bbox": box_in_camera(np.eye(4), T, extent, K, frame.depth.shape)})
        return out

    def handle_track(self, meta, parts):
        if self.pipeline is None or self.anchors is None:
            raise RuntimeError("pipeline not initialized; send an 'init' request first")
        frame = Frame(
            id=self.frame_id,
            rgb=decode_rgb(parts[0]),
            depth=decode_depth(parts[1], meta).astype(np.float32),
            intrinsics=np.asarray(meta["K"], dtype=np.float64).reshape(3, 3),
            depth_factor=float(meta["depth_factor"]),
            timestamp=meta.get("timestamp", time.time()),
        )
        motions = np.asarray(self.pipeline.step(frame), dtype=np.float64)
        # The robot's pixels on this frame, where the client sends them: a
        # gripper on or by a part is in that part's tracked mask.
        self.track_robot = decode_robot_mask(meta, frame.depth.shape)
        regions = self.carried_regions(motions, frame)
        motions = self.check_carried_meshes(motions, frame, regions)
        # The robot's poses of parts it set down (MODEL_CHECK_FAILS).
        self.take_priors(meta, frame.id)
        motions = self.apply_priors(motions, frame, regions)
        # The person's ways round (BINDING_CONFIDENCE): {"object": index, "hypothesis": id,
        # "by": whose -- "person" where a client older than it does not say}.
        for bind in meta.get("bind") or []:
            try:
                self.pending_binds[int(bind["object"])] = (str(bind["hypothesis"]),
                                                           str(bind.get("by") or "person")[:32])
            except (KeyError, TypeError, ValueError, AttributeError):
                print(f"[server] unusable bind {bind!r}", flush=True)
        self.update_bindings(motions, frame, regions, self.track_robot,
                             observe=self.frame_id % BINDING_EVERY == 0)
        self.retry_pending_meshes(motions, frame, regions)
        # A refit taken on this frame has its binding, the old one's ways round
        # carried, before this frame's reply: else that reply has no doubts.
        self.update_bindings(motions, frame, regions, self.track_robot, observe=False)
        self.follow_obstacles(frame, regions)
        if self.frame_id % SURFACE_EVERY == 0:
            self.record_surfaces(motions, frame, regions)
        self.frame_id += 1
        boxes = self.boxes or []
        sizes = [float(np.linalg.norm(b[1])) if b is not None and np.all(np.isfinite(b[1])) else None
                 for b in boxes]
        motions = self.smoother(motions, sizes)
        return motions @ self.anchors, frame, motions

    def without_robot(self, mask):
        """`mask` less the robot's pixels on this frame (handle_track()), where
        the client sent them. Kept in, the fingers around a held part, or by
        one just let go of, are points off its mesh: the carried check
        withheld PP-S4-1's tester in the hand (p95 146 mm) and fitted it again
        there, coarse and wrong, and the tester just placed in PP-S1-1 was
        withheld as the gripper left (fitness 0.54), refitted 180 deg off."""
        robot = getattr(self, "track_robot", None)
        return mask if robot is None or robot.shape != mask.shape else mask & ~robot

    def warm_up(self):
        """Load the models and track one made-up frame, then release: what
        the first init of a server would otherwise pay on top (WARM_UP_SHAPE).
        On the main thread, the init's."""
        started = time.time()
        h, w = WARM_UP_SHAPE
        rgb = np.full((h, w, 3), 160, np.uint8)
        depth = np.full((h, w), 0.8, np.float32)
        top, left = h // 2 - h // 10, w // 2 - w // 8
        rgb[top:h - top, left:w - left] = (30, 60, 200)
        # A rounded block, not a flat face: the pipeline takes the hull of its points.
        rows, cols = np.mgrid[top:h - top, left:w - left]
        bulge = 1.0 - ((rows - h / 2) / (h / 2 - top)) ** 2 - ((cols - w / 2) / (w / 2 - left)) ** 2
        depth[top:h - top, left:w - left] = 0.76 - 0.04 * np.clip(bulge, 0.0, 1.0)
        K = np.array([[0.55 * w, 0.0, w / 2], [0.0, 0.55 * w, h / 2], [0.0, 0.0, 1.0]])
        points, labels = [[[w // 2, h // 2], [w // 2 - w // 16, h // 2], [w // 2 + w // 16, h // 2]]], [[1, 1, 1]]
        self.build_pipeline()
        try:
            self.pipeline.preview_user_masks(rgb, points, labels)
            self.pipeline.add_user_points(points, labels)
            self.pipeline.step(Frame(id=0, rgb=rgb, depth=depth, intrinsics=K, depth_factor=1.0, timestamp=0.0))
        finally:
            self.release_pipeline()
            self.frame_id = 0
        self.log_gpu_memory("after warm-up")
        print(f"[server] warmed up in {time.time() - started:.1f} s", flush=True)

    def handle_preview(self, meta, parts):
        if self.pipeline is None:
            self.build_pipeline()
        rgb = decode_rgb(parts[0])
        logits = self.pipeline.preview_user_masks(rgb, meta["points"], meta["labels"])
        mask = (logits.squeeze(1) > 0).cpu().numpy().astype(np.uint8)
        out = {"mask_shape": list(mask.shape), "mask_sum": mask.sum(-1).sum(-1).tolist()}
        if meta.get("dump_dir"):
            os.makedirs(meta["dump_dir"], exist_ok=True)
            cv2.imwrite(f"{meta['dump_dir']}/rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            overlay[mask[0] > 0] = (0, 0, 255)
            cv2.imwrite(f"{meta['dump_dir']}/mask_overlay.png", overlay)
            out["dumped"] = True
        return out

    def frame_reply(self, cmd, meta, parts, t0):
        """The reply to an init or a track request, its frame worked on."""
        self.track_thread = threading.get_ident()
        if cmd == "init":
            poses, frame, motions = self.handle_init(meta, parts)
            self.log_gpu_memory("after init")
            if torch.cuda.is_available():
                self.gpu_after_init, self.gpu_growth_said = torch.cuda.memory_allocated() / 2**20, 0
        else:
            poses, frame, motions = self.handle_track(meta, parts)
            self.trace_gpu_frame(frame.id)
        lost = [bool(getattr(o, "lost", False)) for o in self.pipeline.objects]
        bboxes = self.bboxes(motions, frame)
        lost = self.recognition_lost(lost, bboxes)
        # A part found in the hand where the robot holds it (HOLD_PRIOR_SOURCE)
        # is not lost on that frame, though the tracker's points on it are.
        seen_held = getattr(self, "hold_seen", None) or {}
        lost = [bool(gone and not (seen_held.get(index) == frame.id and bboxes[index] is not None))
                for index, gone in enumerate(lost)]
        self.note_last_fits(motions, lost)
        reply = {
            "ok": True,
            "frame_id": self.frame_id - 1,
            "poses": np.asarray(poses, dtype=np.float64).tolist(),
            "lost": lost,
            "bboxes": bboxes,
            "model_errors": self.model_errors,
            # Found with no click; see OBSTACLE_MIN_PX.
            "obstacles": self.obstacle_replies(frame),
            "latency_ms": round((time.time() - t0) * 1000, 1),
            # The clicks used, those located by mesh among them, and how.
            **({"points": self.init_points, "init_report": self.init_report,
                # The parts that kept their fits as last seen (carry_kept_fits()).
                "kept": sorted(getattr(self, "kept_fits", None) or ())}
               if cmd == "init" else {}),
        }
        self.last_reply = reply
        return reply

    def settle_track(self):
        """Wait for a frame still worked on (SERVER_REPLY_SHARE) before
        anything else touches the pipeline."""
        job = getattr(self, "track_job", None)
        self.track_job = None
        if job is None:
            return
        try:
            job["future"].result()
        except Exception:
            traceback.print_exc()

    def answer_track(self, meta, parts, t0):
        """A track request's reply: in SERVER_REPLY_SHARE of the client's
        meta["timeout_ms"], stale where its frame is not done by then or an
        earlier one still is; without it, when the frame is done."""
        allowance = meta.get("timeout_ms")
        if allowance is None:
            self.settle_track()
            return self.frame_reply("track", meta, parts, t0)
        allowance = float(allowance) / 1000.0
        job = getattr(self, "track_job", None)
        if job is not None and not job["future"].done():
            return self.stale_reply(job, allowance, t0)
        if job is not None:
            self.track_job = None
            try:
                job["future"].result()
            except Exception as exc:
                print(f"[server] frame {job['frame_id']}, answered stale, failed: {type(exc).__name__}: {exc}",
                      flush=True)
        if getattr(self, "track_executor", None) is None:
            self.track_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="track",
                                                     initializer=no_grad_thread)
        job = {"future": self.track_executor.submit(self.frame_reply, "track", meta, parts, t0),
               "started": time.time(), "frame_id": self.frame_id, "dumped": 0}
        self.track_job = job
        try:
            reply = job["future"].result(timeout=SERVER_REPLY_SHARE * allowance)
        except FutureTimeout:
            return self.stale_reply(job, allowance, t0)
        self.track_job = None
        return reply

    def stale_reply(self, job, allowance, t0):
        """The last frame's reply, marked stale (SERVER_REPLY_SHARE): "stale":
        {"busy_s": how long the frame worked on has taken, "working_on":
        its frame id, "why"}. Past the client's whole allowance, the stack
        the frame is held in is printed, once each allowance."""
        busy = time.time() - job["started"]
        if busy >= allowance * (job["dumped"] + 1):
            job["dumped"] += 1
            stack = sys._current_frames().get(getattr(self, "track_thread", None))
            print(f"[server] frame {job['frame_id']} worked on for {busy:.1f} s, past the client's {allowance:.0f} s; "
                  "where it is:\n" + ("".join(traceback.format_stack(stack)) if stack is not None else "(not found)"),
                  flush=True)
        reply = dict(getattr(self, "last_reply", None) or {"ok": False, "error": "no frame replied to yet"})
        reply["stale"] = {"busy_s": round(busy, 1), "working_on": job["frame_id"],
                          "why": "the server is still working on an earlier frame; this is the last frame's reply"}
        reply["latency_ms"] = round((time.time() - t0) * 1000, 1)
        return reply

    def run(self):
        while True:
            msg = self.sock.recv_multipart()
            meta = json.loads(msg[0])
            cmd = meta.get("cmd")
            t0 = time.time()
            try:
                if cmd == "ping":
                    self.sock.send_json({"ok": True, "pong": True})
                    continue
                if cmd == "track":
                    self.sock.send_json(self.answer_track(meta, msg[1:], t0))
                    continue
                # Nothing else touches the pipeline while a frame is worked on.
                self.settle_track()
                if cmd == "reset":
                    self.archive_session()
                    self.release_pipeline()
                    self.log_gpu_memory("after reset")
                    self.report_gpu_held("after reset")
                    self.sock.send_json({"ok": True})
                    continue
                if cmd == "preview":
                    self.sock.send_json({"ok": True, **self.handle_preview(meta, msg[1:])})
                    continue
                if cmd != "init":
                    raise ValueError(f"unknown cmd: {cmd}")
                self.sock.send_json(self.frame_reply("init", meta, msg[1:], t0))
            except NeedClick as exc:
                # Not a failure: which objects to click, and the clicks of the rest.
                self.sock.send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                     "need_click": exc.requests, "points": exc.points,
                                     "init_report": self.init_report})
            except Exception as exc:
                traceback.print_exc()
                self.sock.send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                     **({"init_report": self.init_report} if cmd == "init" else {})})
            finally:
                if cmd != "track":
                    self.init_report = None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/pipeline/pipeline_test2.yaml")
    ap.add_argument("-b", "--bind", default="tcp://0.0.0.0:5555")
    ap.add_argument("--archive", default=None,
                    help="directory to file each mesh's settled surface and shapes in, and read them from "
                         "(see BINDING_FEATURE_PX); none kept without it")
    ap.add_argument("--trace-gpu-memory", action="store_true",
                    help="keep where each live GPU block was allocated, and say it, with what holds it, when a "
                         f"release leaves more than {GPU_HELD_WARN_MIB} MiB allocated; the memory is logged every "
                         f"{GPU_TRACE_EVERY_FRAMES} track frames too (see GPU_HELD_WARN_MIB)")
    ap.add_argument("--fit-workers", type=int, default=FIT_WORKERS_DEFAULT,
                    help="processes the mesh pose search's starts are shared among (see FIT_WORKERS); "
                         f"0 or 1 fits in the server's own process (default {FIT_WORKERS_DEFAULT})")
    ap.add_argument("--no-warm-up", action="store_true",
                    help="skip loading the models and tracking a made-up frame at start (see WARM_UP_SHAPE); "
                         "the first init then does both")
    args = ap.parse_args()
    # Before the pipeline: the workers start from a fresh interpreter (spawned),
    # so nothing the server loads is copied into them either way.
    start_fit_pool(args.fit_workers)
    server = PoseServer(args.config, args.bind, archive=args.archive, trace_gpu=args.trace_gpu_memory)
    try:
        if not args.no_warm_up:
            try:
                server.warm_up()
            except Exception:
                # Not needed to serve: the first init then loads what it needs.
                traceback.print_exc()
                print("[server] warm-up failed; the first init loads the models", flush=True)
        server.run()
    finally:
        server.settle_track()
        server.archive_session()
        if server.mesh_fit_executor is not None:
            server.mesh_fit_executor.shutdown(wait=True, cancel_futures=True)
        stop_fit_pool()

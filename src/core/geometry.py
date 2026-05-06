import numpy as np


def get_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_axes(ids, landmarks, w, h):
    pts = np.array([get_point(landmarks, i, w, h) for i in ids])

    center = np.mean(pts, axis=0)

    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eig(cov)

    order = np.argsort(eigvals)[::-1]
    major_axis = eigvecs[:, order[0]]
    minor_axis = eigvecs[:, order[1]]

    proj_major = np.dot(pts - center, major_axis)
    proj_minor = np.dot(pts - center, minor_axis)

    long_axis = proj_major.max() - proj_major.min()
    short_axis = proj_minor.max() - proj_minor.min()

    return center, long_axis, short_axis


def compute_center(ids, landmarks, w, h):
    pts = [get_point(landmarks, i, w, h) for i in ids]
    return np.mean(pts, axis=0)
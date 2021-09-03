import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render_points(x, y, c='b', segs=None):
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(x, y, c=c, s=25)
    if segs is not None:
        for a, b in segs:
            ax.plot([x[a], x[b]], [y[a], y[b]], c='grey', alpha=0.5)
    ax.set_aspect('equal', 'box')
    fig.canvas.draw()
    im = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    im = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return im


def render_3d_skeleton_views(skeletons, title, labels=None,
                             colors=['b', 'r', 'g'], axlim=2.5,
                             figsize=(12, 6)):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=figsize)
    fig.suptitle(title)
    for i, s in enumerate(skeletons):
        label = labels[i] if labels is not None else None
        c = colors[i]
        size = 50
        ax1.scatter([x[0] for x in s], [x[2] for x in s], s=size, c=c, label=label)
        ax2.scatter([x[1] for x in s], [x[2] for x in s], s=size, c=c)
        for a, b in s.bones:
            p1 = s[a]
            p2 = s[b]
            ax1.plot([p1[0], p2[0]], [p1[2], p2[2]], c=c, alpha=0.5)
            ax2.plot([p1[1], p2[1]], [p1[2], p2[2]], c=c, alpha=0.5)
    ax1.set_xlim(-axlim, axlim)
    ax1.set_ylim(-axlim, axlim)
    ax1.set_aspect('equal', 'box')
    if labels is not None:
        ax1.legend()

    ax2.set_xlim(-axlim, axlim)
    ax2.set_ylim(-axlim, axlim)
    ax2.set_aspect('equal', 'box')

    ax1.set_title('front')
    ax2.set_title('side')
    fig.canvas.draw()
    im = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    im = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return im


def get_canonical_orientation(X, torso_forward_vec, spine_up_vec,
                              interp_start=45, interp_range=30):
    X_zm = X - np.mean(X, axis=0).flatten()
    _, _, V = np.linalg.svd(X_zm)
    torso_forward_vec = -V[2, :] if V[2, :].dot(torso_forward_vec) < 0 else V[2, :]
    spine_up_vec = -V[0, :] if V[0, :].dot(spine_up_vec) < 0 else V[0, :]

    torso_pitch = np.degrees(np.arcsin(torso_forward_vec[2]))
    if torso_pitch > interp_start:
        if torso_pitch < interp_start + interp_range:
            theta = (torso_pitch - interp_start) / interp_range
            return theta * -spine_up_vec + (1. - theta) * torso_forward_vec
        else:
            return -spine_up_vec
    elif torso_pitch < -interp_start:
        if torso_pitch > -interp_start - interp_range:
            theta = (-torso_pitch - interp_start) / interp_range
            return theta * spine_up_vec + (1. - theta) * torso_forward_vec
        else:
            return spine_up_vec
    else:
        return torso_forward_vec


def flip_skeleton_offsets(arr, idxs):
    flipped = arr[idxs, :].copy()
    assert flipped.shape == arr.shape
    flipped[:, 0] = -flipped[:, 0]
    return flipped

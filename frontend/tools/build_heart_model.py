"""Build public/models/heart.glb from BodyParts3D meshes.

Source data: BodyParts3D, (c) The Database Center for Life Science (DBCLS), licensed
CC BY-SA 2.1 Japan. The generated heart.glb is a derivative and carries the same licence.
Meshes were obtained from https://github.com/olivercase/body_parts_3d_api (meshes/ folder).

What this script does
  1. Loads the heart wall, atria, great vessels and every coronary-artery segment.
  2. Groups the arteries into the three model targets: LAD, LCX, RCA (plus the left main trunk).
  3. Re-orients from BodyParts3D patient coordinates to the app frame
     (+y superior, +z anterior, +x patient's left), centres and scales to ~3 units tall.
  4. Gives the ventricular wall smooth per-vertex territory weights (by distance to each
     coronary artery), so the viewer can tint the muscle each artery supplies.
  5. Exports one GLB whose node names the frontend looks for:
       LAD, LCX, RCA, LeftMain, Myocardium (vertex colours = LAD/LCX/RCA territory weights),
       LeftAtrium, RightAtrium, Aorta, PulmonaryTrunk, SVC, IVC

Usage:
  pip install trimesh scipy numpy
  python tools/build_heart_model.py <folder with BodyParts3D .obj files> public/models/heart.glb
"""

import os
import re
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def classify(name: str) -> str | None:
    n = name.lower()
    if 'coronary' in n or 'interventricular artery' in n:
        if 'left coronary' in n and 'interventricular' in n:
            return 'LAD'
        if 'anterior interventricular artery' in n:  # septal branches of the LAD
            return 'LAD'
        if 'circumflex' in n:
            return 'LCX'
        if 'trunk of left coronary' in n:
            return 'LeftMain'
        if 'right coronary' in n or 'right posterior interventricular' in n:
            return 'RCA'
    table = {
        'wall of ventricle': 'Myocardium',
        'wall of left atrium': 'LeftAtrium',
        'wall of right atrium': 'RightAtrium',
        'ascending aorta': 'Aorta',
        'arch of aorta': 'Aorta',
        'pulmonary trunk': 'PulmonaryTrunk',
        'superior vena cava': 'SVC',
        'inferior vena cava': 'IVC',
    }
    for k, v in table.items():
        if k in n:
            return v
    return None


def english_name(path: str) -> str:
    with open(path, errors='ignore') as fh:
        for line in fh:
            m = re.match(r'#\s*English name\s*:\s*(.+)', line)
            if m:
                return m.group(1).strip()
            if line.startswith('v '):
                break
    return os.path.basename(path)


def main(src: str, out: str) -> None:
    groups: dict[str, list[trimesh.Trimesh]] = {}
    seen: set[bytes] = set()
    for fn in sorted(os.listdir(src)):
        if not fn.endswith('.obj'):
            continue
        path = os.path.join(src, fn)
        group = classify(english_name(path))
        if not group:
            continue
        mesh = trimesh.load(path, force='mesh', process=True)
        key = np.round(mesh.vertices[:50], 3).tobytes()  # some segments appear twice in the set
        if key in seen:
            continue
        seen.add(key)
        groups.setdefault(group, []).append(mesh)

    merged = {g: trimesh.util.concatenate(ms) for g, ms in groups.items()}
    print({g: len(m.faces) for g, m in merged.items()})

    # The IVC runs far down into the abdomen; keep only the part near the heart.
    ivc = merged['IVC']
    cut = merged['Myocardium'].bounds[0][2] + 10
    keep = np.where(ivc.vertices[ivc.faces][:, :, 2].min(axis=1) > cut)[0]
    merged['IVC'] = ivc.submesh([keep], append=True)

    # --- Orientation. BodyParts3D: +x = patient's left, -y = anterior, +z = superior (mm).
    # App frame: +x = patient's left, +y = superior, +z = anterior.
    R = np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0, -1, 0]], dtype=float)
    allv = np.vstack([m.vertices for m in merged.values()]) @ R.T
    lo, hi = allv.min(0), allv.max(0)
    scale = 3.0 / (hi - lo)[1]                     # ~3 units tall, like the procedural heart
    offset = -(lo + hi) / 2 * scale + np.array([0, 0.15, 0])
    T = np.eye(4)
    T[:3, :3] = R * scale
    T[:3, 3] = offset
    for m in merged.values():
        m.apply_transform(T)
    # Anatomical sanity checks in the new frame.
    cen = {g: m.centroid for g, m in merged.items()}
    assert cen['LCX'][0] > cen['RightAtrium'][0], 'patient-left axis looks wrong'
    assert cen['LAD'][2] > cen['LCX'][2], 'anterior axis looks wrong'
    assert cen['Aorta'][1] > cen['Myocardium'][1], 'superior axis looks wrong'

    # --- Territories: every ventricular-wall vertex gets a soft weight for LAD, LCX and RCA from its
    # distance to each artery (nearest artery dominates, with a narrow blend where two meet), then a
    # few rounds of neighbour averaging. Stored in the vertex colours: R=LAD, G=LCX, B=RCA.
    myo = merged['Myocardium']
    dists = []
    for lab in ('LAD', 'LCX', 'RCA'):
        d, _ = cKDTree(merged[lab].vertices).query(myo.vertices)
        dists.append(d)
    D = np.stack(dists, axis=1)
    tau = 0.07                                         # ~3 mm at the model's scale
    W = np.exp(-(D - D.min(axis=1, keepdims=True)) / tau)
    W /= W.sum(axis=1, keepdims=True)
    nbrs = myo.vertex_neighbors
    for _ in range(4):                                 # gentle Laplacian smoothing of the weights
        W = 0.5 * W + 0.5 * np.array([W[n].mean(axis=0) if len(n) else W[i] for i, n in enumerate(nbrs)])
        W /= W.sum(axis=1, keepdims=True)
    rgba = np.concatenate([np.round(W * 255), np.full((len(W), 1), 255)], axis=1).astype(np.uint8)
    myo.visual = trimesh.visual.ColorVisuals(myo, vertex_colors=rgba)
    share = W.argmax(axis=1)
    print('territory share (vertices):', {lab: int((share == i).sum()) for i, lab in enumerate(('LAD', 'LCX', 'RCA'))})

    scene = trimesh.Scene()
    for name, mesh in merged.items():
        if name != 'Myocardium':
            mesh.visual = trimesh.visual.ColorVisuals(mesh)  # drop any source materials
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    scene.export(out)
    print('wrote', out, round(os.path.getsize(out) / 1e6, 2), 'MB')
    lo, hi = scene.bounds
    print('bounds', np.round(lo, 2), np.round(hi, 2))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])

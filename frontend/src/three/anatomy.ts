// Procedural heart anatomy. Everything is defined in one parametric space (u, phi) so the
// coronary tubes always sit on the myocardium surface and the vessel→territory mapping
// stays consistent with the model outputs.
//
// Frame (before the scene tilts the heart): +y = base, -y = apex, +z = anterior,
// +x = patient's LEFT (viewer's right in an anterior view).
// u ∈ [-1, 1] runs apex → base; phi is the angle around the long axis, 0 = anterior,
// +phi towards the patient's left, ±π = posterior.
//
// To swap in a real mesh (.glb from BodyParts3D / NIH 3D), replace <Myocardium/> in
// HeartScene and keep these vessel curves, or name mesh nodes LAD/LCX/RCA and colour those.

import { BufferAttribute, CatmullRomCurve3, Color, SphereGeometry, Vector3 } from 'three';
import type { Vessel } from '../types';

export const SHAPE = { rx: 1.0, ry: 1.3, rz: 0.88, cy: -0.15 };

/** Horizontal taper: narrower towards the apex, slightly narrower at the base. */
const taper = (u: number) => (u < 0 ? 1 - 0.22 * u * u : 1 - 0.05 * u);
/** Flatten the base so the atria/great vessels sit on top. */
const yOf = (u: number) => SHAPE.cy + (u > 0 ? SHAPE.ry * 0.7 * u : SHAPE.ry * u);

export function surfacePoint(u: number, phi: number, lift = 0): Vector3 {
  const r = Math.sqrt(Math.max(0, 1 - u * u)) * taper(u);
  const p = new Vector3(SHAPE.rx * r * Math.sin(phi), yOf(u), SHAPE.rz * r * Math.cos(phi));
  if (lift) {
    const dir = p.clone().sub(new Vector3(0, SHAPE.cy, 0)).normalize();
    p.addScaledVector(dir, lift);
  }
  return p;
}

/** Which vessel typically supplies a point on the LV/RV surface (right-dominant pattern). */
export function territoryOf(u: number, phi: number): Vessel | null {
  if (u > 0.8) return null; // AV groove / base
  if (u < -0.72) return 'LAD'; // apex
  if (phi > -0.55 && phi <= 1.15) return 'LAD';
  if (phi > 1.15 && phi <= 2.6) return 'LCX';
  return 'RCA';
}

export interface MyocardiumGeometry {
  geometry: SphereGeometry;
  territory: (Vessel | null)[];
}

export function buildMyocardium(wSeg = 128, hSeg = 96): MyocardiumGeometry {
  const geometry = new SphereGeometry(1, wSeg, hSeg);
  const pos = geometry.attributes.position as BufferAttribute;
  const territory: (Vessel | null)[] = [];
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const u = Math.max(-1, Math.min(1, y));
    const phi = Math.atan2(x, z);
    const p = surfacePoint(u, phi);
    // A shallow interventricular groove along the LAD/PDA paths for a more anatomical silhouette.
    const groove = 0.035 * (Math.exp(-((phi - 0.35) ** 2) / 0.01) + Math.exp(-((Math.abs(phi) - 3.0) ** 2) / 0.012)) * (u < 0.75 ? 1 : 0);
    if (groove) p.addScaledVector(p.clone().sub(new Vector3(0, SHAPE.cy, 0)).normalize(), -groove);
    pos.setXYZ(i, p.x, p.y, p.z);
    territory.push(territoryOf(u, phi));
  }
  geometry.setAttribute('color', new BufferAttribute(new Float32Array(pos.count * 3), 3));
  geometry.computeVertexNormals();
  return { geometry, territory };
}

// Anatomical-atlas palette: cardiac muscle red-brown, arteries red, and the pulmonary trunk and
// venae cavae blue (oxygen-poor blood, the standard textbook convention). Coronary arteries are
// the one exception: they show the predicted stenosis risk colour.
export const TISSUE = new Color('#9a3b32');
export const PALETTE = {
  tissue: '#9a3b32',
  atrium: '#843029',
  dimTissue: '#2e100d',
  aorta: '#c2453a',
  pulmonary: '#4a5ca8',
  vein: '#3d4c96',
  rim: '#ff8a6a',
} as const;

// Control points in (u, phi). Each vessel = main trunk + side branches.
type Path = [u: number, phi: number][];
const LABEL_AT: Record<Vessel, [number, number]> = { LAD: [0.05, 0.34], LCX: [0.79, 1.85], RCA: [0.74, -1.45] };

const PATHS: Record<Vessel, { trunk: Path; branches: Path[] }> = {
  LAD: {
    trunk: [[0.9, 0.95], [0.82, 0.72], [0.62, 0.45], [0.32, 0.36], [0.0, 0.34], [-0.35, 0.36], [-0.65, 0.45], [-0.86, 0.75], [-0.96, 1.4]],
    branches: [
      [[0.4, 0.37], [0.15, 0.7], [-0.15, 0.98], [-0.4, 1.12]], // first diagonal
      [[-0.1, 0.34], [-0.35, 0.65], [-0.6, 0.85]], // second diagonal
    ],
  },
  LCX: {
    trunk: [[0.9, 0.95], [0.85, 1.25], [0.81, 1.6], [0.77, 2.0], [0.7, 2.35], [0.58, 2.6], [0.35, 2.72]],
    branches: [
      [[0.8, 1.55], [0.45, 1.7], [0.05, 1.82], [-0.35, 1.9]], // obtuse marginal
    ],
  },
  RCA: {
    trunk: [[0.86, -0.3], [0.8, -0.75], [0.76, -1.25], [0.73, -1.8], [0.68, -2.3], [0.58, -2.75], [0.42, -3.0], [0.1, -3.05], [-0.25, -3.02], [-0.55, -2.9]],
    branches: [
      [[0.74, -1.5], [0.35, -1.6], [-0.05, -1.62], [-0.4, -1.5]], // acute marginal
    ],
  },
};

const LEFT_MAIN: Vector3[] = [new Vector3(0.12, SHAPE.cy + 1.05, 0.38), new Vector3(0.38, SHAPE.cy + 0.98, 0.42)];
const RCA_OSTIUM = new Vector3(-0.12, SHAPE.cy + 1.03, 0.42);

/** Resample a (u,phi) path densely in parameter space, then map onto the surface. */
function pathToCurve(path: Path, lift: number, prefix: Vector3[] = []): CatmullRomCurve3 {
  const param = new CatmullRomCurve3(path.map(([u, phi]) => new Vector3(u, phi, 0)));
  const pts = param.getSpacedPoints(64).map((q) => surfacePoint(q.x, q.y, lift));
  return new CatmullRomCurve3([...prefix, ...pts]);
}

export interface VesselCurves {
  trunk: CatmullRomCurve3;
  branches: CatmullRomCurve3[];
  labelAnchor: Vector3;
}

export function buildVesselCurves(lift = 0.035): Record<Vessel, VesselCurves> & { leftMain: CatmullRomCurve3 } {
  const make = (v: Vessel, prefix: Vector3[] = []): VesselCurves => {
    const trunk = pathToCurve(PATHS[v].trunk, lift, prefix);
    const branches = PATHS[v].branches.map((b) => pathToCurve(b, lift));
    const [u, phi] = LABEL_AT[v];
    return { trunk, branches, labelAnchor: surfacePoint(u, phi, 0.28) };
  };
  return {
    LAD: make('LAD', [LEFT_MAIN[1]]),
    LCX: make('LCX', [LEFT_MAIN[1]]),
    RCA: make('RCA', [RCA_OSTIUM]),
    leftMain: new CatmullRomCurve3(LEFT_MAIN),
  };
}

// Great vessels & atria (decorative, not selectable).
export const AORTA = new CatmullRomCurve3([
  new Vector3(-0.05, SHAPE.cy + 0.8, 0.22),
  new Vector3(-0.1, SHAPE.cy + 1.45, 0.2),
  new Vector3(0.0, SHAPE.cy + 1.95, -0.15),
  new Vector3(0.35, SHAPE.cy + 2.0, -0.55),
  new Vector3(0.55, SHAPE.cy + 1.6, -0.75),
  new Vector3(0.6, SHAPE.cy + 0.9, -0.8),
]);
export const PULMONARY = new CatmullRomCurve3([
  new Vector3(0.3, SHAPE.cy + 0.8, 0.5),
  new Vector3(0.38, SHAPE.cy + 1.3, 0.45),
  new Vector3(0.45, SHAPE.cy + 1.62, 0.15),
  new Vector3(0.75, SHAPE.cy + 1.7, -0.1),
]);
export const SVC = new CatmullRomCurve3([
  new Vector3(-0.62, SHAPE.cy + 1.1, -0.1),
  new Vector3(-0.6, SHAPE.cy + 1.7, -0.15),
  new Vector3(-0.58, SHAPE.cy + 2.1, -0.2),
]);

// Anatomical heart from public/models/heart.glb (built from BodyParts3D by tools/build_heart_model.py).
// The real LAD, LCX and RCA meshes are coloured by their predicted stenosis probability, and the
// ventricular wall is split into the territory each artery supplies (nearest-artery assignment).

import { useEffect, useMemo, useRef, useState } from 'react';
import { useFrame, type ThreeEvent } from '@react-three/fiber';
import {
  AdditiveBlending, BackSide, Box3, BufferAttribute, BufferGeometry, Color, Group, Mesh, MeshPhysicalMaterial,
  MeshStandardMaterial, ShaderMaterial, Vector3, type Object3D,
} from 'three';
import { VESSELS, type Target, type Vessel } from '../types';
import { riskColor } from '../risk';
import { contraction } from '../beat';
import { LabelTracker, type LabelRefs } from './labels';
import { PALETTE } from './anatomy';

export const CORONARY_NODES = ['LAD', 'LCX', 'RCA'] as const;
export const hasCoronaryNodes = (root: Object3D) => CORONARY_NODES.every((n) => !!root.getObjectByName(n));

const NEUTRAL = '#4f7d68';
const TISSUE = new Color(PALETTE.tissue);
const DIM = new Color(PALETTE.dimTissue);
// Directions used to pick a readable label position on each vessel.
const LABEL_DIR: Record<Vessel, Vector3> = {
  LAD: new Vector3(0.25, 0, 1).normalize(),
  LCX: new Vector3(1, 0.4, -0.4).normalize(),
  RCA: new Vector3(-1, 0, 0.35).normalize(),
};

type Geoms = Record<string, BufferGeometry | undefined>;

function bakedGeometry(root: Object3D, name: string): BufferGeometry | undefined {
  const o = root.getObjectByName(name) as Mesh | undefined;
  if (!o || !(o as Mesh).isMesh) return undefined;
  o.updateWorldMatrix(true, false);
  const g = o.geometry.clone().applyMatrix4(o.matrixWorld);
  if (!g.attributes.normal) g.computeVertexNormals();
  return g;
}

/** Fresnel rim drawn as a slightly inflated back-face shell. */
function useRimMaterial() {
  return useMemo(() => new ShaderMaterial({
    uniforms: { color: { value: new Color(PALETTE.rim) } },
    vertexShader: `varying vec3 vN; varying vec3 vV;
      void main(){ vec3 p = position + normal * 0.018; vec4 mv = modelViewMatrix*vec4(p,1.0);
        vec3 n = normalMatrix*normal; vN = length(n) > 1e-5 ? normalize(n) : vec3(0.,0.,1.);
        vV = normalize(-mv.xyz); gl_Position = projectionMatrix*mv; }`,
    fragmentShader: `uniform vec3 color; varying vec3 vN; varying vec3 vV;
      void main(){ float f = pow(clamp(1.0-abs(dot(vN,vV)),0.0,1.0), 2.4); gl_FragColor = vec4(color*f*0.55, f*0.6); }`,
    transparent: true, blending: AdditiveBlending, depthWrite: false, side: BackSide,
  }), []);
}

function VesselMesh({ vessel, geometry, p, selected, hovered, dimmed, onSelect, setHover }: {
  vessel: Vessel; geometry: BufferGeometry; p: number | null; selected: boolean; hovered: boolean; dimmed: boolean;
  onSelect: () => void; setHover: (v: Vessel | null) => void;
}) {
  const color = p == null ? NEUTRAL : riskColor(p);
  const target = useMemo(() => new Color(color), [color]);
  const mat = useMemo(() => new MeshStandardMaterial({ roughness: 0.25, metalness: 0.1, toneMapped: false }), []);
  useEffect(() => {
    if (mat.userData.init) return;
    mat.userData.init = true;
    mat.color.copy(target);
    mat.emissive.copy(target);
  }, [mat, target]);
  useFrame(() => {
    const risk = p ?? 0;
    const base = selected ? 1.4 : hovered ? 1.1 : dimmed ? 0.25 : 0.75;
    mat.emissiveIntensity = base + (0.3 + 1.0 * risk) * contraction() * (dimmed ? 0.3 : 1);
    mat.color.lerp(target, 0.06);
    mat.emissive.lerp(target, 0.06);
  });
  return (
    <mesh
      geometry={geometry}
      material={mat}
      onClick={(e) => { e.stopPropagation(); onSelect(); }}
      onPointerOver={(e) => { e.stopPropagation(); setHover(vessel); }}
      onPointerOut={() => setHover(null)}
    />
  );
}

/**
 * Ventricular wall. Its vertex colours carry the LAD/LCX/RCA territory weights (built offline by
 * tools/build_heart_model.py); here they are turned into display colours, so a selected vessel's
 * territory fades in smoothly instead of ending at hard triangle edges.
 */
function Myocardium({ geometry, probs, selected, rim, onSelect, setHover }: {
  geometry: BufferGeometry; probs: Record<Vessel, number> | null; selected: Target; rim: ShaderMaterial;
  onSelect: (t: Target) => void; setHover: (v: Vessel | null) => void;
}) {
  const { weights, display } = useMemo(() => {
    const src = geometry.attributes.color as BufferAttribute | undefined;
    const n = geometry.attributes.position.count;
    const w = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      if (src) { w[i * 3] = src.getX(i); w[i * 3 + 1] = src.getY(i); w[i * 3 + 2] = src.getZ(i); }
      else { w[i * 3] = w[i * 3 + 1] = w[i * 3 + 2] = 1 / 3; }
    }
    const d = geometry.clone();
    d.deleteAttribute('color');
    d.setAttribute('color', new BufferAttribute(new Float32Array(n * 3), 3));
    return { weights: w, display: d };
  }, [geometry]);
  useEffect(() => () => display.dispose(), [display]);

  useEffect(() => {
    const col = display.attributes.color as BufferAttribute;
    const n = col.count;
    const focus = selected !== 'CAD';
    const si = focus ? VESSELS.indexOf(selected as Vessel) : -1;
    const risk = VESSELS.map((v) => new Color(probs ? riskColor(probs[v]) : NEUTRAL));
    const c = new Color();
    const lit = new Color();
    for (let i = 0; i < n; i++) {
      if (focus) {
        const w = weights[i * 3 + si];
        lit.copy(TISSUE).lerp(risk[si], 0.55);
        c.copy(DIM).lerp(TISSUE, 0.3).lerp(lit, Math.min(1, w * 1.15));
      } else {
        c.copy(TISSUE);
        for (let k = 0; k < 3; k++) c.lerp(risk[k], 0.04 * weights[i * 3 + k]);
      }
      col.setXYZ(i, c.r, c.g, c.b);
    }
    col.needsUpdate = true;
  }, [display, weights, probs, selected]);

  const vesselAt = (e: ThreeEvent<PointerEvent | MouseEvent>): Vessel | null => {
    if (!e.face) return null;
    const i = e.face.a * 3;
    const w = [weights[i], weights[i + 1], weights[i + 2]];
    return VESSELS[w.indexOf(Math.max(...w))];
  };

  return (
    <group>
      <mesh
        geometry={display}
        onClick={(e: ThreeEvent<MouseEvent>) => { e.stopPropagation(); const v = vesselAt(e); if (v) onSelect(selected === v ? 'CAD' : v); }}
        onPointerMove={(e) => { e.stopPropagation(); setHover(vesselAt(e)); }}
        onPointerOut={() => setHover(null)}
      >
        <meshPhysicalMaterial vertexColors roughness={0.5} metalness={0} clearcoat={0.45} clearcoatRoughness={0.3} sheen={0.5} sheenColor="#ffb0a0" sheenRoughness={0.6} />
      </mesh>
      <mesh geometry={display} material={rim} raycast={() => null} />
    </group>
  );
}

function Shell({ geometry, color, opacity = 1 }: { geometry?: BufferGeometry; color: string; opacity?: number }) {
  const mat = useMemo(() => new MeshPhysicalMaterial({
    color, emissive: color, emissiveIntensity: 0.05, roughness: 0.5, metalness: 0, clearcoat: 0.45, clearcoatRoughness: 0.3,
    transparent: opacity < 1, opacity,
  }), [color, opacity]);
  if (!geometry) return null;
  return <mesh geometry={geometry} material={mat} raycast={() => null} />;
}

export default function CoronaryHeart({ root, probs, selected, onSelect, labelRefs }: {
  root: Object3D;
  probs: Record<Vessel, number> | null;
  selected: Target;
  onSelect: (t: Target) => void;
  labelRefs: LabelRefs;
}) {
  const g: Geoms = useMemo(() => {
    const names = ['LAD', 'LCX', 'RCA', 'LeftMain', 'Myocardium',
      'LeftAtrium', 'RightAtrium', 'Aorta', 'PulmonaryTrunk', 'SVC', 'IVC'];
    return Object.fromEntries(names.map((n) => [n, bakedGeometry(root, n)]));
  }, [root]);
  useEffect(() => () => Object.values(g).forEach((x) => x?.dispose()), [g]);

  // Heart centre, scan-ring extent and label anchors from the actual geometry.
  const { center, anchors } = useMemo(() => {
    const box = new Box3();
    for (const k of ['Myocardium']) {
      const gg = g[k];
      if (gg) { gg.computeBoundingBox(); box.union(gg.boundingBox!); }
    }
    const c = box.getCenter(new Vector3());
    const anchors = {} as Record<Vessel, Vector3>;
    for (const v of VESSELS) {
      const pos = g[v]!.attributes.position;
      const dir = LABEL_DIR[v];
      let best = -Infinity;
      const p = new Vector3();
      const winner = new Vector3();
      for (let i = 0; i < pos.count; i += 3) {
        p.fromBufferAttribute(pos, i);
        // prefer points mid-height on the chosen side
        const score = p.clone().sub(c).dot(dir) - Math.abs(p.y - c.y) * 0.6;
        if (score > best) { best = score; winner.copy(p); }
      }
      anchors[v] = winner.clone().add(winner.clone().sub(c).normalize().multiplyScalar(0.22));
    }
    return { center: c, anchors };
  }, [g]);

  const [hover, setHover] = useState<Vessel | null>(null);
  useEffect(() => {
    document.body.style.cursor = hover ? 'pointer' : '';
    return () => { document.body.style.cursor = ''; };
  }, [hover]);

  const beat = useRef<Group>(null);
  useFrame(() => { if (beat.current) beat.current.scale.setScalar(1 + 0.035 * contraction()); });

  const rim = useRimMaterial();
  const focus = selected !== 'CAD';
  const leftMainP = probs ? Math.max(probs.LAD, probs.LCX) : null;

  return (
    <group position={[center.x, center.y, center.z]}>
      <group ref={beat}>
        <group position={[-center.x, -center.y, -center.z]}>
          {g.Myocardium && (
            <Myocardium geometry={g.Myocardium} probs={probs} selected={selected} rim={rim} onSelect={onSelect} setHover={setHover} />
          )}
          <Shell geometry={g.LeftAtrium} color={PALETTE.atrium} />
          <Shell geometry={g.RightAtrium} color={PALETTE.atrium} />
          <Shell geometry={g.Aorta} color={PALETTE.aorta} />
          <Shell geometry={g.PulmonaryTrunk} color={PALETTE.pulmonary} />
          <Shell geometry={g.SVC} color={PALETTE.vein} />
          <Shell geometry={g.IVC} color={PALETTE.vein} />
          {g.LeftMain && (
            <Shell geometry={g.LeftMain} color={leftMainP == null ? NEUTRAL : riskColor(leftMainP)} />
          )}
          {VESSELS.map((v) => g[v] && (
            <VesselMesh key={v} vessel={v} geometry={g[v]!} p={probs ? probs[v] : null}
              selected={selected === v} hovered={hover === v} dimmed={focus && selected !== v}
              onSelect={() => onSelect(selected === v ? 'CAD' : v)} setHover={setHover} />
          ))}
          <LabelTracker anchors={anchors} center={center} labelRefs={labelRefs} />
        </group>
      </group>
    </group>
  );
}

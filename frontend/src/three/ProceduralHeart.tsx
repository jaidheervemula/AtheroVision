// Procedural heart: myocardium, great vessels and coronary tubes generated in code (see anatomy.ts).
// Used when the anatomical model is switched off, and as the vessel overlay for a generic .glb heart.

import { Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { useFrame, type ThreeEvent } from '@react-three/fiber';
import { AdditiveBlending, BackSide, BufferAttribute, Color, Group, MeshStandardMaterial, ShaderMaterial, TubeGeometry, Vector3 } from 'three';
import AnatomicalHeart from './AnatomicalHeart';
import { type Align } from './modelConfig';
import { LabelTracker, type LabelRefs } from './labels';
import { AORTA, PALETTE, PULMONARY, SVC, SHAPE, TISSUE, buildMyocardium, buildVesselCurves } from './anatomy';
import { VESSELS, type Target, type Vessel } from '../types';
import { riskColor } from '../risk';
import { contraction, setBpm } from '../beat';

export type VesselProbs = Record<Vessel, number> | null;

export interface HeartProps {
  probs: VesselProbs;
  selected: Target;
  onSelect: (t: Target) => void;
  labelRefs: LabelRefs;
  pulseRate: number;
  anatomical: boolean;
  align: Align;
}

export const NEUTRAL = '#4f7d68';
const HEART_CENTER = new Vector3(0, SHAPE.cy, 0);

function Myocardium({ probs, selected, onSelect, setHover }: {
  probs: VesselProbs; selected: Target; onSelect: (t: Target) => void; setHover: (v: Vessel | null) => void;
}) {
  const { geometry, territory } = useMemo(buildMyocardium, []);

  // Tint vessel territories: all faintly in overview, the selected one strongly.
  useEffect(() => {
    const col = geometry.attributes.color as BufferAttribute;
    const tint = new Color();
    const c = new Color();
    const dim = new Color(PALETTE.dimTissue);
    const focus = selected !== 'CAD';
    for (let i = 0; i < territory.length; i++) {
      const t = territory[i];
      c.copy(TISSUE);
      if (focus && t !== selected) {
        c.lerp(dim, 0.7); // fade everything outside the selected vessel's territory
      } else if (t && probs) {
        tint.set(riskColor(probs[t]));
        c.lerp(tint, focus ? 0.55 : 0.04);
      }
      col.setXYZ(i, c.r, c.g, c.b);
    }
    col.needsUpdate = true;
  }, [geometry, territory, probs, selected]);

  const vesselAt = (e: ThreeEvent<PointerEvent | MouseEvent>) => (e.face ? territory[e.face.a] : null);

  return (
    <mesh
      geometry={geometry}
      onClick={(e) => { e.stopPropagation(); const v = vesselAt(e); onSelect(v && v !== selected ? v : 'CAD'); }}
      onPointerMove={(e) => { e.stopPropagation(); setHover(vesselAt(e)); }}
      onPointerOut={() => setHover(null)}
    >
      <meshPhysicalMaterial vertexColors roughness={0.5} metalness={0} clearcoat={0.45} clearcoatRoughness={0.3} sheen={0.5} sheenColor="#ffb0a0" sheenRoughness={0.6} />
    </mesh>
  );
}

/** Holographic shell: a coarse cyan wireframe plus a magenta Fresnel rim glow. */
function Hologram() {
  const wire = useMemo(() => buildMyocardium(40, 28).geometry, []);
  const rim = useMemo(() => new ShaderMaterial({
    uniforms: { color: { value: new Color(PALETTE.rim) }, power: { value: 2.4 } },
    vertexShader: `varying vec3 vN; varying vec3 vV;
      void main(){ vec4 mv = modelViewMatrix * vec4(position,1.0); vN = normalize(normalMatrix*normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix*mv; }`,
    fragmentShader: `uniform vec3 color; uniform float power; varying vec3 vN; varying vec3 vV;
      void main(){ float f = pow(1.0 - abs(dot(vN, vV)), power); gl_FragColor = vec4(color * f * 0.55, f * 0.6); }`,
    transparent: true, blending: AdditiveBlending, depthWrite: false, side: BackSide,
  }), []);
  return (
    <>
      <mesh geometry={wire} scale={1.012} raycast={() => null}>
        <meshBasicMaterial color="#ffd0c0" wireframe transparent opacity={0.02} depthWrite={false} />
      </mesh>
      <mesh geometry={wire} scale={1.06} material={rim} raycast={() => null} />
    </>
  );
}

function VesselTube({ vessel, curves, p, selected, hovered, onSelect, setHover }: {
  vessel: Vessel; curves: ReturnType<typeof buildVesselCurves>[Vessel]; p: number | null;
  selected: boolean; hovered: boolean; onSelect: () => void; setHover: (v: Vessel | null) => void;
}) {
  const thick = selected || hovered;
  const geoms = useMemo(() => {
    const k = thick ? 1.6 : 1;
    return [
      new TubeGeometry(curves.trunk, 160, 0.05 * k, 12, false),
      ...curves.branches.map((b) => new TubeGeometry(b, 80, 0.032 * k, 10, false)),
    ];
  }, [curves, thick]);
  useEffect(() => () => geoms.forEach((g) => g.dispose()), [geoms]);

  const color = p == null ? NEUTRAL : riskColor(p);
  const mat = useMemo(() => new MeshStandardMaterial({ roughness: 0.25, metalness: 0.1, toneMapped: false }), []);
  // Ease towards the new colour so scenario changes are visible as a transition.
  const target = useMemo(() => new Color(color), [color]);
  useEffect(() => { if (mat.userData.init) return; mat.userData.init = true; mat.color.copy(target); mat.emissive.copy(target); }, [mat, target]);

  // Flash with every heartbeat; riskier vessels flash harder.
  useFrame(() => {
    const risk = p ?? 0;
    const base = selected ? 1.1 : hovered ? 0.9 : 0.6;
    const pulse = (0.25 + 0.9 * risk) * contraction();
    mat.emissiveIntensity = base + pulse;
    mat.color.lerp(target, 0.06);
    mat.emissive.lerp(target, 0.06);
  });

  return (
    <group>
      {geoms.map((g, i) => (
        <mesh
          key={i}
          geometry={g}
          material={mat}
          onClick={(e) => { e.stopPropagation(); onSelect(); }}
          onPointerOver={(e) => { e.stopPropagation(); setHover(vessel); }}
          onPointerOut={() => setHover(null)}
        />
      ))}
    </group>
  );
}

function Tube({ curve, r, color }: { curve: typeof AORTA; r: number; color: string }) {
  const g = useMemo(() => new TubeGeometry(curve, 64, r, 20, false), [curve, r]);
  return (
    <mesh geometry={g} raycast={() => null}>
      <meshPhysicalMaterial color={color} emissive={color} emissiveIntensity={0.05} roughness={0.5} metalness={0} clearcoat={0.45} />
    </mesh>
  );
}

export function Heart({ probs, selected, onSelect, labelRefs, pulseRate, anatomical, align }: HeartProps) {
  const beat = useRef<Group>(null);
  useEffect(() => setBpm(pulseRate), [pulseRate]);
  useFrame(() => {
    if (beat.current) beat.current.scale.setScalar(1 + 0.04 * contraction());
  });
  const curves = useMemo(() => buildVesselCurves(), []);
  const anchors = useMemo(() => ({ LAD: curves.LAD.labelAnchor, LCX: curves.LCX.labelAnchor, RCA: curves.RCA.labelAnchor }), [curves]);
  const [hover, setHover] = useState<Vessel | null>(null);

  useEffect(() => {
    document.body.style.cursor = hover ? 'pointer' : '';
    return () => { document.body.style.cursor = ''; };
  }, [hover]);

  return (
    // Tilt so the apex points down and to the patient's left, as in an anterior view.
    <group rotation={[0.1, 0.12, 0.5]} position={[-0.05, -0.25, 0]}>
     <group ref={beat} position={[0, SHAPE.cy, 0]}>
     <group position={[0, -SHAPE.cy, 0]}>
      {anatomical ? (
        <Suspense fallback={null}>
          <AnatomicalHeart align={align} />
        </Suspense>
      ) : (
      <>
      <Myocardium probs={probs} selected={selected} onSelect={onSelect} setHover={setHover} />
      <Hologram />
      {/* Atria (left behind, right above the right AV groove) */}
      <mesh position={[0.4, SHAPE.cy + 1.0, -0.48]} scale={[0.52, 0.4, 0.42]} raycast={() => null}>
        <sphereGeometry args={[1, 48, 32]} />
        <meshPhysicalMaterial color={PALETTE.atrium} roughness={0.5} metalness={0} clearcoat={0.45} />
      </mesh>
      <mesh position={[-0.6, SHAPE.cy + 1.2, -0.28]} scale={[0.4, 0.34, 0.42]} raycast={() => null}>
        <sphereGeometry args={[1, 48, 32]} />
        <meshPhysicalMaterial color={PALETTE.atrium} roughness={0.5} metalness={0} clearcoat={0.45} />
      </mesh>
      <Tube curve={AORTA} r={0.21} color={PALETTE.aorta} />
      <Tube curve={PULMONARY} r={0.17} color={PALETTE.pulmonary} />
      <Tube curve={SVC} r={0.14} color={PALETTE.vein} />
      </>
      )}
      <Tube curve={curves.leftMain} r={0.055} color={probs ? riskColor(Math.max(probs.LAD, probs.LCX)) : NEUTRAL} />
      {VESSELS.map((v) => (
        <VesselTube
          key={v}
          vessel={v}
          curves={curves[v]}
          p={probs ? probs[v] : null}
          selected={selected === v}
          hovered={hover === v}
          onSelect={() => onSelect(selected === v ? 'CAD' : v)}
          setHover={setHover}
        />
      ))}
      <LabelTracker anchors={anchors} center={HEART_CENTER} labelRefs={labelRefs} />
     </group>
     </group>
    </group>
  );
}


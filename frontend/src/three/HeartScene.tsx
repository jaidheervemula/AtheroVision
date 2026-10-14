import { Suspense, useEffect, useMemo, useRef, useState, type MutableRefObject } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, PerformanceMonitor, useGLTF, useProgress } from '@react-three/drei';
import { Bloom, EffectComposer } from '@react-three/postprocessing';
import { CanvasTexture, SRGBColorSpace } from 'three';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import { MODEL_URL, type Align } from './modelConfig';
import CoronaryHeart, { hasCoronaryNodes } from './CoronaryHeart';
import { type LabelRefs } from './labels';
import { Heart, NEUTRAL, type HeartProps, type VesselProbs } from './ProceduralHeart';
import { VESSELS, type Target } from '../types';
import { pct, riskColor } from '../risk';
import { setBpm } from '../beat';

export type { VesselProbs } from './ProceduralHeart';

interface Props {
  probs: VesselProbs;
  selected: Target;
  onSelect: (t: Target) => void;
  autoRotate: boolean;
  resetKey: number;
  /** Beats per minute for the heartbeat animation (from the patient's pulse rate). */
  pulseRate: number;
  /** Bloom glow on the arteries. Turn off on slow machines. */
  fx: boolean;
  /** Use public/models/heart.glb instead of the procedural myocardium. */
  anatomical: boolean;
  align: Align;
  /** The colours shown come from the what-if simulator, not the current patient. */
  simulated: boolean;
  /** Current-patient probabilities, shown as "was" on the labels during a simulation. */
  baseProbs?: VesselProbs;
  /** Reports whether the loaded .glb carries its own LAD/LCX/RCA meshes or is a plain heart shell. */
  onModelKind?: (kind: 'coronary' | 'generic') => void;
  /** Filled with a function that renders one frame and returns it as a PNG data URL (for reports). */
  captureRef?: MutableRefObject<(() => string | null) | null>;
  /** Called once if the frame rate stays low, so the app can turn effects off. */
  onLowPerformance?: () => void;
}

function Effects() {
  // A gentle glow on the emissive coronary arteries only; no lens effects.
  return (
    <EffectComposer multisampling={0}>
      <Bloom mipmapBlur intensity={0.55} luminanceThreshold={0.8} luminanceSmoothing={0.15} radius={0.55} />
    </EffectComposer>
  );
}

/** Real anatomy: BodyParts3D coronaries if the .glb has them, otherwise a generic shell over procedural vessels. */
function AnatomicalSwitch(props: HeartProps & { onModelKind?: Props['onModelKind'] }) {
  const gltf = useGLTF(MODEL_URL);
  const coronary = useMemo(() => hasCoronaryNodes(gltf.scene), [gltf]);
  const { onModelKind } = props;
  useEffect(() => { onModelKind?.(coronary ? 'coronary' : 'generic'); }, [coronary, onModelKind]);
  useEffect(() => setBpm(props.pulseRate), [props.pulseRate]);
  if (coronary) {
    return <CoronaryHeart root={gltf.scene} probs={props.probs} selected={props.selected} onSelect={props.onSelect} labelRefs={props.labelRefs} />;
  }
  return <Heart {...props} />;
}

// Deep phosphor-green backdrop (ICU-monitor green) with a soft glow behind the heart. Green is
// red's complement, so the red muscle and aorta stand out against it.
const BG_EDGE = '#0b1612';
function Backdrop() {
  const { scene } = useThree();
  useEffect(() => {
    const c = document.createElement('canvas');
    c.width = c.height = 512;
    const ctx = c.getContext('2d')!;
    const g = ctx.createRadialGradient(256, 230, 10, 256, 256, 360);
    g.addColorStop(0, '#1a3a30');
    g.addColorStop(0.5, '#11251e');
    g.addColorStop(1, BG_EDGE);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 512, 512);
    const tex = new CanvasTexture(c);
    tex.colorSpace = SRGBColorSpace;
    const prev = scene.background;
    scene.background = tex;
    return () => { scene.background = prev; tex.dispose(); };
  }, [scene]);
  return null;
}

function Controls({ autoRotate, resetKey }: { autoRotate: boolean; resetKey: number }) {
  const ref = useRef<OrbitControlsImpl>(null);
  useEffect(() => { ref.current?.reset(); }, [resetKey]);
  return (
    <OrbitControls
      ref={ref}
      makeDefault
      enableDamping
      enablePan={false}
      minDistance={3}
      maxDistance={11}
      autoRotate={autoRotate}
      autoRotateSpeed={0.8}
      target={[0, 0.3, 0]}
    />
  );
}

/** Lets the report capture the current view without keeping the drawing buffer alive every frame. */
function Capture({ captureRef }: { captureRef?: Props['captureRef'] }) {
  const { gl, scene, camera } = useThree();
  useEffect(() => {
    if (!captureRef) return;
    captureRef.current = () => {
      try { gl.render(scene, camera); return gl.domElement.toDataURL('image/png'); } catch { return null; }
    };
    return () => { captureRef.current = null; };
  }, [captureRef, gl, scene, camera]);
  return null;
}

function LoadingOverlay() {
  const { active, progress } = useProgress();
  if (!active) return null;
  return (
    <div className="scene__loading" role="status">
      <span>Loading heart model</span>
      <div className="scene__progress"><div style={{ width: `${Math.round(progress)}%` }} /></div>
    </div>
  );
}

export default function HeartScene(props: Props) {
  const [dpr, setDpr] = useState(Math.min(2, window.devicePixelRatio || 1));
  const declined = useRef(false);
  const labelRefs: LabelRefs = useRef({ LAD: null, LCX: null, RCA: null });
  const heartProps = {
    probs: props.probs, selected: props.selected, onSelect: props.onSelect, labelRefs,
    pulseRate: props.pulseRate, anatomical: props.anatomical, align: props.align,
  };
  return (
    <div className="scene">
      <Canvas
        dpr={dpr}
        camera={{ position: [0, 0.6, 6.1], fov: 38 }}
        gl={{ antialias: true, powerPreference: 'low-power' }}
        onPointerMissed={() => props.onSelect('CAD')}
      >
        <PerformanceMonitor
          onDecline={() => {
            setDpr(1);
            if (!declined.current) { declined.current = true; props.onLowPerformance?.(); }
          }}
        />
        <Capture captureRef={props.captureRef} />
        <Backdrop />
        <fog attach="fog" args={[BG_EDGE, 7, 16]} />
        <hemisphereLight args={['#fff6f0', '#2a1210', 0.8]} />
        <directionalLight position={[3, 4, 5]} intensity={2.2} color="#fff4ec" />
        <directionalLight position={[-3, 1, -4]} intensity={0.7} color="#ffe2d6" />
        <pointLight position={[0, -2.5, 3]} intensity={4} color="#ffd9cc" distance={12} />
        {props.anatomical ? (
          <Suspense fallback={null}>
            <AnatomicalSwitch {...heartProps} onModelKind={props.onModelKind} />
          </Suspense>
        ) : (
          <Heart {...heartProps} anatomical={false} />
        )}
        <Controls autoRotate={props.autoRotate} resetKey={props.resetKey} />
        {props.fx && <Effects />}
      </Canvas>
      {props.anatomical && <LoadingOverlay />}
      <div className="scene__labels">
        {VESSELS.map((v) => {
          const p = props.probs ? props.probs[v] : null;
          const color = p == null ? NEUTRAL : riskColor(p);
          return (
            <button
              key={v}
              ref={(el) => { labelRefs.current[v] = el; }}
              className={`vessel-tag${props.selected === v ? ' is-selected' : ''}`}
              style={{ ['--risk' as string]: color }}
              onClick={() => props.onSelect(props.selected === v ? 'CAD' : v)}
            >
              <span className="vessel-tag__dot" />
              {v}
              <span className="vessel-tag__pct">{p == null ? '—' : pct(p)}</span>
              {props.simulated && props.baseProbs && p != null && Math.round(props.baseProbs[v] * 100) !== Math.round(p * 100) && (
                <span className="vessel-tag__was">was {pct(props.baseProbs[v])}</span>
              )}
              {props.simulated && <span className="vessel-tag__sim">What-if</span>}
            </button>
          );
        })}
      </div>
    </div>
  );
}

import { useMemo, useRef, type MutableRefObject } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { Object3D, Vector3 } from 'three';
import { VESSELS, type Vessel } from '../types';

export type LabelRefs = MutableRefObject<Record<Vessel, HTMLButtonElement | null>>;

interface Placed { v: Vessel; el: HTMLButtonElement; x: number; y: number; w: number; h: number; facing: number }

/**
 * Projects each vessel's label anchor to screen space every frame and moves the DOM label.
 * Labels on the far side of the heart fade, and labels that would overlap are pushed apart
 * vertically (the label nearer the viewer keeps its place).
 */
export function LabelTracker({ anchors, center, labelRefs }: {
  anchors: Record<Vessel, Vector3>;
  center: Vector3;
  labelRefs: LabelRefs;
}) {
  const markers = useRef<Record<Vessel, Object3D | null>>({ LAD: null, LCX: null, RCA: null });
  const centerRef = useRef<Object3D>(null);
  const { camera, size } = useThree();
  const tmp = useMemo(() => ({ p: new Vector3(), c: new Vector3(), n: new Vector3(), v: new Vector3() }), []);

  useFrame(() => {
    if (!centerRef.current) return;
    centerRef.current.getWorldPosition(tmp.c);
    const placed: Placed[] = [];
    for (const v of VESSELS) {
      const m = markers.current[v];
      const el = labelRefs.current[v];
      if (!m || !el) continue;
      m.getWorldPosition(tmp.p);
      tmp.n.copy(tmp.p).sub(tmp.c).normalize();
      tmp.v.copy(camera.position).sub(tmp.p).normalize();
      const facing = tmp.n.dot(tmp.v);
      tmp.p.project(camera);
      placed.push({
        v, el, facing,
        x: ((tmp.p.x + 1) / 2) * size.width,
        y: ((1 - tmp.p.y) / 2) * size.height,
        w: el.offsetWidth || 90,
        h: el.offsetHeight || 24,
      });
    }

    // Resolve overlaps: front-facing labels first, others move out of their way.
    placed.sort((a, b) => b.facing - a.facing);
    const GAP = 4;
    for (let i = 1; i < placed.length; i++) {
      for (let pass = 0; pass < 3; pass++) {
        let moved = false;
        for (let j = 0; j < i; j++) {
          const a = placed[j], b = placed[i];
          const overlapX = Math.abs(a.x - b.x) < (a.w + b.w) / 2 + GAP;
          const overlapY = Math.abs(a.y - b.y) < (a.h + b.h) / 2 + GAP;
          if (overlapX && overlapY) {
            const dir = b.y >= a.y ? 1 : -1;
            b.y = a.y + dir * ((a.h + b.h) / 2 + GAP);
            moved = true;
          }
        }
        if (!moved) break;
      }
    }

    for (const p of placed) {
      p.el.style.transform = `translate(-50%, -50%) translate(${p.x}px, ${p.y}px)`;
      p.el.style.opacity = p.facing > 0.05 ? '1' : '0.4';
      p.el.style.zIndex = p.facing > 0.05 ? '3' : '2';
    }
  });

  return (
    <>
      <object3D ref={centerRef} position={center} />
      {VESSELS.map((v) => (
        <object3D key={v} position={anchors[v]} ref={(o) => { markers.current[v] = o; }} />
      ))}
    </>
  );
}

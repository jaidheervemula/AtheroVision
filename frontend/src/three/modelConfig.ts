// Model file locations and alignment settings. Deliberately free of three.js imports so the
// dashboard can use them without pulling the 3D bundle into the first download.

import { useEffect, useState } from 'react';

export const MODEL_URL = '/models/heart.glb';
export const ALIGN_URL = '/models/heart.json';

export interface Align {
  scale: number;
  x: number; y: number; z: number;
  rx: number; ry: number; rz: number; // degrees
}
export const DEFAULT_ALIGN: Align = { scale: 1, x: 0, y: 0, z: 0, rx: 0, ry: 0, rz: 0 };

/**
 * Checks the file really is a binary glTF (Vite's dev server answers missing files with index.html).
 * Returns null while checking, so the viewer can wait instead of flashing the procedural heart.
 */
export function useModelAvailable(url = MODEL_URL): boolean | null {
  const [ok, setOk] = useState<boolean | null>(null);
  useEffect(() => {
    let alive = true;
    fetch(url, { headers: { Range: 'bytes=0-3' } })
      .then((r) => (r.ok ? r.arrayBuffer() : null))
      .then((buf) => {
        if (!alive) return;
        if (!buf || buf.byteLength < 4) { setOk(false); return; }
        const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 4));
        setOk(magic === 'glTF');
      })
      .catch(() => { if (alive) setOk(false); });
    return () => { alive = false; };
  }, [url]);
  return ok;
}

/** Team-shared alignment from public/models/heart.json, if present. */
export async function fetchSharedAlign(): Promise<Align | null> {
  try {
    const r = await fetch(ALIGN_URL);
    if (!r.ok) return null;
    const j = JSON.parse(await r.text());
    return { ...DEFAULT_ALIGN, ...j };
  } catch {
    return null;
  }
}

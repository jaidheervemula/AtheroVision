import { useState } from 'react';
import { DEFAULT_ALIGN, type Align } from '../three/modelConfig';

// Fine-tune how the anatomical mesh sits over the procedural coronary vessels.
// "Copy JSON" gives the contents for public/models/heart.json so teammates share it.

const SLIDERS: { key: keyof Align; label: string; min: number; max: number; step: number }[] = [
  { key: 'scale', label: 'Scale', min: 0.4, max: 2, step: 0.01 },
  { key: 'x', label: 'Move X', min: -2, max: 2, step: 0.01 },
  { key: 'y', label: 'Move Y', min: -2, max: 2, step: 0.01 },
  { key: 'z', label: 'Move Z', min: -2, max: 2, step: 0.01 },
  { key: 'rx', label: 'Rotate X', min: -180, max: 180, step: 1 },
  { key: 'ry', label: 'Rotate Y', min: -180, max: 180, step: 1 },
  { key: 'rz', label: 'Rotate Z', min: -180, max: 180, step: 1 },
];

export default function AlignPanel({ align, onChange, onClose }: { align: Align; onChange: (a: Align) => void; onClose: () => void }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(align, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* clipboard blocked; the values are visible on screen */ }
  };
  return (
    <div className="align" role="dialog" aria-label="Align anatomical model">
      <div className="align__head">
        <b>Align model</b>
        <button className="align__x" onClick={onClose} aria-label="Close">×</button>
      </div>
      {SLIDERS.map((s) => (
        <label key={s.key} className="align__row">
          <span>{s.label}</span>
          <input type="range" min={s.min} max={s.max} step={s.step} value={align[s.key]}
            onChange={(e) => onChange({ ...align, [s.key]: Number(e.target.value) })} />
          <code>{align[s.key]}</code>
        </label>
      ))}
      <div className="align__buttons">
        <button className="chip" onClick={() => onChange(DEFAULT_ALIGN)}>Reset</button>
        <button className="chip" onClick={copy}>{copied ? 'Copied' : 'Copy JSON'}</button>
      </div>
      <p className="muted small">Line the coloured vessels up with the surface. Save the JSON as <code>public/models/heart.json</code> to share it.</p>
    </div>
  );
}

import { useState } from 'react';
import type { CalibrationInfo, Target } from '../types';

// Predicted vs observed (reliability) plot for one target: each point is a group of ~30
// patients sorted by cross-validated probability; whiskers are 95% Wilson intervals.
// Points on the dashed diagonal mean the probabilities can be read at face value.

const W = 132;
const H = 132;
const PAD = { l: 22, r: 6, t: 6, b: 18 };
const x = (v: number) => PAD.l + v * (W - PAD.l - PAD.r);
const y = (v: number) => H - PAD.b - v * (H - PAD.t - PAD.b);
const pct = (v: number) => `${Math.round(v * 100)}%`;

export default function CalibrationPlot({ target, info }: { target: Target; info: CalibrationInfo }) {
  const [hover, setHover] = useState<number | null>(null);
  const slope = info.calibration_slope;
  const h = hover == null ? null : info.bins[hover];
  return (
    <figure className="calib" aria-label={`${target} calibration: predicted versus observed`}>
      <figcaption>
        <b>{target}</b>
        {slope && <span className="muted"> slope {slope.estimate.toFixed(2)}</span>}
      </figcaption>
      <svg
        viewBox={`0 0 ${W} ${H}`} role="img" aria-hidden
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          // Nearest point to the cursor, so close-together points are all reachable.
          const r = e.currentTarget.getBoundingClientRect();
          const mx = ((e.clientX - r.left) / r.width) * W;
          const my = ((e.clientY - r.top) / r.height) * H;
          let best = -1;
          let bestD = Infinity;
          info.bins.forEach((b, i) => {
            const d = (x(b.mean_predicted) - mx) ** 2 + (y(b.observed) - my) ** 2;
            if (d < bestD) { bestD = d; best = i; }
          });
          setHover(bestD < 30 ** 2 ? best : null);
        }}
      >
        {[0, 0.5, 1].map((g) => (
          <g key={g}>
            <line className="calib__grid" x1={x(0)} x2={x(1)} y1={y(g)} y2={y(g)} />
            <text className="calib__tick" x={PAD.l - 4} y={y(g) + 3} textAnchor="end">{pct(g)}</text>
            <text className="calib__tick" x={x(g)} y={H - 5} textAnchor="middle">{pct(g)}</text>
          </g>
        ))}
        <line className="calib__diag" x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} />
        {info.bins.map((b, i) => (
          <g key={i} className={hover === i ? 'is-hover' : ''}>
            <line className="calib__ci" x1={x(b.mean_predicted)} x2={x(b.mean_predicted)} y1={y(b.ci_low)} y2={y(b.ci_high)} />
            <circle className="calib__pt" cx={x(b.mean_predicted)} cy={y(b.observed)} r={hover === i ? 5 : 4} />
          </g>
        ))}
      </svg>
      <p className="calib__tip" aria-live="polite">
        {h ? <>Predicted <b>{pct(h.mean_predicted)}</b> · observed <b>{pct(h.observed)}</b> ({pct(h.ci_low)}–{pct(h.ci_high)}), {h.n} patients</>
          : <span className="muted">Hover a point</span>}
      </p>
    </figure>
  );
}

// One shared heartbeat clock. The ECG trace, the 3D heart's contraction, the vessel glow
// and the ♥ icon all read this, so they stay in step and all change speed together when
// the pulse rate changes (including in the what-if simulator).
//
// Phase 0..1 over one cardiac cycle, aligned to the ECG waveform: the QRS spike is at ~0.22,
// and the mechanical contraction (systole) follows it at ~0.28.

export const heartbeat = { bpm: 72, phase: 0 };

export function setBpm(bpm: number) {
  heartbeat.bpm = Math.min(160, Math.max(35, Number.isFinite(bpm) && bpm > 0 ? bpm : 72));
}

const g = (x: number, m: number, w: number) => Math.exp(-(((x - m) / w) ** 2));

/** Contraction strength 0..1: a strong systolic squeeze after the QRS and a smaller atrial kick. */
export function contraction(phase = heartbeat.phase): number {
  return Math.min(1, g(phase, 0.3, 0.06) + 0.35 * g(phase, 0.1, 0.04));
}

let last = performance.now();
function tick(now: number) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  heartbeat.phase = (heartbeat.phase + (dt * heartbeat.bpm) / 60) % 1;
  requestAnimationFrame(tick);
}
if (typeof window !== 'undefined') requestAnimationFrame(tick);

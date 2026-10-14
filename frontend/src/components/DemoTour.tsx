import { useEffect } from 'react';

// Presenter mode: a fixed caption bar that walks through the key screens in a set order.
// Each step runs its action (load a case, select a vessel, open the what-if scenario) and
// scrolls the relevant panel into view, so a live demo can't drift off-script.
// Keys: → / Space next, ← back, Esc exit.

export interface TourStep {
  title: string;
  text: string;
  /** CSS selector of the panel to bring into view. */
  focus?: string;
  /** Sets up the screen for this step; called by the parent when the step opens. */
  run: () => void | Promise<void>;
}

interface Props {
  steps: TourStep[];
  index: number;
  onIndex: (i: number) => void;
  onExit: () => void;
}

export default function DemoTour({ steps, index, onIndex, onExit }: Props) {
  const step = steps[index];

  // The step's action is run by the parent when the step is opened; here we only bring its panel into view.
  const focus = step.focus;
  useEffect(() => {
    if (!focus) return;
    const t = setTimeout(() => {
      const el = document.querySelector(focus);
      el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      el?.classList.add('tour-focus');
      setTimeout(() => el?.classList.remove('tour-focus'), 2200);
    }, 700); // let the new result render first
    return () => clearTimeout(t);
  }, [index, focus]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest('input, select, textarea')) return;
      if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); if (index < steps.length - 1) onIndex(index + 1); else onExit(); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); if (index > 0) onIndex(index - 1); }
      else if (e.key === 'Escape') onExit();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [index, steps.length, onIndex, onExit]);

  const last = index === steps.length - 1;
  return (
    <div className="tour" role="dialog" aria-label="Guided demo" aria-live="polite">
      <div className="tour__progress" aria-hidden>
        {steps.map((s, i) => (
          <button key={s.title} className={i === index ? 'on' : i < index ? 'done' : ''} onClick={() => onIndex(i)} tabIndex={-1} title={s.title} />
        ))}
      </div>
      <div className="tour__body">
        <span className="tour__count">{index + 1}/{steps.length}</span>
        <div>
          <h2 className="tour__title">{step.title}</h2>
          <p className="tour__text">{step.text}</p>
        </div>
      </div>
      <div className="tour__nav">
        <button className="chip" onClick={() => onIndex(index - 1)} disabled={index === 0}>← Back</button>
        <button className="chip chip--on" onClick={() => (last ? onExit() : onIndex(index + 1))}>{last ? 'Finish' : 'Next →'}</button>
        <button className="tour__x" onClick={onExit} aria-label="Exit guided demo">×</button>
      </div>
    </div>
  );
}

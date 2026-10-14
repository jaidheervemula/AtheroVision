import { clinicalFlags, outOfRange } from '../clinical';
import type { PatientFeatures } from '../types';

interface Props {
  features: PatientFeatures;
  ranges?: Record<string, [number, number]>;
  compact?: boolean;
}

/** Danger/warning chips plus an "outside training data" reliability notice. */
export default function Alerts({ features, ranges, compact = false }: Props) {
  const flags = clinicalFlags(features);
  const ood = outOfRange(features, ranges);
  if (!flags.length && !ood.length) {
    return compact ? null : <div className="alerts alerts--clear"><span className="alerts__ok">No critical values</span></div>;
  }
  return (
    <div className={`alerts${compact ? ' alerts--compact' : ''}`} role="status">
      {flags.length > 0 && (
        <div className="alerts__flags">
          {flags.map((f) => (
            <span key={f.key} className={`alert alert--${f.level}`}>{f.text}</span>
          ))}
        </div>
      )}
      {ood.length > 0 && (
        <div className="alerts__ood">
          <b>Outside the training data, so this prediction is less reliable:</b>{' '}
          {ood.map((o, i) => (
            <span key={o.key}>
              {i > 0 && ', '}
              {o.label} {o.value}{o.unit && ` ${o.unit}`} <span className="muted">(trained on {o.range[0]}–{o.range[1]})</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

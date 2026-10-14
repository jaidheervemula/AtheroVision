import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from 'react';
// The 3D view (three.js) loads as its own chunk so the dashboard appears first.
const HeartScene = lazy(() => import('./three/HeartScene'));
import PatientForm from './components/PatientForm';
import RiskSummary from './components/RiskSummary';
import ContributionChart from './components/ContributionChart';
import MeasurementsTable from './components/MeasurementsTable';
import MetricsCard from './components/MetricsCard';
import EcgMonitor from './components/EcgMonitor';
import Vitals from './components/Vitals';
import WhatIf from './components/WhatIf';
import AlignPanel from './components/AlignPanel';
import Alerts from './components/Alerts';
import AppHeader from './components/AppHeader';
import PrintReport from './components/PrintReport';
import DemoTour, { type TourStep } from './components/DemoTour';
import { DEFAULT_ALIGN, fetchSharedAlign, useModelAvailable, type Align } from './three/modelConfig';
import { NEW_CASE, PRESETS, withDerived } from './features';
import { apiMode, fetchDatasetPatient, fetchMetrics, predict } from './api/client';
import { DEMO_NOTE, DISCLAIMER } from './disclaimer';
import { TARGETS, type DatasetPatient, type FeatureValue, type PatientFeatures, type PredictionResponse, type Target } from './types';

// Browser storage can be unavailable (private mode, blocked site data) — never let it break the app.
const store = {
  get<T>(k: string, fallback: T, s: 'local' | 'session' = 'local'): T {
    try {
      const v = (s === 'local' ? localStorage : sessionStorage).getItem(k);
      return v == null ? fallback : (JSON.parse(v) as T);
    } catch { return fallback; }
  },
  set(k: string, v: unknown, s: 'local' | 'session' = 'local') {
    try { (s === 'local' ? localStorage : sessionStorage).setItem(k, JSON.stringify(v)); } catch { /* ignore */ }
  },
};

/** Carry the last explanation onto a probabilities-only result, marked as not yet updated. */
function withPreviousExplanation(fast: PredictionResponse, prev: PredictionResponse | null): PredictionResponse {
  if (!prev) return { ...fast, explained: false };
  const predictions = { ...fast.predictions };
  for (const t of TARGETS) {
    const old = prev.predictions[t];
    predictions[t] = {
      ...old, // explanation fields
      probability: fast.predictions[t].probability,
      label: fast.predictions[t].label,
      threshold: fast.predictions[t].threshold,
      default_label: fast.predictions[t].default_label,
      default_threshold: fast.predictions[t].default_threshold,
      operating_point: fast.predictions[t].operating_point,
      reality_check: fast.predictions[t].reality_check,
    };
  }
  return { ...fast, predictions, explained: false };
}

/**
 * Re-predict whenever `features` changes. With the real model service this is done in two
 * steps: probabilities first (about 0.1 s, so the numbers and 3D colours update at once), then
 * explanations (about 2 s, mostly the LAD random forest). `explain: false` skips step two.
 */
function usePrediction(features: PatientFeatures | null, { explain = true, datasetRow = null }: { explain?: boolean; datasetRow?: number | null } = {}) {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [explaining, setExplaining] = useState(false);
  // True from the moment the inputs change until the result for exactly those inputs is complete.
  const [pending, setPending] = useState(false);
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);
  const lastExplained = useRef<PredictionResponse | null>(null);
  useEffect(() => {
    if (!features) { setResult(null); return; }
    setPending(true);
    const ctrl = new AbortController();
    const t = setTimeout(async () => {
      setLoading(true);
      try {
        if (apiMode === 'mock') {
          const r = await predict(features, { signal: ctrl.signal });
          lastExplained.current = r;
          setResult(r);
        } else {
          const fast = await predict(features, { signal: ctrl.signal, explain: false, datasetRow });
          if (ctrl.signal.aborted) return;
          setResult(explain ? withPreviousExplanation(fast, lastExplained.current) : fast);
          setUpdatedAt(new Date());
          setError(null);
          setLoading(false);
          if (!explain) { setPending(false); return; }
          setExplaining(true);
          const full = await predict(features, { signal: ctrl.signal, explain: true, datasetRow });
          if (ctrl.signal.aborted) return;
          lastExplained.current = full;
          setResult(full);
        }
        setUpdatedAt(new Date());
        setError(null);
        setPending(false);
      } catch (e) {
        if (!ctrl.signal.aborted) { setError((e as Error).message); setPending(false); }
      } finally {
        if (!ctrl.signal.aborted) { setLoading(false); setExplaining(false); }
      }
    }, apiMode === 'api' ? 250 : 0);
    return () => { clearTimeout(t); ctrl.abort(); };
  }, [features, explain, datasetRow]);
  return { result, error, loading, explaining, pending, updatedAt };
}

interface SavedCase {
  features: PatientFeatures; preset: string | null; caseId: string; caseBase: string; edited: boolean;
  overrides: Partial<PatientFeatures>;
}
// The last case is kept in this browser so a refresh doesn't lose work. Saved values are merged
// onto a complete default patient, so an older save can't leave a feature missing.
const saved = store.get<SavedCase | null>('cm.case', null);

export default function App() {
  const [features, setFeatures] = useState<PatientFeatures>(() =>
    saved ? withDerived({ ...NEW_CASE, ...saved.features }) : PRESETS[1].features);
  const [preset, setPreset] = useState<string | null>(saved ? saved.preset : PRESETS[1].id);
  const newCaseId = () => `AV-${String(Math.floor(1000 + Math.random() * 9000))}`;
  const [caseId, setCaseId] = useState(() => saved?.caseId?.replace(/^CM-/, 'AV-') ?? newCaseId());
  const [caseBase, setCaseBase] = useState(saved?.caseBase ?? `${PRESETS[1].label} (example case)`);
  const [edited, setEdited] = useState(saved?.edited ?? false);
  const caseLabel = edited ? `${caseBase}, edited` : caseBase;
  const [selected, setSelected] = useState<Target>('CAD');
  const [autoRotate, setAutoRotate] = useState(false);
  const [resetKey, setResetKey] = useState(0);
  const [fx, setFx] = useState<boolean>(() => store.get('cm.fx', true));

  // What-if scenario
  const [overrides, setOverrides] = useState<Partial<PatientFeatures>>(() => saved?.overrides ?? {});
  const [showSim, setShowSim] = useState(false);
  const simFeatures = useMemo(
    () => (Object.keys(overrides).length ? withDerived({ ...features, ...overrides } as PatientFeatures) : null),
    [features, overrides],
  );

  // Anatomical model
  const modelAvailable = useModelAvailable();
  const [anatomical, setAnatomical] = useState<boolean>(() => store.get('cm.anatomical', true));
  const [align, setAlign] = useState<Align>(() => store.get('cm.align', DEFAULT_ALIGN));
  const [aligning, setAligning] = useState(false);
  const [modelKind, setModelKind] = useState<'coronary' | 'generic' | null>(null);
  useEffect(() => {
    // A saved per-browser alignment wins; otherwise use the team file if there is one.
    if (store.get<Align | null>('cm.align', null)) return;
    fetchSharedAlign().then((a) => a && setAlign(a));
  }, []);

  // Exact training ranges from the API when available (falls back to approximate built-ins).
  const [ranges, setRanges] = useState<Record<string, [number, number]> | undefined>(undefined);
  useEffect(() => { fetchMetrics().then((m) => m?.feature_ranges && setRanges(m.feature_ranges)); }, []);

  useEffect(() => {
    store.set('cm.case', { features, preset, caseId, caseBase, edited, overrides } satisfies SavedCase);
  }, [features, preset, caseId, caseBase, edited, overrides]);

  // A real training patient, loaded with its angiography result (kept apart from the inputs).
  const [truth, setTruth] = useState<DatasetPatient | null>(null);
  const base = usePrediction(features, { datasetRow: truth?.row ?? null });
  const sim = usePrediction(simFeatures, { explain: false }); // the what-if panel only needs probabilities

  // Only a deliberate toggle is remembered; an automatic downgrade applies to this visit only.
  const toggleFx = () => setFx((f) => { store.set('cm.fx', !f); return !f; });
  useEffect(() => store.set('cm.anatomical', anatomical), [anatomical]);
  const updateAlign = (a: Align) => { setAlign(a); store.set('cm.align', a); };

  // A real training patient, with its angiography result kept apart from the inputs.
  const [datasetBusy, setDatasetBusy] = useState(false);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const onLoadDataset = async (row: number) => {
    setDatasetBusy(true);
    setDatasetError(null);
    try {
      const d = await fetchDatasetPatient(row);
      // Exact dataset values (no re-derivation), so the model sees the patient as recorded.
      setFeatures({ ...NEW_CASE, ...d.features });
      setTruth(d);
      setPreset(null);
      setOverrides({});
      setShowSim(false);
      setSelected('CAD');
      setCaseId(`DS-${String(row).padStart(3, '0')}`);
      setCaseBase(`Dataset patient, row ${row}`);
      setEdited(false);
    } catch (e) {
      setDatasetError((e as Error).message);
    } finally {
      setDatasetBusy(false);
    }
  };

  const onChange = (key: string, value: FeatureValue) => {
    setTruth(null); // an edited patient no longer has a known outcome
    setPreset(null);
    setEdited(true);
    setFeatures((f) => withDerived({ ...f, [key]: value }));
  };
  const onPreset = (id: string) => {
    const p = PRESETS.find((x) => x.id === id)!;
    setTruth(null);
    setPreset(id);
    setFeatures(p.features);
    setOverrides({});
    setShowSim(false);
    setCaseId(newCaseId());
    setCaseBase(`${p.label} (example case)`);
    setEdited(false);
  };
  const onNewCase = () => {
    setTruth(null);
    setPreset(null);
    setFeatures(NEW_CASE);
    setOverrides({});
    setShowSim(false);
    setSelected('CAD');
    setCaseId(newCaseId());
    setCaseBase('New case');
    setEdited(false);
  };

  const captureRef = useRef<(() => string | null) | null>(null);
  const [perfNotice, setPerfNotice] = useState(false);
  const onExportRef = useRef(() => {});
  const onNewCaseRef = useRef(() => {});
  const onPresentRef = useRef(() => {});
  // Export: snapshot the 3D view, then print the report (browsers offer "Save as PDF").
  const [report, setReport] = useState<{ image: string | null; at: Date } | null>(null);
  const [openedAt] = useState(() => new Date()); // date for a report printed with Ctrl+P before any export
  // The report includes explanations, so wait for the current one before printing.
  const [exportPending, setExportPending] = useState(false);
  const doExport = () => {
    const image = captureRef.current?.() ?? null;
    setReport({ image, at: new Date() });
  };
  const onExport = () => {
    if (base.pending || base.explaining || base.result?.explained === false) setExportPending(true);
    else doExport();
  };
  useEffect(() => {
    if (exportPending && !base.pending && !base.explaining && base.result?.explained !== false) {
      setExportPending(false);
      doExport();
    }
  });
  useEffect(() => {
    if (!report) return;
    const t = setTimeout(() => window.print(), 200);
    return () => clearTimeout(t);
  }, [report]);
  useEffect(() => { onExportRef.current = onExport; onNewCaseRef.current = onNewCase; });
  const onOverride = useCallback((key: string, value: FeatureValue | undefined) => {
    setOverrides((o) => {
      const next = { ...o };
      if (value === undefined) delete next[key]; else next[key] = value;
      return next;
    });
  }, []);
  // Show the scenario on the heart as soon as the first lever moves; hide it when all are reset.
  const hadOverrides = useRef(false);
  useEffect(() => {
    const has = Object.keys(overrides).length > 0;
    if (has && !hadOverrides.current) setShowSim(true);
    if (!has) setShowSim(false);
    hadOverrides.current = has;
  }, [overrides]);


  const useSim = showSim && !!sim.result;
  const shown = useSim ? sim.result : base.result;
  const probs = useMemo(() => shown && {
    LAD: shown.predictions.LAD.probability,
    LCX: shown.predictions.LCX.probability,
    RCA: shown.predictions.RCA.probability,
  }, [shown]);
  const useAnatomical = modelAvailable === true && anatomical;
  const checkingModel = modelAvailable === null;
  // Everything in the viewer + HUD follows the simulated patient while the simulation is shown.
  const shownFeatures = useSim && simFeatures ? simFeatures : features;
  const baseProbs = useMemo(() => base.result && {
    LAD: base.result.predictions.LAD.probability,
    LCX: base.result.predictions.LCX.probability,
    RCA: base.result.predictions.RCA.probability,
  }, [base.result]);

  // Guided demo (presenter mode). Each step's action runs from the click/key that opens it.
  const [tour, setTour] = useState<number | null>(null);
  const tourSteps: TourStep[] = (() => {
    const a = () => ({ onPreset, onLoadDataset, setSelected, setOverrides });
    // Each step sets up its own screen, so Back/Next work in any order.
    const typical = (overrides: Partial<PatientFeatures> = {}) => {
      if (preset !== 'typical') onPreset('typical');
      setOverrides(overrides);
    };
    const all: (TourStep & { apiOnly?: boolean })[] = [
      {
        title: 'A patient in the clinic',
        text: 'A 61-year-old smoker with hypertension and typical chest pain. Four independent models, trained on 303 patients who had coronary angiography, score overall CAD and each main artery. The heart is coloured by those scores.',
        focus: '.viewer',
        run: () => { typical(); a().setSelected('CAD'); },
      },
      {
        title: 'Why the model says so',
        text: "Each bar is an exact SHAP value: how much one input pushed this patient's score up or down. The measurements table shows each physiological value with its share. These are associations learned from the data, not causes.",
        focus: '.contrib',
        run: () => { typical(); a().setSelected('CAD'); },
      },
      {
        title: 'Can we trust the number?',
        text: 'The reality check takes the 40 patients in our data whose held-out score was closest to this one and counts how many truly had disease on angiography. A score becomes an observed frequency, with its uncertainty.',
        focus: '.reality',
        apiOnly: true,
        run: () => { typical(); a().setSelected('LCX'); },
      },
      {
        title: 'Built for screening',
        text: 'At a 50% cut-off the LCX and RCA models missed about half of narrowed arteries. We flag them from 30%: they now catch 84% and 83%, at the cost of more false alarms. The tick on each bar marks its cut-off.',
        focus: '.vessels',
        apiOnly: true,
        run: () => { typical(); a().setSelected('LCX'); },
      },
      {
        title: 'What if?',
        text: 'Stop smoking, lower LDL and blood pressure: the models re-run instantly and the heart shows the scenario. This shows how the models respond, not the effect of a treatment.',
        focus: '.whatif',
        run: () => { typical({ 'Current Smoker': 0, LDL: 100, BP: 125 }); a().setSelected('CAD'); },
      },
      {
        title: 'A real patient',
        text: "Row 32: a real patient with narrowing in all three arteries, and the true angiography result. 'Cross-val.' is what each model predicted while this patient was held out: the fair test. All four are right, and the LCX (40%) and RCA (44%) would have been missed at the old 50% cut-off.",
        focus: '.truth',
        apiOnly: true,
        run: () => a().onLoadDataset(32),
      },
      {
        title: 'Honest about errors',
        text: "Row 93 is recorded as 'no CAD' but with a narrowed LAD, which contradicts the dataset's own definition. AtheroVision flags the label conflict instead of hiding the miss.",
        focus: '.truth',
        apiOnly: true,
        run: () => a().onLoadDataset(93),
      },
      {
        title: 'Limits we state upfront',
        text: 'Cross-validated AUC: CAD 0.92, LAD 0.84, LCX and RCA 0.74, with the calibration charts showing predicted against observed risk. 303 patients from one centre, no external validation. AtheroVision is educational decision support, not a diagnosis.',
        focus: '.metrics',
        run: () => a().setSelected('CAD'),
      },
    ];
    return all.filter((s) => apiMode === 'api' || !s.apiOnly);
  })();
  const goToStep = (i: number | null) => {
    setTour(i);
    if (i !== null) void tourSteps[i].run();
  };
  useEffect(() => { onPresentRef.current = () => { if (tour === null) goToStep(0); }; });

  // Keyboard shortcuts (ignored while typing in a field).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (e.ctrlKey || e.metaKey || e.altKey || el.closest('input, select, textarea')) return;
      if (e.key.toLowerCase() === 'p') { onPresentRef.current(); return; }
      const n = Number(e.key);
      if (n >= 1 && n <= 4) { setSelected(TARGETS[n - 1]); return; }
      const k = e.key.toLowerCase();
      if (k === 'r') { setResetKey((x) => x + 1); setSelected('CAD'); }
      else if (k === 'g') toggleFx();
      else if (k === 'e') onExportRef.current();
      else if (k === 'n') onNewCaseRef.current();
      else if (e.key === '?') (document.querySelector('.pop--keys .appbar__btn') as HTMLButtonElement | null)?.click();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return (
    <>
    <div className={`app${tour !== null ? ' app--tour' : ''}`}>

      <div className="disclaimer" role="alert">
        <span>{DISCLAIMER}</span>
        {apiMode === 'mock' && <strong className="disclaimer__demo">{DEMO_NOTE}</strong>}
      </div>

      <AppHeader
        mode={apiMode} result={base.result} updatedAt={base.updatedAt} loading={base.loading || sim.loading}
        caseId={caseId} caseLabel={caseLabel} features={shownFeatures} ranges={ranges} simulated={useSim} error={base.error}
        onNewCase={onNewCase} onExport={onExport} exportPending={exportPending}
        onPresent={() => goToStep(0)}
      />

      {base.error && <div className="error" role="alert">{base.error}{base.result ? ' Showing the last successful result.' : ''}</div>}

      <main className="layout">
        <PatientForm
          features={features} activePreset={preset} onChange={onChange} onPreset={onPreset} unusedInputs={base.result?.unused_inputs}
          dataset={apiMode === 'api' ? { onLoad: onLoadDataset, row: truth?.row ?? null, busy: datasetBusy, error: datasetError, nRows: truth?.n_rows ?? 303 } : undefined}
        />

        <div className="center">
          <section
            className={`panel viewer${useSim ? ' viewer--sim' : ''}`}
            aria-label="3D heart"
            aria-describedby="viewer-desc"
          >
            <p id="viewer-desc" className="sr-only">
              {probs
                ? `Interactive 3D heart. Coronary arteries coloured by predicted stenosis: LAD ${Math.round(probs.LAD * 100)}%, LCX ${Math.round(probs.LCX * 100)}%, RCA ${Math.round(probs.RCA * 100)}%. Use keys 1 to 4 or the vessel buttons to select a vessel.`
                : 'Interactive 3D heart. Waiting for a prediction.'}
            </p>
            {checkingModel ? (
              <div className="scene__loading" role="status"><span>Loading 3D view</span><div className="scene__progress scene__progress--busy"><div /></div></div>
            ) : (
            <Suspense fallback={<div className="scene__loading" role="status"><span>Loading 3D view</span><div className="scene__progress scene__progress--busy"><div /></div></div>}>
            <HeartScene
              probs={probs} selected={selected} onSelect={setSelected} autoRotate={autoRotate} resetKey={resetKey}
              pulseRate={Number(shownFeatures.PR)} fx={fx} anatomical={useAnatomical} align={align} simulated={useSim}
              baseProbs={useSim ? baseProbs : null} onModelKind={setModelKind}
              captureRef={captureRef}
              onLowPerformance={() => { if (fx) { setFx(false); setPerfNotice(true); } }}
            />
            </Suspense>
            )}
            {perfNotice && (
              <div className="notice" role="status">
                Glow was turned off to keep the 3D view smooth on this device.
                <button className="notice__x" onClick={() => setPerfNotice(false)} aria-label="Dismiss">×</button>
              </div>
            )}
            <div className="viewer__hud">
              <span className="muted">Drag to rotate, scroll to zoom, click a vessel or region</span>
              <div className="viewer__buttons">
                {modelAvailable === true && (
                  <button className="chip" onClick={() => setAnatomical((a) => !a)}>
                    {anatomical ? 'Anatomical' : 'Procedural'}
                  </button>
                )}
                {useAnatomical && modelKind === 'generic' && <button className={`chip${aligning ? ' chip--on' : ''}`} onClick={() => setAligning((a) => !a)}>Align</button>}
                <button className={`chip${fx ? ' chip--on' : ''}`} onClick={toggleFx} title="Glow effects. Turn off on slow laptops.">Glow</button>
                <button className="chip" onClick={() => setAutoRotate((a) => !a)}>{autoRotate ? 'Stop rotating' : 'Rotate'}</button>
                <button className="chip" onClick={() => { setResetKey((k) => k + 1); setSelected('CAD'); }}>Reset</button>
              </div>
            </div>
            {useSim && <div className="sim-banner">Showing the what-if scenario</div>}
            {aligning && useAnatomical && modelKind === 'generic' && <AlignPanel align={align} onChange={updateAlign} onClose={() => setAligning(false)} />}
            {useAnatomical && modelKind === 'coronary' && (
              <a className="credit" href="https://dbarchive.biosciencedbc.jp/en/bodyparts3d/" target="_blank" rel="noreferrer">
                Anatomy: BodyParts3D, © DBCLS, CC BY-SA 2.1 JP
              </a>
            )}
            <div className="scale" aria-hidden>
              <span>0%</span><div className="scale__bar" /><span>100%</span>
              <span className="muted">stenosis probability</span>
            </div>
          </section>

          <section className={`panel hud${useSim ? ' hud--sim' : ''}`} aria-label="Monitor">
            <EcgMonitor features={shownFeatures} simulated={useSim} />
            <Vitals features={shownFeatures} compare={useSim ? features : undefined} />
            <Alerts features={shownFeatures} ranges={ranges} />
          </section>

          <div className="center__grid">
            <ContributionChart result={base.result} target={selected} onTarget={setSelected} explaining={base.explaining} />
            <MeasurementsTable features={features} result={base.result} target={selected} />
          </div>
        </div>

        <aside className="side">
          <RiskSummary result={base.result} selected={selected} onSelect={setSelected} truth={truth} />
          <WhatIf
            features={features} overrides={overrides} onOverride={onOverride} onReset={() => setOverrides({})}
            base={base.result} sim={sim.result} showOnHeart={showSim} onShowOnHeart={setShowSim}
            scenario={simFeatures} ranges={ranges}
          />
          <MetricsCard />
        </aside>
      </main>
    </div>
    {tour !== null && <DemoTour steps={tourSteps} index={tour} onIndex={goToStep} onExit={() => setTour(null)} />}
    <PrintReport
      features={features} result={base.result} caseId={caseId} caseLabel={caseLabel}
      image={report?.image ?? null} generatedAt={report?.at ?? base.updatedAt ?? openedAt} ranges={ranges}
    />
    </>
  );
}

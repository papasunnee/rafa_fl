import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

// Default AgentConfig thresholds (config.py's AgentConfig, sent to POST
// /decide as the `config` field -- see ui/backend/routes/playground.py).
// theta_delta/theta_d were tuned by experiments/pilot.py; the rest are
// analytic defaults. What each one gates:
const DEFAULT_CONFIG = {
  theta_delta: 0.013148, // min loss improvement (delta_t) that counts as "still learning" (rule P3)
  theta_d: 0.137177,     // KL-divergence (drift_t) above which drift is considered detected (rule P2)
  theta_rho: 0.20,       // resource sample (rho_t) below which a client is considered resource-constrained (rule P4)
  s_max: 5,              // max consecutive rounds a client may be skipped before it's force-included (rule P1)
  e_min: 1,              // fewest local training epochs the adaptive epoch function can choose
  e_max: 5,              // most local training epochs the adaptive epoch function can choose
  e_std: 3,              // fixed epoch count used by non-agent baselines and the "fixed epochs" ablation
  lam: 0.6,              // weight on drift vs. quality signal in the adaptive epoch function (0=quality only, 1=drift only)
}

// The five priority-ordered rules the agent's decide() checks in order --
// the first one whose condition is met determines the action. See
// agent/reactive_agent.py.
const RULES = ['P1_force', 'P2_drift', 'P3_quality', 'P4_defer', 'P5_default']
const RULE_LABELS = {
  P1_force: 'P1 — bounded participation (forced)',
  P2_drift: 'P2 — drift-triggered',
  P3_quality: 'P3 — quality-triggered',
  P4_defer: 'P4 — resource-aware deferral',
  P5_default: 'P5 — default participation',
}

function Slider({ label, value, onChange, min, max, step }) {
  return (
    <label className="block">
      <div className="mb-1 flex justify-between text-sm">
        <span className="text-neutral-500">{label}</span>
        <span className="font-mono">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
    </label>
  )
}

function NumberField({ label, value, onChange, step }) {
  return (
    <label className="block text-sm">
      <span className="mb-1 block text-neutral-500">{label}</span>
      <input
        type="number"
        step={step ?? 'any'}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full rounded-md border border-neutral-300 bg-white px-2 py-1 dark:border-neutral-700 dark:bg-neutral-900"
      />
    </label>
  )
}

export default function Playground() {
  const [deltaT, setDeltaT] = useState(0.02)
  const [driftT, setDriftT] = useState(0.05)
  const [rhoT, setRhoT] = useState(0.5)
  const [config, setConfig] = useState(DEFAULT_CONFIG)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    const handle = setTimeout(() => {
      fetch('/decide', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ delta_t: deltaT, drift_t: driftT, rho_t: rhoT, config }),
      })
        .then((res) => {
          if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
          return res.json()
        })
        .then((data) => { setResult(data); setError(null) })
        .catch((err) => setError(err.message))
    }, 150) // debounce continuous slider drags
    return () => clearTimeout(handle)
  }, [deltaT, driftT, rhoT, config])

  const setConfigField = (key) => (val) => setConfig((c) => ({ ...c, [key]: val }))

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold">Agent Playground</h1>
        <p className="mt-1 text-sm text-neutral-500">
          Calls the real <code className="rounded bg-neutral-100 px-1 dark:bg-neutral-800">ReactiveAgent.decide()</code> on
          every change below -- this is not a reimplementation of the priority logic in JavaScript.
          Unlike the <Link to="/" className="underline">Dashboard</Link> (past results) or the{' '}
          <Link to="/runner" className="underline">Runner</Link> (real training), this page doesn't need any
          results data or a training run -- it's just the agent's one-shot decision rule in isolation, for
          building intuition about when each of the 5 rules below fires.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
        <section className="space-y-4">
          <h2 className="text-lg font-medium">Observation (per-round signals)</h2>
          <p className="text-xs text-neutral-400">
            The three numbers a real client would compute each round and hand to the agent.
          </p>
          <Slider label="delta_t (loss improvement)" value={deltaT} onChange={setDeltaT} min={-0.2} max={1} step={0.005} />
          <p className="-mt-3 text-xs text-neutral-400">Drop in local validation loss vs. last round this client participated. Higher = learning faster.</p>
          <Slider label="drift_t (KL divergence)" value={driftT} onChange={setDriftT} min={0} max={1} step={0.005} />
          <p className="-mt-3 text-xs text-neutral-400">KL divergence between this client's current label distribution and its round-0 reference. Higher = more concept drift.</p>
          <Slider label="rho_t (resource sample)" value={rhoT} onChange={setRhoT} min={0} max={1} step={0.01} />
          <p className="-mt-3 text-xs text-neutral-400">Simulated resource availability (0 = fully constrained, 1 = fully available).</p>
        </section>

        <section className="space-y-4">
          <h2 className="text-lg font-medium">AgentConfig thresholds</h2>
          <div className="grid grid-cols-2 gap-3">
            <NumberField label="theta_delta" value={config.theta_delta} onChange={setConfigField('theta_delta')} step={0.001} />
            <NumberField label="theta_d" value={config.theta_d} onChange={setConfigField('theta_d')} step={0.001} />
            <NumberField label="theta_rho" value={config.theta_rho} onChange={setConfigField('theta_rho')} step={0.01} />
            <NumberField label="s_max" value={config.s_max} onChange={setConfigField('s_max')} step={1} />
            <NumberField label="e_min" value={config.e_min} onChange={setConfigField('e_min')} step={1} />
            <NumberField label="e_max" value={config.e_max} onChange={setConfigField('e_max')} step={1} />
            <NumberField label="e_std" value={config.e_std} onChange={setConfigField('e_std')} step={1} />
            <NumberField label="lam" value={config.lam} onChange={setConfigField('lam')} step={0.05} />
          </div>
        </section>
      </div>

      <section>
        <h2 className="mb-2 text-lg font-medium">Decision</h2>
        <p className="mb-2 text-xs text-neutral-400">
          participate = whether this client trains this round. epochs = how many local epochs, if it does
          (0 if deferring). communicate = whether it sends its updated model back to the server. rule = which
          of the 5 priority rules below decided this, evaluated top to bottom -- the first match wins.
        </p>
        {error && (
          <div className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
            Backend error: {error}. Is <code>ui/backend</code> running on :8000?
          </div>
        )}
        {!error && !result && <p className="text-sm text-neutral-500">Waiting for first response…</p>}
        {result && (
          <div className="rounded-lg border border-neutral-200 p-4 dark:border-neutral-800">
            <div className="mb-3 grid grid-cols-3 gap-4 text-sm">
              <div>
                <div className="text-neutral-500">participate</div>
                <div className="text-lg font-semibold">{result.participate ? 'yes' : 'no'}</div>
              </div>
              <div>
                <div className="text-neutral-500">epochs</div>
                <div className="text-lg font-semibold">{result.epochs}</div>
              </div>
              <div>
                <div className="text-neutral-500">communicate</div>
                <div className="text-lg font-semibold">{result.communicate ? 'yes' : 'no'}</div>
              </div>
            </div>
            <div className="space-y-1">
              {RULES.map((rule) => (
                <div
                  key={rule}
                  className={`rounded-md px-3 py-2 text-sm ${
                    result.rule === rule
                      ? 'bg-red-600 font-medium text-white'
                      : 'bg-neutral-50 text-neutral-400 dark:bg-neutral-900'
                  }`}
                >
                  {RULE_LABELS[rule]}
                </div>
              ))}
            </div>
          </div>
        )}
      </section>
    </div>
  )
}

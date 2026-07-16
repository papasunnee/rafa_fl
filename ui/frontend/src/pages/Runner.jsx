import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'

// Mirrors experiments/run_experiment.py's METHODS list and ABLATIONS dict --
// update both places together if either changes.
const METHODS = ['rafa_fl', 'fedavg', 'fedprox', 'scaffold', 'fedcs', 'oort']
const ABLATIONS = ['(none)', 'rafa_full', 'rafa_no_drift', 'rafa_no_resource', 'rafa_fixed_epochs', 'rafa_equal_weight']

export default function Runner() {
  const [method, setMethod] = useState('rafa_fl')
  const [ablation, setAblation] = useState('(none)')
  const [alpha, setAlpha] = useState(0.5)
  const [numClients, setNumClients] = useState(50)
  const [seed, setSeed] = useState(42)
  const [drift, setDrift] = useState(true)

  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState(null)
  const [output, setOutput] = useState('')
  const [submitError, setSubmitError] = useState(null)
  const pollRef = useRef(null)

  useEffect(() => () => clearInterval(pollRef.current), [])

  const submit = async (e) => {
    e.preventDefault()
    setSubmitError(null)
    setOutput('')
    setStatus(null)
    try {
      const res = await fetch('/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          method,
          ablation: ablation === '(none)' ? null : ablation,
          alpha,
          num_clients: numClients,
          seed,
          drift,
        }),
      })
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      const data = await res.json()
      setJobId(data.job_id)
      setStatus('running')

      clearInterval(pollRef.current)
      pollRef.current = setInterval(async () => {
        const pollRes = await fetch(`/run/${data.job_id}`)
        const pollData = await pollRes.json()
        setStatus(pollData.status)
        setOutput(pollData.output ?? '')
        if (pollData.status !== 'running') clearInterval(pollRef.current)
      }, 1500)
    } catch (err) {
      setSubmitError(err.message)
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold">Runner</h1>
        <p className="mt-1 text-sm text-neutral-500">
          Local dev only. Launches a real <code className="rounded bg-neutral-100 px-1 dark:bg-neutral-800">
            python -m experiments.run_experiment
          </code> subprocess -- a single run can take several minutes. It writes a new file into{' '}
          <code className="rounded bg-neutral-100 px-1 dark:bg-neutral-800">results/</code>, but that won't
          appear on the <Link to="/" className="underline">Dashboard</Link> until you re-run{' '}
          <code className="rounded bg-neutral-100 px-1 dark:bg-neutral-800">python ui/scripts/export_results.py</code>.
        </p>
      </div>

      <form onSubmit={submit} className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3">
        <label className="text-sm">
          <span className="mb-1 block text-neutral-500" title="Which FL algorithm to run">method</span>
          <select
            className="w-full rounded-md border border-neutral-300 bg-white px-2 py-1 dark:border-neutral-700 dark:bg-neutral-900"
            value={method}
            onChange={(e) => setMethod(e.target.value)}
          >
            {METHODS.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </label>
        <label className="text-sm">
          <span className="mb-1 block text-neutral-500" title="Optional RAFA-FL variant with one mechanism toggled off, for isolating its effect. Only applies when method=rafa_fl.">ablation</span>
          <select
            className="w-full rounded-md border border-neutral-300 bg-white px-2 py-1 dark:border-neutral-700 dark:bg-neutral-900"
            value={ablation}
            onChange={(e) => setAblation(e.target.value)}
          >
            {ABLATIONS.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
        </label>
        <label className="text-sm">
          <span className="mb-1 block text-neutral-500" title="Dirichlet non-IID heterogeneity across clients -- lower means each client's data is more skewed toward a few classes">alpha</span>
          <input
            type="number" step="0.1" value={alpha} onChange={(e) => setAlpha(Number(e.target.value))}
            className="w-full rounded-md border border-neutral-300 bg-white px-2 py-1 dark:border-neutral-700 dark:bg-neutral-900"
          />
        </label>
        <label className="text-sm">
          <span className="mb-1 block text-neutral-500" title="Number of simulated clients in this run">num_clients</span>
          <input
            type="number" step="1" value={numClients} onChange={(e) => setNumClients(Number(e.target.value))}
            className="w-full rounded-md border border-neutral-300 bg-white px-2 py-1 dark:border-neutral-700 dark:bg-neutral-900"
          />
        </label>
        <label className="text-sm">
          <span className="mb-1 block text-neutral-500" title="Random seed controlling client data partitioning and training randomness -- same seed + same settings should reproduce closely, though Flower's simulation isn't perfectly deterministic across worker processes">seed</span>
          <input
            type="number" step="1" value={seed} onChange={(e) => setSeed(Number(e.target.value))}
            className="w-full rounded-md border border-neutral-300 bg-white px-2 py-1 dark:border-neutral-700 dark:bg-neutral-900"
          />
        </label>
        <label className="flex items-end gap-2 text-sm">
          <input type="checkbox" checked={drift} onChange={(e) => setDrift(e.target.checked)} />
          <span className="text-neutral-500" title="Whether to inject concept drift (a label-mapping shift on a subset of clients) partway through the run">drift</span>
        </label>

        <div className="sm:col-span-2 md:col-span-3">
          <button
            type="submit"
            disabled={status === 'running'}
            className="rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:bg-neutral-100 dark:text-neutral-900"
          >
            {status === 'running' ? 'Running…' : 'Run experiment'}
          </button>
        </div>
      </form>

      {submitError && (
        <div className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
          {submitError}
        </div>
      )}

      {jobId && (
        <section>
          <h2 className="mb-2 text-lg font-medium">
            Job {jobId} — <span className="font-mono">{status}</span>
          </h2>
          <pre className="max-h-96 overflow-auto rounded-lg bg-neutral-950 p-4 text-xs text-neutral-100">
            {output || '(no output yet)'}
          </pre>
        </section>
      )}
    </div>
  )
}

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'

// Mirrors experiments/run_experiment.py's METHODS list and ABLATIONS dict --
// update both places together if either changes.
const METHODS = ['rafa_fl', 'fedavg', 'fedprox', 'scaffold', 'fedcs', 'oort']
const ABLATIONS = ['(none)', 'rafa_full', 'rafa_no_drift', 'rafa_no_resource', 'rafa_fixed_epochs', 'rafa_equal_weight']

// Colors for the live client grid. "rule" (P1-P5) only exists for the RAFA-FL
// agent path -- non-agent baselines report deferred=false for every sampled
// client, so they only ever show as "participated" or "not sampled".
const RULE_COLORS = {
  P1_force: '#7c3aed',
  P2_drift: '#dc2626',
  P3_quality: '#2563eb',
  P4_defer: '#d97706',
  P5_default: '#059669',
}
const NOT_SAMPLED_COLOR = '#e5e5e5'
const PARTICIPATED_COLOR = '#059669' // baselines: no rule, just "it trained"
const RETURN_LINE_COLOR = '#059669'
const BROADCAST_ONLY_LINE_COLOR = '#9ca3af'

function clientColor(clientEvent) {
  if (!clientEvent) return NOT_SAMPLED_COLOR
  if (clientEvent.deferred) return RULE_COLORS.P4_defer
  return RULE_COLORS[clientEvent.rule] ?? PARTICIPATED_COLOR
}

function clientTitle(cid, clientEvent) {
  if (!clientEvent) return `client ${cid}: not sampled this round`
  if (clientEvent.deferred) return `client ${cid}: received the global model, deferred (P4_defer) -- no update sent back`
  return `client ${cid}: received the global model, trained, and sent an update back${clientEvent.rule ? ` (${clientEvent.rule})` : ''}`
}

function ClientGrid({ numClients, event, registerTileRef }) {
  const byCid = useMemo(() => {
    const m = new Map()
    for (const c of event?.clients ?? []) m.set(c.cid, c)
    return m
  }, [event])

  return (
    <div className="flex flex-wrap gap-1">
      {Array.from({ length: numClients }, (_, cid) => {
        const clientEvent = byCid.get(cid)
        return (
          <div
            key={cid}
            ref={(el) => registerTileRef(cid, el)}
            title={clientTitle(cid, clientEvent)}
            className="flex h-7 w-7 items-center justify-center rounded text-[10px] font-mono text-white"
            style={{ backgroundColor: clientColor(clientEvent) }}
          >
            {cid}
          </div>
        )
      })}
    </div>
  )
}

function Legend() {
  const items = [
    ['not sampled this round', NOT_SAMPLED_COLOR],
    ['P1 forced', RULE_COLORS.P1_force],
    ['P2 drift', RULE_COLORS.P2_drift],
    ['P3 quality', RULE_COLORS.P3_quality],
    ['P4 deferred', RULE_COLORS.P4_defer],
    ['P5 default / baseline participated', RULE_COLORS.P5_default],
  ]
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-neutral-500">
      {items.map(([label, color]) => (
        <span key={label} className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded" style={{ backgroundColor: color }} />
          {label}
        </span>
      ))}
      <span className="flex items-center gap-1.5">
        <span className="inline-block h-0.5 w-4" style={{ backgroundColor: RETURN_LINE_COLOR }} />
        sent an update back
      </span>
      <span className="flex items-center gap-1.5">
        <svg width="16" height="2"><line x1="0" y1="1" x2="16" y2="1" stroke={BROADCAST_ONLY_LINE_COLOR} strokeDasharray="3,2" /></svg>
        received broadcast only
      </span>
    </div>
  )
}

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
  const [events, setEvents] = useState([])
  const [submittedNumClients, setSubmittedNumClients] = useState(numClients)
  const [viewIdx, setViewIdx] = useState(0)
  const [followLatest, setFollowLatest] = useState(true)
  const [submitError, setSubmitError] = useState(null)
  const pollRef = useRef(null)

  // Client-server interaction diagram: a "Server" node with lines fanning
  // out to whichever clients were sampled the currently-viewed round. Only
  // sampled clients need a line (the rest are just gray tiles below), so
  // this stays cheap even at num_clients=100.
  const diagramRef = useRef(null)
  const serverRef = useRef(null)
  const tileRefs = useRef(new Map())
  const [lines, setLines] = useState([])

  useEffect(() => () => clearInterval(pollRef.current), [])

  // Auto-advance to the latest round as new events stream in, unless the
  // user has manually dragged the scrubber back to look at an earlier round.
  useEffect(() => {
    if (followLatest && events.length > 0) setViewIdx(events.length - 1)
  }, [events, followLatest])

  const currentEvent = events[viewIdx]

  const registerTileRef = (cid, el) => {
    if (el) tileRefs.current.set(cid, el)
    else tileRefs.current.delete(cid)
  }

  // Recompute the server->client lines whenever the viewed round changes or
  // the window resizes (tiles wrap responsively, so their positions shift).
  useLayoutEffect(() => {
    function recompute() {
      const container = diagramRef.current
      const serverEl = serverRef.current
      if (!container || !serverEl || !currentEvent) { setLines([]); return }
      const containerRect = container.getBoundingClientRect()
      const serverRect = serverEl.getBoundingClientRect()
      const sx = serverRect.left + serverRect.width / 2 - containerRect.left
      const sy = serverRect.bottom - containerRect.top
      const next = []
      for (const c of currentEvent.clients) {
        const tileEl = tileRefs.current.get(c.cid)
        if (!tileEl) continue
        const tileRect = tileEl.getBoundingClientRect()
        const tx = tileRect.left + tileRect.width / 2 - containerRect.left
        const ty = tileRect.top - containerRect.top
        next.push({ cid: c.cid, x1: sx, y1: sy, x2: tx, y2: ty, returned: !c.deferred })
      }
      setLines(next)
    }
    recompute()
    window.addEventListener('resize', recompute)
    return () => window.removeEventListener('resize', recompute)
  }, [currentEvent, submittedNumClients])

  const sampledCount = currentEvent?.clients.length ?? 0
  const returnedCount = currentEvent?.clients.filter((c) => !c.deferred).length ?? 0
  const cumulativeTransmissions = useMemo(
    () => events.slice(0, viewIdx + 1).reduce(
      (sum, e) => sum + e.clients.filter((c) => !c.deferred).length, 0,
    ),
    [events, viewIdx],
  )

  const submit = async (e) => {
    e.preventDefault()
    setSubmitError(null)
    setOutput('')
    setStatus(null)
    setEvents([])
    setViewIdx(0)
    setFollowLatest(true)
    setSubmittedNumClients(numClients)
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
        setEvents(pollData.events ?? [])
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
        <>
          <section>
            <h2 className="mb-2 text-lg font-medium">
              Job {jobId} — <span className="font-mono">{status}</span>
            </h2>

            {events.length === 0 ? (
              <p className="text-sm text-neutral-500">
                Waiting for the first round to finish (each client tile below only appears once its round's
                results come back)…
              </p>
            ) : (
              <div className="space-y-3 rounded-lg border border-neutral-200 p-4 dark:border-neutral-800">
                <div className="flex items-center gap-3">
                  <span className="text-sm font-medium">
                    Round {currentEvent?.round} of {events[events.length - 1]?.round}
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={events.length - 1}
                    value={viewIdx}
                    onChange={(e) => {
                      setFollowLatest(false)
                      setViewIdx(Number(e.target.value))
                    }}
                    className="flex-1"
                  />
                  <button
                    type="button"
                    onClick={() => setFollowLatest(true)}
                    disabled={followLatest}
                    className="rounded-md border border-neutral-300 px-2 py-1 text-xs disabled:opacity-40 dark:border-neutral-700"
                  >
                    Jump to latest
                  </button>
                </div>

                {/* Client-server interaction diagram */}
                <div ref={diagramRef} className="relative">
                  <div className="mb-6 flex justify-center">
                    <div
                      ref={serverRef}
                      className="rounded-lg border-2 border-neutral-800 bg-neutral-800 px-4 py-2 text-center text-sm font-medium text-white dark:border-neutral-200 dark:bg-neutral-200 dark:text-neutral-900"
                    >
                      Server
                      <div className="text-xs font-normal opacity-80">
                        round {currentEvent?.round}: broadcast to {sampledCount} sampled, {returnedCount} returned an update
                      </div>
                      <div className="text-xs font-normal opacity-80">
                        total updates aggregated so far: {cumulativeTransmissions}
                      </div>
                    </div>
                  </div>
                  <svg className="pointer-events-none absolute inset-0 h-full w-full">
                    {lines.map((l) => (
                      <line
                        key={l.cid}
                        x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2}
                        stroke={l.returned ? RETURN_LINE_COLOR : BROADCAST_ONLY_LINE_COLOR}
                        strokeWidth={l.returned ? 1.5 : 1}
                        strokeDasharray={l.returned ? undefined : '4,3'}
                      />
                    ))}
                  </svg>
                  <ClientGrid numClients={submittedNumClients} event={currentEvent} registerTileRef={registerTileRef} />
                </div>

                <Legend />
                <p className="text-xs text-neutral-400">
                  Only ~{Math.round(100 * sampledCount / submittedNumClients) || 0}% of clients are sampled
                  each round (fraction_fit=0.10) -- everyone else shown gray simply wasn't picked this round,
                  they haven't necessarily deferred anything. Every sampled client receives the broadcast
                  model regardless of what it decides; only clients that don't defer send anything back.
                </p>
              </div>
            )}
          </section>

          <section>
            <h2 className="mb-2 text-lg font-medium">Raw log</h2>
            <pre className="max-h-96 overflow-auto rounded-lg bg-neutral-950 p-4 text-xs text-neutral-100">
              {output || '(no output yet)'}
            </pre>
          </section>
        </>
      )}
    </div>
  )
}

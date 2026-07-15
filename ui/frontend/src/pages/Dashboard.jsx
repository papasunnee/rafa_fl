import { useMemo, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import resultsData from '../data/results_export.json'

// Fixed per-method colors so charts stay readable/consistent across pages.
const METHOD_COLORS = {
  rafa_fl: '#dc2626',
  fedavg: '#2563eb',
  fedprox: '#7c3aed',
  scaffold: '#059669',
  fedcs: '#d97706',
  oort: '#0891b2',
}
const colorFor = (method) => METHOD_COLORS[method] ?? '#6b7280'

function fmtPair(pair, decimals = 4) {
  if (!pair || pair[0] == null) return 'n/a'
  return `${pair[0].toFixed(decimals)} ± ${pair[1].toFixed(decimals)}`
}

function unique(arr) {
  return [...new Set(arr)].sort((a, b) => a - b)
}

export default function Dashboard() {
  const { n_runs: nRuns, runs, summary } = resultsData

  const alphas = useMemo(() => unique(summary.map((s) => s.alpha)), [summary])
  const clientCounts = useMemo(() => unique(summary.map((s) => s.num_clients)), [summary])

  const [alpha, setAlpha] = useState(alphas[0])
  const [numClients, setNumClients] = useState(clientCounts[0])
  const [drift, setDrift] = useState(true)

  if (!nRuns) {
    return (
      <div className="rounded-lg border border-dashed border-neutral-300 p-8 text-center text-neutral-500 dark:border-neutral-700">
        <p className="font-medium">No results exported yet.</p>
        <p className="mt-2 text-sm">
          Run <code className="rounded bg-neutral-100 px-1.5 py-0.5 dark:bg-neutral-800">
            python ui/scripts/export_results.py
          </code> from the repo root, then reload this page.
        </p>
      </div>
    )
  }

  const conditionRows = summary
    .filter((s) => s.alpha === alpha && s.num_clients === numClients && s.drift === drift)
    .sort((a, b) => a.method.localeCompare(b.method))

  const matchingRuns = runs.filter(
    (r) => r.alpha === alpha && r.num_clients === numClients && r.drift === drift,
  )
  const methodsPresent = [...new Set(matchingRuns.map((r) => r.method))].sort()

  // Per-round curves: average each method's per-seed curves index-wise. Real
  // per-round data (accuracy_curve, round_log) -- nothing here is interpolated.
  const numRounds = matchingRuns[0]?.accuracy_curve.length ?? 0
  const accuracyChartData = Array.from({ length: numRounds }, (_, i) => {
    const point = { round: i + 1 }
    for (const method of methodsPresent) {
      const curves = matchingRuns.filter((r) => r.method === method).map((r) => r.accuracy_curve)
      point[method] = curves.reduce((sum, c) => sum + c[i], 0) / curves.length
    }
    return point
  })

  const transmissionsChartData = Array.from({ length: numRounds }, (_, i) => {
    const point = { round: i + 1 }
    for (const method of methodsPresent) {
      const logs = matchingRuns.filter((r) => r.method === method).map((r) => r.round_log)
      const vals = logs.map((log) => log[i]?.cumulative_transmissions ?? null).filter((v) => v != null)
      if (vals.length) point[method] = vals.reduce((a, b) => a + b, 0) / vals.length
    }
    return point
  })

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold">RAFA-FL Results Dashboard</h1>
        <p className="mt-1 text-sm text-neutral-500">
          {nRuns} runs loaded from {resultsData.generated_from}
        </p>
      </div>

      <div className="flex flex-wrap gap-4">
        <label className="text-sm">
          <span className="mr-2 text-neutral-500">alpha</span>
          <select
            className="rounded-md border border-neutral-300 bg-white px-2 py-1 dark:border-neutral-700 dark:bg-neutral-900"
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
          >
            {alphas.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
        </label>
        <label className="text-sm">
          <span className="mr-2 text-neutral-500">num_clients</span>
          <select
            className="rounded-md border border-neutral-300 bg-white px-2 py-1 dark:border-neutral-700 dark:bg-neutral-900"
            value={numClients}
            onChange={(e) => setNumClients(Number(e.target.value))}
          >
            {clientCounts.map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        <label className="text-sm">
          <span className="mr-2 text-neutral-500">drift</span>
          <select
            className="rounded-md border border-neutral-300 bg-white px-2 py-1 dark:border-neutral-700 dark:bg-neutral-900"
            value={String(drift)}
            onChange={(e) => setDrift(e.target.value === 'true')}
          >
            <option value="true">True</option>
            <option value="false">False</option>
          </select>
        </label>
      </div>

      {conditionRows.length === 0 ? (
        <p className="text-sm text-neutral-500">No runs match this condition.</p>
      ) : (
        <>
          <section>
            <h2 className="mb-2 text-lg font-medium">Summary (mean ± 95% CI across seeds)</h2>
            <div className="overflow-x-auto rounded-lg border border-neutral-200 dark:border-neutral-800">
              <table className="w-full text-left text-sm">
                <thead className="bg-neutral-50 dark:bg-neutral-900">
                  <tr>
                    <th className="px-3 py-2">method</th>
                    <th className="px-3 py-2">n_seeds</th>
                    <th className="px-3 py-2">peak accuracy</th>
                    <th className="px-3 py-2">rounds to convergence</th>
                    <th className="px-3 py-2">transmissions</th>
                  </tr>
                </thead>
                <tbody>
                  {conditionRows.map((row) => (
                    <tr key={row.method} className="border-t border-neutral-200 dark:border-neutral-800">
                      <td className="px-3 py-2 font-medium" style={{ color: colorFor(row.method) }}>
                        {row.method}
                      </td>
                      <td className="px-3 py-2">{row.n_seeds}</td>
                      <td className="px-3 py-2">{fmtPair(row.peak_accuracy)}</td>
                      <td className="px-3 py-2">{fmtPair(row.rounds_to_convergence, 1)}</td>
                      <td className="px-3 py-2">{fmtPair(row.total_transmissions, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section>
            <h2 className="mb-2 text-lg font-medium">Accuracy per round (mean across seeds)</h2>
            <div className="h-80 w-full">
              <ResponsiveContainer>
                <LineChart data={accuracyChartData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-neutral-200 dark:stroke-neutral-800" />
                  <XAxis dataKey="round" tick={{ fontSize: 12 }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Legend />
                  {methodsPresent.map((m) => (
                    <Line key={m} type="monotone" dataKey={m} stroke={colorFor(m)} dot={false} strokeWidth={2} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section>
            <h2 className="mb-2 text-lg font-medium">Cumulative transmissions per round (mean across seeds)</h2>
            <div className="h-80 w-full">
              <ResponsiveContainer>
                <LineChart data={transmissionsChartData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-neutral-200 dark:stroke-neutral-800" />
                  <XAxis dataKey="round" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Legend />
                  {methodsPresent.map((m) => (
                    <Line key={m} type="monotone" dataKey={m} stroke={colorFor(m)} dot={false} strokeWidth={2} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          <p className="text-xs text-neutral-400">
            Per-client traces are not available in these logs (round_log records aggregate
            active/deferred counts only) -- no per-client breakdown is shown or fabricated.
          </p>
        </>
      )}
    </div>
  )
}

import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import Dashboard from './pages/Dashboard.jsx'
import Playground from './pages/Playground.jsx'
import Runner from './pages/Runner.jsx'

const RUNNER_ENABLED = import.meta.env.VITE_ENABLE_RUNNER === 'true'

// How the three pages relate:
//   Runner (/runner)      -- triggers a real training run -> writes results/*.json
//   (offline)                 -> `python ui/scripts/export_results.py` -> results_export.json
//   Dashboard (/)          -- reads that static export; shows what already happened
//   Playground (/playground) -- calls the live agent directly; independent of the other two,
//                                needs neither results data nor a training run

const navLinkClass = ({ isActive }) =>
  `px-3 py-2 rounded-md text-sm font-medium ${
    isActive
      ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900'
      : 'text-neutral-600 hover:bg-neutral-100 dark:text-neutral-300 dark:hover:bg-neutral-800'
  }`

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100">
        <header className="border-b border-neutral-200 dark:border-neutral-800">
          <nav className="mx-auto flex max-w-5xl items-center gap-2 px-4 py-3">
            <span className="mr-4 font-semibold">RAFA-FL</span>
            <NavLink to="/" end className={navLinkClass} title="Past results, read-only">Dashboard</NavLink>
            <NavLink to="/playground" className={navLinkClass} title="Test the agent's decision rule live, no training needed">Agent Playground</NavLink>
            {RUNNER_ENABLED && (
              <NavLink to="/runner" className={navLinkClass} title="Launch a real training run (local dev only)">Runner</NavLink>
            )}
          </nav>
        </header>
        <main className="mx-auto max-w-5xl px-4 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/playground" element={<Playground />} />
            {RUNNER_ENABLED && <Route path="/runner" element={<Runner />} />}
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App

import { Link, NavLink, Route, Routes } from 'react-router-dom'
import { Component, type ReactNode } from 'react'
import { useStore } from './store/useStore'
import { SessionsPage } from './pages/SessionsPage'
import { ScenePage } from './pages/ScenePage'
import { DevicesPage } from './pages/DevicesPage'
import { RTConfigPage } from './pages/RTConfigPage'
import { ResultMetricsPage } from './pages/ResultMetricsPage'
import { JobRunPage } from './pages/JobRunPage'
import { ResultsPage } from './pages/ResultsPage'
import { ScenarioPage } from './pages/ScenarioPage'
import { ScenarioResultsPage } from './pages/ScenarioResultsPage'

/** 전역 에러 경계 — 페이지 렌더 에러가 앱 전체를 흰 화면으로 만들지 않도록. */
class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  constructor(props: { children: ReactNode }) { super(props); this.state = { error: null } }
  static getDerivedStateFromError(error: Error) { return { error } }
  render() {
    if (this.state.error) {
      return (
        <div className="card m-6 border-red-300 bg-red-50">
          <h2 className="text-lg font-semibold text-red-700 mb-2">페이지 렌더 중 오류가 발생했습니다</h2>
          <pre className="text-xs text-red-600 whitespace-pre-wrap mb-3">{String(this.state.error?.message || this.state.error)}</pre>
          <div className="flex gap-2">
            <button className="btn btn-secondary" onClick={() => this.setState({ error: null })}>다시 시도</button>
            <a className="btn btn-primary" href="/">1.Sessions 로</a>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}

const nav = [
  { to: '/',         label: '1. Sessions'  },
  { to: '/scene',    label: '2. Scene'    },
  { to: '/devices',  label: '3. TX / RX'  },
  { to: '/rt',       label: '4. RT'       },
  { to: '/metrics',  label: '5. Metrics'  },
  { to: '/run',      label: '6. Run'      },
  { to: '/results',  label: '7. RT Results'  },
  { to: '/scenario', label: '8. Scenario' },
  { to: '/scenario-results', label: '9. Scenario Results' },
]

export default function App() {
  const session = useStore((s) => s.session)
  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-slate-900 text-white px-6 py-3 flex items-center gap-6 shadow">
        <Link to="/" className="text-xl font-bold tracking-tight">RT Web Agent</Link>
        <nav className="flex gap-2">
          {nav.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.to === '/'}
              className={({ isActive }) =>
                `text-sm px-2 py-1 rounded ${isActive ? 'bg-brand-600' : 'hover:bg-slate-700'}`
              }
            >
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="ml-auto text-xs opacity-80">
          {session ? <>session: <span className="font-mono">{session.label}</span></> : 'no session'}
        </div>
      </header>

      <main className="flex-1 p-6 max-w-7xl mx-auto w-full">
        <ErrorBoundary>
          <Routes>
            <Route path="/"        element={<SessionsPage />} />
            <Route path="/scene"   element={<ScenePage />} />
            <Route path="/devices" element={<DevicesPage />} />
            <Route path="/rt"      element={<RTConfigPage />} />
            <Route path="/metrics" element={<ResultMetricsPage />} />
            <Route path="/run"     element={<JobRunPage />} />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="/scenario" element={<ScenarioPage />} />
            <Route path="/scenario-results" element={<ScenarioResultsPage />} />
          </Routes>
        </ErrorBoundary>
      </main>

      <footer className="text-center text-xs text-slate-500 py-3">
        25* pipeline · Sionna RT · single-user mode
      </footer>
    </div>
  )
}

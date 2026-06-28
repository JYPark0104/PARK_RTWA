import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiClient, type MetricSpec } from '../lib/api'
import { useStore } from '../store/useStore'

export function ResultMetricsPage() {
  const navigate = useNavigate()
  const selected = useStore((s) => s.selectedMetrics)
  const toggle = useStore((s) => s.toggleMetric)
  const setSelected = useStore((s) => s.setSelectedMetrics)
  const setRT = useStore((s) => s.setRT)
  const rt = useStore((s) => s.rt)
  const session = useStore((s) => s.session)

  const [catalog, setCatalog] = useState<MetricSpec[]>([])
  const [presets, setPresets] = useState<Record<string, { label: string; description: string; metrics: string[] }>>({})
  const [resolve, setResolve] = useState<any>(null)

  useEffect(() => {
    apiClient.metricsCatalog().then((r) => setCatalog(r.metrics))
    apiClient.metricsPresets().then((r) => setPresets(r.presets))
  }, [])

  useEffect(() => {
    if (selected.length === 0) { setResolve(null); return }
    apiClient.resolveMetrics(selected).then(setResolve)
  }, [selected])

  function applyPreset(metrics: string[]) {
    setSelected(metrics)
    // coverage_only 프리셋이면 covg map enable
    if (metrics.length === 1 && metrics[0] === 'coverage_map') {
      setRT({ coverage_map: { ...rt.coverage_map, enabled: true } })
    }
  }

  const groupOrder = [
    { title: 'Coverage / Raw',    ids: ['coverage_map', 'ray_dump'] },
    { title: 'Derived (rays)',    ids: ['rsrp', 'pdp', 'padp', 'ray_stats'] },
    { title: 'Channel / Capacity',ids: ['ae_ofdm_channel', 'marginal_ccm', 'coupling_matrix', 'mean_channel', 'separability', 'su_capacity'] },
    { title: 'Beam Management',   ids: ['greedy_beams', 'swomp_beams', 'uplink_beams', 'qie_clustering', 'padp_dft_alignment', 'beam_pattern', 'eigenbeam_pattern'] },
  ]

  if (!session) return <div className="card text-sm text-slate-500">먼저 1.Sessions 에서 세션을 선택하세요.</div>

  return (
    <div className="space-y-4">
      <section className="card">
        <h2 className="text-lg font-semibold mb-3">프리셋 ({Object.keys(presets).length}종)</h2>
        <div className="flex flex-wrap gap-2">
          {Object.entries(presets).map(([key, p]) => (
            <button key={key} className="btn btn-secondary text-xs"
              title={p.description}
              onClick={() => applyPreset(p.metrics)}>
              {p.label}
            </button>
          ))}
        </div>
      </section>

      <section className="card">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold">결과 metric 선택</h2>
          <div className="text-xs text-slate-500">선택: {selected.length} / 19 (coverage_map 포함)</div>
        </div>

        {groupOrder.map((g) => (
          <div key={g.title} className="mb-4">
            <h3 className="text-sm font-semibold text-slate-700 mb-1">{g.title}</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
              {catalog
                .filter((c) => g.ids.includes(c.id))
                .map((c) => (
                  <label key={c.id} className={`border rounded-md p-2 flex gap-2 cursor-pointer text-sm
                                                 ${selected.includes(c.id) ? 'border-brand-500 bg-brand-50' : 'border-slate-300 hover:bg-slate-50'}`}>
                    <input type="checkbox" checked={selected.includes(c.id)} onChange={() => toggle(c.id)} />
                    <div className="flex-1">
                      <div className="font-medium">{c.label}</div>
                      <div className="text-xs text-slate-500">{c.description}</div>
                      <div className="text-[10px] font-mono text-slate-400">
                        {c.stages.length > 0 && <>stages: {c.stages.join(' → ')}</>}
                        {c.derived && <span className="text-emerald-600 ml-2">[derived]</span>}
                        {c.coverage_map && <span className="text-indigo-600 ml-2">[coverage]</span>}
                        {!c.needs_rx && <span className="text-rose-600 ml-2">[no-rx]</span>}
                      </div>
                    </div>
                  </label>
                ))}
            </div>
          </div>
        ))}
      </section>

      {resolve && (
        <section className="card text-sm bg-slate-50">
          <h3 className="font-semibold mb-1">의존성 분석</h3>
          <div className="font-mono text-xs space-y-1">
            <div>stages_ordered: [{resolve.stages_ordered.join(', ')}]</div>
            <div>derived: [{resolve.derived.join(', ')}]</div>
            <div>coverage_map: {String(resolve.coverage_map)} | needs_rx: {String(resolve.needs_rx)}</div>
          </div>
        </section>
      )}

      <button className="btn btn-primary"
        disabled={selected.length === 0}
        onClick={() => navigate('/run')}>
        다음: 잡 실행
      </button>
    </div>
  )
}

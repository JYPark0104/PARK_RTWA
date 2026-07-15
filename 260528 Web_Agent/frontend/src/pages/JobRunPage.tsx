import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiClient, formatApiError, type TimeEstimate } from '../lib/api'
import { useStore } from '../store/useStore'

export function JobRunPage() {
  const navigate = useNavigate()
  const session = useStore((s) => s.session)
  const sceneInfo = useStore((s) => s.sceneInfo)
  const tx = useStore((s) => s.tx)
  const rx = useStore((s) => s.rx)
  const antenna = useStore((s) => s.antenna)
  const rt = useStore((s) => s.rt)
  const selected = useStore((s) => s.selectedMetrics)
  const currentJob = useStore((s) => s.currentJob)
  const setJob = useStore((s) => s.setJob)
  const events = useStore((s) => s.jobEvents)
  const appendEvent = useStore((s) => s.appendEvent)
  const clearEvents = useStore((s) => s.clearEvents)

  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [queued, setQueued] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  // (A) 실행 전 시간 예측 — config 가 바뀔 때마다 디바운스 후 조회
  const [est, setEst] = useState<TimeEstimate | null>(null)
  const [estBusy, setEstBusy] = useState(false)

  useEffect(() => {
    return () => { wsRef.current?.close() }
  }, [])

  useEffect(() => {
    if (!session || tx.length === 0) { setEst(null); return }
    let alive = true
    setEstBusy(true)
    const timer = window.setTimeout(async () => {
      try {
        const payload = buildPayload(session.uuid, tx, rx, antenna, rt, selected)
        const e = await apiClient.estimateJob(session.uuid, payload, rxCount(rx))
        if (alive) setEst(e)
      } catch {
        if (alive) setEst(null)
      } finally {
        if (alive) setEstBusy(false)
      }
    }, 450)
    return () => { alive = false; window.clearTimeout(timer) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.uuid, tx, rx, antenna, rt, selected])

  async function submit() {
    if (!session) return
    setBusy(true); setErr(null); setQueued(false); clearEvents()
    try {
      const payload = buildPayload(session.uuid, tx, rx, antenna, rt, selected)
      const job = await apiClient.submitJob(session.uuid, payload)
      setJob(job)
      setQueued(true)
      openWS(job.job_id)
    } catch (ex: any) {
      setErr(formatApiError(ex))
    } finally {
      setBusy(false)
    }
  }

  function openWS(jobId: string) {
    wsRef.current?.close()
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const url = `${proto}://${location.host}/ws/jobs/${jobId}`
    const ws = new WebSocket(url)
    wsRef.current = ws
    ws.onmessage = (e) => {
      try {
        const ev = JSON.parse(e.data)
        appendEvent(ev)
        if (ev.kind === 'done' || ev.kind === 'error') {
          apiClient.getJob(jobId).then(setJob)
        } else {
          apiClient.getJob(jobId).then(setJob).catch(() => {})
        }
      } catch {}
    }
    ws.onclose = () => { /* noop */ }
  }

  async function cancel() {
    if (!currentJob) return
    await apiClient.cancelJob(currentJob.job_id)
    const j = await apiClient.getJob(currentJob.job_id)
    setJob(j)
  }

  if (!session) return <div className="card text-sm text-slate-500">먼저 1.Sessions 에서 세션을 선택하세요.</div>

  return (
    <div className="space-y-4">
      <section className="card">
        <h2 className="text-lg font-semibold mb-2">잡 제출</h2>
        <div className="text-sm grid grid-cols-2 lg:grid-cols-4 gap-2 mb-3">
          <div>session: <span className="font-mono">{session?.label ?? '-'}</span></div>
          <div>scene: <span className="font-mono">{sceneInfo ? `${sceneInfo.n_vertices}v / ${sceneInfo.n_faces}f` : '미업로드'}</span></div>
          <div>TX: <span className="font-mono">{tx.length}</span></div>
          <div>RX method: <span className="font-mono">{rx.method}</span></div>
          <div>antenna BS/UE: <span className="font-mono">{antenna.bs_rows}×{antenna.bs_cols} / {antenna.ue_rows}×{antenna.ue_cols}</span></div>
          <div>frequency: <span className="font-mono">{rt.frequency_ghz} GHz</span></div>
          <div>metrics: <span className="font-mono">{selected.length}개</span></div>
          <div>coverage: <span className="font-mono">{rt.coverage_map.enabled || selected.includes('coverage_map') ? 'on' : 'off'}</span></div>
        </div>

        <EstimateCard est={est} busy={estBusy} engine={rt.engine} />

        <div className="flex gap-3">
          <button className="btn btn-primary" disabled={busy || !session || tx.length === 0 || selected.length === 0} onClick={submit}>
            {busy ? '제출 중...' : '잡 실행'}
          </button>
          <button className="btn btn-secondary" disabled={!currentJob} onClick={cancel}>취소</button>
          <button className="btn btn-secondary" onClick={() => navigate('/results')}>
            RT Results → (큐 현황)
          </button>
        </div>
        {queued && (
          <div className="mt-3 rounded border border-emerald-300 bg-emerald-50 text-emerald-800 text-sm p-3">
            ✅ <b>'queueing' 상태로 등록되었습니다.</b> 앞선 작업이 끝나면 FIFO 순서로 자동 실행됩니다.
            <button className="btn btn-primary text-xs ml-3" onClick={() => navigate('/results')}>
              RT Results 에서 현황 보기 →
            </button>
          </div>
        )}
        {err && <div className="text-sm text-red-600 mt-2 whitespace-pre-wrap">{typeof err === 'string' ? err : JSON.stringify(err)}</div>}
      </section>

      {currentJob && (
        <section className="card">
          <h3 className="text-lg font-semibold mb-2">진행 상태</h3>
          <div className="text-sm space-y-1">
            <div>job_id: <span className="font-mono">{currentJob.job_id}</span></div>
            <div>state: <StateBadge s={currentJob.state} /></div>
            <div>current stage: <span className="font-mono">{currentJob.current_stage || '-'}</span></div>
            <div>progress: <ProgressBar v={currentJob.progress} /></div>
            <div>stages_done: <span className="font-mono">{currentJob.stages_done.join(', ') || '-'}</span></div>
            {currentJob.error && <div className="text-red-600">error: {currentJob.error}</div>}
          </div>

          <BatchProgress events={events} />

          <div className="mt-3 max-h-72 overflow-auto border border-slate-200 rounded bg-slate-900 text-slate-100 p-2 text-xs font-mono">
            {events.filter((e) => e.kind !== 'batch_progress').map((e, i) => (
              <div key={i} className={
                e.kind === 'error' ? 'text-red-300' :
                e.kind === 'stage_start' ? 'text-emerald-300' :
                e.kind === 'stage_end' ? 'text-blue-300' :
                e.kind === 'done' ? 'text-emerald-200 font-bold' : 'text-slate-200'
              }>
                [{e.timestamp?.slice(11, 19) ?? ''}] {e.kind}{e.stage ? ` (${e.stage})` : ''}: {e.message ?? ''}
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

/** (A) 실행 전 예상 소요 시간 카드. */
function EstimateCard({ est, busy, engine }: { est: TimeEstimate | null; busy: boolean; engine: string }) {
  const batchLike = engine === 'batch' || engine === 'intg'
  return (
    <div className="mb-3 rounded border border-indigo-200 bg-indigo-50 text-indigo-900 text-sm p-3">
      <div className="flex items-center gap-2">
        <span className="font-semibold">⏳ 예상 소요 시간</span>
        {busy && <span className="inline-block w-3.5 h-3.5 animate-spin rounded-full border-2 border-indigo-300 border-t-indigo-600" />}
      </div>
      {!batchLike && (
        <div className="text-xs text-indigo-700 mt-1">
          현재 예측은 batch / intg RT 엔진에서 지원됩니다. (현재 엔진: <span className="font-mono">{engine}</span>)
        </div>
      )}
      {batchLike && est?.available && (
        <div className="mt-1">
          <div className="text-lg font-bold">{est.summary_text}</div>
          <div className="text-xs text-indigo-700 mt-0.5">{est.detail_text}</div>
          {est.machine && (
            <div className="text-[11px] text-indigo-500 mt-0.5 font-mono">
              {est.machine.machine_id} · {est.machine.gpu_mode}{est.machine.gpu_name ? ` · ${est.machine.gpu_name}` : ''}
            </div>
          )}
        </div>
      )}
      {batchLike && est && !est.available && (
        <div className="text-xs text-indigo-700 mt-1">{est.reason || '예측 데이터 수집 중'}</div>
      )}
      {batchLike && !est && !busy && (
        <div className="text-xs text-indigo-700 mt-1">TX/RX 를 배치하면 예측이 표시됩니다.</div>
      )}
    </div>
  )
}

function StateBadge({ s }: { s: string }) {
  const color =
    s === 'succeeded' ? 'bg-emerald-100 text-emerald-700' :
    s === 'running'   ? 'bg-blue-100 text-blue-700' :
    s === 'failed'    ? 'bg-red-100 text-red-700' :
    s === 'cancelled' ? 'bg-slate-200 text-slate-700' :
    'bg-yellow-100 text-yellow-700'
  return <span className={`inline-block px-2 py-0.5 rounded-md text-xs ${color}`}>{s}</span>
}

function ProgressBar({ v }: { v: number }) {
  const pct = Math.min(100, Math.max(0, Math.round(v * 100)))
  return (
    <div className="w-full bg-slate-200 rounded h-2 overflow-hidden">
      <div className="bg-brand-600 h-2 transition-all" style={{ width: `${pct}%` }} />
    </div>
  )
}

function BatchProgress({ events }: { events: any[] }) {
  // 가장 최근 batch_progress 이벤트만 골라 '제자리 갱신' 바로 표시 (로그 스팸 방지)
  let last: any = null
  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i]?.kind === 'batch_progress') { last = events[i]; break }
  }
  if (!last) return null

  const batchPct = Math.round((last.batch_frac ?? 0) * 100)
  const txCur = (last.tx_index ?? 0) + 1
  const numTx = last.num_tx ?? 1
  const txPos = Array.isArray(last.tx_pos)
    ? `(${last.tx_pos.map((c: number) => c.toFixed(1)).join(', ')})`
    : ''
  const rsrp =
    last.rsrp_min != null && last.rsrp_max != null
      ? `${last.rsrp_min.toFixed(1)} ~ ${last.rsrp_max.toFixed(1)} dBm`
      : 'N/A'

  return (
    <div className="mt-3 border border-emerald-200 bg-emerald-50 rounded p-3 space-y-2">
      <div className="flex justify-between items-center text-sm">
        <span className="font-semibold text-emerald-800">
          Batch RT 진행 — TX {txCur}/{numTx} <span className="font-mono text-emerald-600">{txPos}</span>
        </span>
        <span className="font-mono text-emerald-700">
          배치 {last.batch}/{last.num_batches} ({batchPct}%)
        </span>
      </div>
      {/* 배치 진행 바 (제자리 갱신) */}
      <div className="w-full bg-emerald-100 rounded h-3 overflow-hidden">
        <div className="bg-emerald-500 h-3 transition-all" style={{ width: `${batchPct}%` }} />
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-x-4 gap-y-1 text-xs font-mono text-slate-700">
        <div>누적 RX: <b>{last.accumulated}</b>{last.total_rx ? ` / ${last.total_rx}` : ''}</div>
        <div>직전배치 유효: <b>{last.valid}/{last.batch_rx}</b></div>
        <div className="col-span-2">RSRP: <b>{rsrp}</b></div>
      </div>
    </div>
  )
}

/** RX 개수 추정 (예측 API 힌트용). ground_grid/facade 는 미리보기로 채워진 좌표 수 사용. */
function rxCount(rx: any): number | null {
  const nz = Array.isArray(rx.z_values) && rx.z_values.length ? rx.z_values.length : 1
  switch (rx.method) {
    case 'clicks':
      return rx.click_positions?.length ?? null
    case 'ground_grid':
      return rx.ground_positions?.length || null
    case 'facade':
      return (rx.facade_positions?.length || rx.facade_count) ?? null
    case 'grid':
      return rx.x_num && rx.y_num ? rx.x_num * rx.y_num * nz : null
    case 'explicit':
      return rx.x_num && rx.y_num ? rx.x_num * rx.y_num * nz : null
    case 'radial':
      return rx.radii_m?.length && rx.angles_num ? rx.radii_m.length * rx.angles_num * nz : null
    case 'street':
      return rx.num_points ? rx.num_points * nz : null
    default:
      return null
  }
}

function buildPayload(uuid: string, tx: any[], rx: any, antenna: any, rt: any, metrics: string[]) {
  const rx_grid = rx.method === 'clicks' ? null : {
    method: rx.method,
    x_start: rx.x_start, x_stop: rx.x_stop, x_num: rx.x_num,
    y_start: rx.y_start, y_stop: rx.y_stop, y_num: rx.y_num,
    z_values: rx.z_values,
    x_coords: rx.method === 'explicit' ? linspace(rx.x_start, rx.x_stop, rx.x_num) : undefined,
    y_coords: rx.method === 'explicit' ? linspace(rx.y_start, rx.y_stop, rx.y_num) : undefined,
    center_xy: rx.center_xy,
    radii_m: rx.radii_m,
    angles_start: rx.angles_start, angles_stop: rx.angles_stop, angles_num: rx.angles_num,
    path_points: rx.path_points, num_points: rx.num_points,
    // ground_grid (PARK_2 방식)
    grid_n: rx.grid_n, margin: rx.margin, rx_height: rx.rx_height, raycasting_z: rx.raycasting_z,
    max_height: rx.max_height,
    spacing: rx.rx_layout === 'spacing' ? rx.spacing_m : null,
    // facade (O2I, 건물 벽면)
    z_min: rx.z_min, z_max: rx.z_max, z_distance: rx.z_distance,
    facade_spacing: rx.facade_spacing, facade_epsilon: rx.facade_epsilon,
    facade_x_min: rx.method === 'facade' ? rx.facade_x_min : null,
    facade_x_max: rx.method === 'facade' ? rx.facade_x_max : null,
    facade_y_min: rx.method === 'facade' ? rx.facade_y_min : null,
    facade_y_max: rx.method === 'facade' ? rx.facade_y_max : null,
  }
  const rx_clicks = rx.method === 'clicks' ? { positions: rx.click_positions } : null
  return {
    session_uuid: uuid,
    // TX az/el(deg) → Sionna orientation (α=az, β=el, γ=0) [radian]. 백엔드가 Transmitter 에 적용.
    tx_list: tx.map((t: any) => ({
      position: t.position,
      name: t.name,
      orientation: [((t.az_deg ?? 0) * Math.PI) / 180, ((t.el_deg ?? 0) * Math.PI) / 180, 0],
    })),
    rx_grid,
    rx_clicks,
    antenna: { mode: 'simple', simple: antenna },
    rt,
    metrics: { metrics },
  }
}

function linspace(a: number, b: number, n: number): number[] {
  if (n <= 1) return [a]
  const step = (b - a) / (n - 1)
  return Array.from({ length: n }, (_, i) => a + step * i)
}

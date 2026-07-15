import { useEffect, useRef, useState } from 'react'
import { apiClient, type QueueDashboard as Dash, type QueueItem, type QueueScatter } from '../lib/api'

/**
 * 큐 대시보드 — Queueing / Processing / Done 3단.
 * 1초 폴링(기본)으로 /api/queue 갱신. 액션 핸들러는 제공된 것만 버튼으로 노출.
 * 소유자 게이팅: currentUserId 와 세션 user_id 가 일치(또는 세션에 소유자 없음)할 때만 액션 활성화.
 */
export function QueueDashboard({
  pollMs = 1000,
  currentUserId,
  openLabel = '열기',
  onOpen, onCancel, onDelete, onRename, onScenario,
  title,
}: {
  pollMs?: number
  currentUserId?: string
  openLabel?: string
  onOpen?: (uuid: string) => void
  onCancel?: (uuid: string) => void
  onDelete?: (uuid: string) => void
  onRename?: (uuid: string, label: string) => void
  onScenario?: (uuid: string) => void
  title?: string
}) {
  const [dash, setDash] = useState<Dash | null>(null)
  const timer = useRef<number | null>(null)
  const loaded = dash !== null

  useEffect(() => {
    let alive = true
    const tick = async () => {
      try { const d = await apiClient.queue(); if (alive) setDash(d) } catch { /* ignore */ }
    }
    tick()
    timer.current = window.setInterval(tick, pollMs)
    return () => { alive = false; if (timer.current) window.clearInterval(timer.current) }
  }, [pollMs])

  // 소유자 게이팅: user 선택돼 있고 (소유자 없음 or 본인) 이면 액션 허용
  const canAct = (it: QueueItem) =>
    !!currentUserId && (!it.user_id || it.user_id === currentUserId)

  const p = dash?.processing ?? null
  const q = dash?.queueing ?? []
  const done = dash?.done ?? []

  return (
    <div className="space-y-4">
      {title && <h2 className="text-lg font-semibold">{title}</h2>}

      {!loaded && (
        <section className="card flex items-center justify-center gap-3 py-10 text-slate-500">
          <span className="inline-block w-6 h-6 animate-spin rounded-full border-2 border-slate-300 border-t-blue-500" />
          <span className="text-sm">작업 현황 불러오는 중…</span>
        </section>
      )}

      {loaded && (<>
      {/* Queueing */}
      <section className="card">
        <h3 className="font-semibold mb-2">⏳ Queueing (대기, FIFO {q.length})</h3>
        {q.length ? (
          <Table>
            {q.map((it, i) => (
              <Row key={it.uuid} idx={i + 1} it={it}>
                {onOpen && canAct(it) && <Btn onClick={() => onOpen(it.uuid)}>{openLabel}/수정</Btn>}
                {onCancel && canAct(it) && <Btn danger onClick={() => onCancel(it.uuid)}>취소</Btn>}
                {onDelete && canAct(it) && <Btn danger onClick={() => onDelete(it.uuid)}>삭제</Btn>}
                {!canAct(it) && <LockNote />}
              </Row>
            ))}
          </Table>
        ) : (
          <div className="text-sm text-slate-400">대기 중인 작업이 없습니다.</div>
        )}
      </section>

      {/* Processing */}
      <section className="card">
        <h3 className="font-semibold mb-2 flex items-center gap-2">
          <span>🚀 Processing (실행 중)</span>
          <span className="inline-block w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
        </h3>
        {p ? (
          <div className="flex flex-col lg:flex-row gap-4">
            <div className="flex-1 space-y-2 min-w-0">
              <div className="flex flex-wrap justify-between items-center text-sm gap-2">
                <span className="flex items-center gap-3">
                  <span className="font-semibold text-slate-700">👤 {p.user_name || '—'}</span>
                  <span className="text-slate-500">🗺 {p.display_name || p.scene_name || '—'}</span>
                  <span className="font-mono text-xs text-slate-500">{antStr(p)}</span>
                  {p.engine && <span className="text-xs px-1.5 py-0.5 rounded bg-slate-100">{p.engine}</span>}
                </span>
                <span className="text-xs text-slate-500 flex items-center gap-2">
                  {p.elapsed_sec != null && <span>⏱ {fmtElapsed(p.elapsed_sec)}</span>}
                  {p.eta_sec != null
                    ? <span className="text-blue-600 font-medium">⏳ 남은 {fmtElapsed(p.eta_sec)}</span>
                    : <span className="text-slate-400">⏳ 준비 중…</span>}
                </span>
              </div>
              <div className="font-mono text-xs text-slate-500 truncate">{p.label}</div>
              <Bar v={p.progress} />
              <div className="flex justify-between items-center text-xs text-slate-500">
                <span>stage: {p.current_stage || '-'} · {Math.round((p.progress || 0) * 100)}%</span>
                {onCancel && canAct(p) && (
                  <button className="btn text-xs bg-red-50 text-red-600 border border-red-200 hover:bg-red-100"
                    onClick={() => onCancel(p.uuid)}>중단</button>
                )}
              </div>
            </div>
            {p.scatter && p.scatter.points.length > 0 && (
              <LiveScatter scatter={p.scatter} />
            )}
          </div>
        ) : (
          <div className="text-sm text-slate-400">실행 중인 작업이 없습니다.</div>
        )}
      </section>

      {/* Done */}
      <section className="card">
        <h3 className="font-semibold mb-2">✅ Done (완료 히스토리 {done.length})</h3>
        {done.length ? (
          <Table>
            {done.map((it) => (
              <Row key={it.uuid} it={it} badge={<StatusBadge s={it.status} />}>
                {onOpen && canAct(it) && <Btn onClick={() => onOpen(it.uuid)}>{openLabel}</Btn>}
                {onScenario && it.status === 'done' && canAct(it) &&
                  <Btn primary onClick={() => onScenario(it.uuid)}>scenario 생성 →</Btn>}
                {onRename && canAct(it) && <Btn onClick={() => {
                  const nl = prompt('새 라벨', it.label); if (nl && nl !== it.label) onRename(it.uuid, nl)
                }}>이름변경</Btn>}
                {onDelete && canAct(it) && <Btn danger onClick={() => onDelete(it.uuid)}>삭제</Btn>}
                {!canAct(it) && <LockNote />}
              </Row>
            ))}
          </Table>
        ) : (
          <div className="text-sm text-slate-400">완료된 작업이 없습니다.</div>
        )}
      </section>
      </>)}
    </div>
  )
}

function antStr(it: QueueItem): string {
  return `${it.bs_rows}×${it.bs_cols} / ${it.ue_rows}×${it.ue_cols}`
}

function Table({ children }: { children: React.ReactNode }) {
  return (
    <table className="w-full text-sm">
      <thead className="text-slate-400 text-xs">
        <tr className="border-b">
          <th className="text-left py-1 pr-2">User</th>
          <th className="text-left pr-2">Scene</th>
          <th className="text-left pr-2">Antenna(BS/UE)</th>
          <th className="text-left pr-2">Label</th>
          <th className="text-right">Actions</th>
        </tr>
      </thead>
      <tbody>{children}</tbody>
    </table>
  )
}

function Row({ idx, it, badge, children }: {
  idx?: number; it: QueueItem; badge?: React.ReactNode; children: React.ReactNode
}) {
  return (
    <tr className="border-b hover:bg-slate-50">
      <td className="py-1.5 pr-2 font-medium text-slate-700 whitespace-nowrap">
        {idx != null && <span className="font-mono text-xs text-slate-400 mr-1">{idx}.</span>}
        👤 {it.user_name || '—'}
      </td>
      <td className="pr-2 text-slate-600 whitespace-nowrap">{it.display_name || it.scene_name || '—'}</td>
      <td className="pr-2 font-mono text-xs text-slate-500 whitespace-nowrap">{antStr(it)}</td>
      <td className="pr-2 font-mono text-xs text-slate-500 max-w-[260px] truncate">
        {badge} <span title={it.label}>{it.label}</span>
      </td>
      <td className="text-right space-x-1 whitespace-nowrap">{children}</td>
    </tr>
  )
}

function LockNote() {
  return <span className="text-xs text-slate-400">다른 user 세션</span>
}

function Btn({ children, onClick, danger, primary }: {
  children: React.ReactNode; onClick: () => void; danger?: boolean; primary?: boolean
}) {
  const cls = primary
    ? 'btn btn-primary text-xs'
    : danger
      ? 'btn text-xs bg-red-50 text-red-600 border border-red-200 hover:bg-red-100'
      : 'btn btn-secondary text-xs'
  return <button className={cls} onClick={onClick}>{children}</button>
}

function StatusBadge({ s }: { s: string }) {
  const map: Record<string, string> = {
    done: 'bg-emerald-100 text-emerald-700',
    failed: 'bg-red-100 text-red-700',
    interrupted: 'bg-amber-100 text-amber-700',
    cancelled: 'bg-slate-200 text-slate-600',
  }
  return <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] ${map[s] || 'bg-slate-100 text-slate-600'}`}>{s}</span>
}

function Bar({ v }: { v: number }) {
  const pct = Math.min(100, Math.max(0, Math.round((v || 0) * 100)))
  return (
    <div className="w-full bg-slate-200 rounded h-2.5 overflow-hidden">
      <div className="bg-blue-500 h-2.5 transition-all" style={{ width: `${pct}%` }} />
    </div>
  )
}

function fmtElapsed(sec: number): string {
  const s = Math.floor(sec % 60), m = Math.floor((sec / 60) % 60), h = Math.floor(sec / 3600)
  if (h > 0) return `${h}h ${m}m ${s}s`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

/** 라이브 커버리지 스캐터 — 배치마다 채워지는 RSRP 2D 맵 (추가 RT 비용 0). */
function LiveScatter({ scatter }: { scatter: QueueScatter }) {
  const ref = useRef<HTMLCanvasElement | null>(null)
  const W = 240, H = 200
  useEffect(() => {
    const cv = ref.current
    if (!cv) return
    const ctx = cv.getContext('2d')!
    ctx.fillStyle = '#0f0f1a'; ctx.fillRect(0, 0, W, H)
    const [minX, minY, maxX, maxY] = scatter.bounds
    const spanX = Math.max(maxX - minX, 1e-6), spanY = Math.max(maxY - minY, 1e-6)
    const pad = 10
    const sx = (x: number) => pad + ((x - minX) / spanX) * (W - 2 * pad)
    const sy = (y: number) => H - pad - ((y - minY) / spanY) * (H - 2 * pad)  // y flip
    // RSRP 컬러 스케일 (lo=파랑 → hi=노랑/빨강), dead=어두운 회색
    const lo = scatter.rsrp_min ?? -120, hi = scatter.rsrp_max ?? -40
    const span = Math.max(hi - lo, 1e-6)
    for (const [x, y, r] of scatter.points) {
      ctx.beginPath()
      ctx.arc(sx(x), sy(y), 2.2, 0, Math.PI * 2)
      ctx.fillStyle = r == null ? '#3a3a44' : rsrpColor((r - lo) / span)
      ctx.fill()
    }
    return undefined
  }, [scatter])

  const pct = scatter.total_rx ? Math.round((scatter.done_rx / scatter.total_rx) * 100) : 0
  return (
    <div className="shrink-0">
      <div className="text-xs text-slate-500 mb-1">
        라이브 커버리지 (RSRP) · TX {scatter.tx_index + 1}/{scatter.num_tx} · {scatter.done_rx}/{scatter.total_rx} ({pct}%)
      </div>
      <canvas ref={ref} width={W} height={H} className="rounded border border-slate-200 bg-slate-950" />
      {scatter.rsrp_min != null && scatter.rsrp_max != null && (
        <div className="text-[10px] text-slate-400 mt-0.5">
          RSRP {scatter.rsrp_min.toFixed(0)} ~ {scatter.rsrp_max.toFixed(0)} dBm · dead=회색
        </div>
      )}
    </div>
  )
}

/** 0(약)~1(강) → 파랑→청록→초록→노랑→빨강 근사 (viridis-lite). */
function rsrpColor(t: number): string {
  const x = Math.max(0, Math.min(1, t))
  const stops = [
    [40, 50, 140], [30, 130, 160], [40, 170, 110], [180, 200, 60], [240, 90, 50],
  ]
  const seg = x * (stops.length - 1)
  const i = Math.min(Math.floor(seg), stops.length - 2)
  const f = seg - i
  const a = stops[i], b = stops[i + 1]
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f))
  return `rgb(${c[0]},${c[1]},${c[2]})`
}

export type { QueueItem }

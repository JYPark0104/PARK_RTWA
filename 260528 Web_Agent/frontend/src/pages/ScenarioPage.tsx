import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiClient, formatApiError} from '../lib/api'
import { useStore } from '../store/useStore'

type RX = { idx: number; x: number; y: number; z: number }
type TX = { idx: number; x: number; y: number; z: number }
type SData = { rx: RX[]; tx: TX[]; edges: number[][][]; num_rx: number; num_tx: number }

/**
 * 8. Scenario — 2D 탑뷰에서 마우스 드래그로 RX 경로를 그려 Mobility Scenario 구성.
 * 원본 m_mobility_scenario_builder.py 의 인터랙티브 뷰어를 웹 canvas 로 재현.
 *   드래그: 커서에 가장 가까운 RX 를 경로에 연속 추가 (중복 제외)
 *   undo / reset / TX 전환 / FPS·FRAMES_PER_STEP / 생성
 */
export function ScenarioPage() {
  const navigate = useNavigate()
  const session = useStore((s) => s.session)
  const rt = useStore((s) => s.rt)
  const setScenarioResult = useStore((s) => s.setScenarioResult)

  const [data, setData] = useState<SData | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [path, setPath] = useState<number[]>([])     // 선택된 RX index 순서
  const [txIndex, setTxIndex] = useState(0)
  const [fps, setFps] = useState(30)
  const [framesPerStep, setFramesPerStep] = useState(3)
  const [busy, setBusy] = useState(false)

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const dragging = useRef(false)
  const lastRx = useRef<number | null>(null)
  const panning = useRef<{ sx: number; sy: number; px: number; py: number } | null>(null)
  // 마우스 휠 줌/팬 (1=기본). pan 은 화면 픽셀 오프셋.
  const [view, setView] = useState({ zoom: 1, panX: 0, panY: 0 })

  async function loadData() {
    if (!session) return
    setLoading(true); setErr(null)
    try {
      const d = await apiClient.scenarioData(session.uuid)
      setData(d)
    } catch (ex: any) {
      setErr(formatApiError(ex))
    } finally { setLoading(false) }
  }
  useEffect(() => { loadData() }, [session?.uuid])

  // 세션을 다시 열면 이전에 생성/저장한 경로·키프레임 설정을 복원.
  useEffect(() => {
    if (!session) return
    let cancelled = false
    apiClient.scenarioResult(session.uuid)
      .then((r) => {
        if (cancelled || !r || !r.exists) return
        if (Array.isArray(r.path) && r.path.length) setPath(r.path)
        if (typeof r.tx_index === 'number') setTxIndex(r.tx_index)
        if (typeof r.fps === 'number') setFps(r.fps)
        if (typeof r.frames_per_step === 'number') setFramesPerStep(r.frames_per_step)
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [session?.uuid])

  // 키보드: z=undo, r=reset
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'z') setPath((p) => p.slice(0, -1))
      else if (e.key === 'r') setPath([])
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // world bounds (edges + rx)
  const bounds = useMemo(() => {
    if (!data) return null
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
    const upd = (x: number, y: number) => {
      if (x < minX) minX = x; if (x > maxX) maxX = x
      if (y < minY) minY = y; if (y > maxY) maxY = y
    }
    for (const r of data.rx) upd(r.x, r.y)
    for (const e of data.edges) { upd(e[0][0], e[0][1]); upd(e[1][0], e[1][1]) }
    if (!isFinite(minX)) return null
    const padX = (maxX - minX) * 0.03 + 1, padY = (maxY - minY) * 0.03 + 1
    return { minX: minX - padX, minY: minY - padY, maxX: maxX + padX, maxY: maxY + padY }
  }, [data])

  const CW = 900, CH = 680
  const tf = useMemo(() => {
    if (!bounds) return null
    const worldW = bounds.maxX - bounds.minX, worldH = bounds.maxY - bounds.minY
    const baseScale = Math.min(CW / worldW, CH / worldH)
    const scale = baseScale * view.zoom
    const ox = view.panX + (CW - worldW * scale) / 2
    const oy = view.panY + (CH - worldH * scale) / 2
    return {
      toScreen: (x: number, y: number): [number, number] => [
        (x - bounds.minX) * scale + ox,
        CH - ((y - bounds.minY) * scale + oy), // flip Y
      ],
      toWorld: (px: number, py: number): [number, number] => [
        (px - ox) / scale + bounds.minX,
        (CH - py - oy) / scale + bounds.minY,
      ],
    }
  }, [bounds, view])

  // 새 데이터(세션)면 줌/팬 초기화
  useEffect(() => { setView({ zoom: 1, panX: 0, panY: 0 }) }, [bounds])

  // 마우스 휠 줌 (커서 기준). passive:false 네이티브 리스너로 페이지 스크롤 방지.
  useEffect(() => {
    const cv = canvasRef.current
    if (!cv || !bounds) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = cv.getBoundingClientRect()
      const px = (e.clientX - rect.left) * (CW / rect.width)
      const py = (e.clientY - rect.top) * (CH / rect.height)
      const worldW = bounds.maxX - bounds.minX, worldH = bounds.maxY - bounds.minY
      const baseScale = Math.min(CW / worldW, CH / worldH)
      setView((v) => {
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
        const zoom = Math.max(0.5, Math.min(20, v.zoom * factor))
        // 커서 아래 world 점이 고정되도록 pan 재계산
        const scaleOld = baseScale * v.zoom
        const oxOld = v.panX + (CW - worldW * scaleOld) / 2
        const oyOld = v.panY + (CH - worldH * scaleOld) / 2
        const wx = (px - oxOld) / scaleOld + bounds.minX
        const wy = (CH - py - oyOld) / scaleOld + bounds.minY
        const scaleNew = baseScale * zoom
        const panX = px - (wx - bounds.minX) * scaleNew - (CW - worldW * scaleNew) / 2
        const panY = (CH - py) - (wy - bounds.minY) * scaleNew - (CH - worldH * scaleNew) / 2
        return { zoom, panX, panY }
      })
    }
    cv.addEventListener('wheel', onWheel, { passive: false })
    return () => cv.removeEventListener('wheel', onWheel)
  }, [bounds])

  function nearestRx(wx: number, wy: number): number | null {
    if (!data) return null
    let best = -1, bestD = Infinity
    for (const r of data.rx) {
      const d = (r.x - wx) ** 2 + (r.y - wy) ** 2
      if (d < bestD) { bestD = d; best = r.idx }
    }
    return best >= 0 ? best : null
  }

  // 캔버스 렌더
  useEffect(() => {
    const cv = canvasRef.current
    if (!cv || !data || !tf) return
    const ctx = cv.getContext('2d')!
    ctx.clearRect(0, 0, CW, CH)
    ctx.fillStyle = '#0f0f1a'; ctx.fillRect(0, 0, CW, CH)

    // edges
    ctx.strokeStyle = 'rgba(74,144,217,0.45)'; ctx.lineWidth = 0.4
    ctx.beginPath()
    for (const e of data.edges) {
      const [x1, y1] = tf.toScreen(e[0][0], e[0][1])
      const [x2, y2] = tf.toScreen(e[1][0], e[1][1])
      ctx.moveTo(x1, y1); ctx.lineTo(x2, y2)
    }
    ctx.stroke()

    // RX dots
    const rN = data.rx.length
    const rad = rN > 1500 ? 1.4 : rN > 600 ? 2 : 3
    ctx.fillStyle = 'rgba(0,229,255,0.7)'
    for (const r of data.rx) {
      const [sx, sy] = tf.toScreen(r.x, r.y)
      ctx.beginPath(); ctx.arc(sx, sy, rad, 0, Math.PI * 2); ctx.fill()
    }

    // path (orange) + lines + numbers
    const rxByIdx = new Map(data.rx.map((r) => [r.idx, r]))
    ctx.strokeStyle = 'orange'; ctx.lineWidth = 2
    ctx.beginPath()
    path.forEach((ri, i) => {
      const r = rxByIdx.get(ri); if (!r) return
      const [sx, sy] = tf.toScreen(r.x, r.y)
      if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy)
    })
    ctx.stroke()
    ctx.fillStyle = 'orange'
    path.forEach((ri, i) => {
      const r = rxByIdx.get(ri); if (!r) return
      const [sx, sy] = tf.toScreen(r.x, r.y)
      ctx.beginPath(); ctx.arc(sx, sy, 5, 0, Math.PI * 2); ctx.fill()
      ctx.fillStyle = '#fff'; ctx.font = '10px monospace'
      ctx.fillText(String(i + 1), sx + 6, sy - 6)
      ctx.fillStyle = 'orange'
    })

    // TX (selected red star, others gray)
    data.tx.forEach((t) => {
      const [sx, sy] = tf.toScreen(t.x, t.y)
      const sel = t.idx === txIndex
      ctx.fillStyle = sel ? '#ff2d2d' : '#888'
      drawStar(ctx, sx, sy, sel ? 11 : 7)
      ctx.fillStyle = sel ? '#ff7777' : '#aaa'; ctx.font = 'bold 11px sans-serif'
      ctx.fillText(`TX${t.idx}${sel ? ' ★' : ''}`, sx + 8, sy - 8)
    })
  }, [data, tf, path, txIndex])

  function evtWorld(e: React.PointerEvent): [number, number] | null {
    const cv = canvasRef.current; if (!cv || !tf) return null
    const rect = cv.getBoundingClientRect()
    const px = (e.clientX - rect.left) * (CW / rect.width)
    const py = (e.clientY - rect.top) * (CH / rect.height)
    return tf.toWorld(px, py)
  }

  function addAt(e: React.PointerEvent) {
    const w = evtWorld(e); if (!w) return
    const ri = nearestRx(w[0], w[1])
    if (ri == null || ri === lastRx.current) return
    lastRx.current = ri
    setPath((p) => [...p, ri])
  }

  async function generate() {
    if (!session || path.length === 0) return
    setBusy(true); setErr(null)
    try {
      const res = await apiClient.scenarioGenerate(session.uuid, {
        path, tx_index: txIndex, fps, frames_per_step: framesPerStep,
      })
      setScenarioResult({ ...res, session_uuid: session.uuid })
      navigate('/scenario-results')
    } catch (ex: any) {
      setErr(formatApiError(ex))
    } finally { setBusy(false) }
  }

  if (!session) return <div className="card">먼저 세션을 선택하세요.</div>
  if (rt.engine !== 'batch' && rt.engine !== 'intg') {
    return (
      <div className="card text-sm">
        Scenario Generator 는 <b>batch RT 엔진</b> 산출물(USDA/OBJ)이 필요합니다.
        4.RT 에서 batch RT 로 잡을 실행한 뒤 사용하세요.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <section className="card">
        <div className="flex justify-between items-center mb-2">
          <h2 className="text-lg font-semibold">8. Scenario — Mobility 경로 그리기</h2>
          <button className="btn btn-secondary" onClick={loadData} disabled={loading}>
            {loading ? '로딩...' : 'RT 결과 다시 불러오기'}
          </button>
        </div>
        {err && <div className="text-sm text-red-600 mb-2">{err}</div>}
        <p className="text-xs text-slate-500">
          맵 위에서 <b>마우스를 드래그(좌클릭)</b>하면 커서에 가장 가까운 RX 가 순서대로 경로에 추가됩니다.
          · <b>마우스 휠</b>: 확대/축소 · <b>우클릭 드래그</b>: 평행이동 · <b>z</b>: 마지막 취소 · <b>r</b>: 전체 초기화
          <button className="btn btn-secondary text-xs ml-2" onClick={() => setView({ zoom: 1, panX: 0, panY: 0 })}>
            줌 초기화
          </button>
        </p>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        <div className="lg:col-span-3 card">
          {data ? (
            <canvas
              ref={canvasRef}
              width={CW} height={CH}
              className="w-full rounded bg-slate-950 cursor-crosshair touch-none"
              onContextMenu={(e) => e.preventDefault()}
              onPointerDown={(e) => {
                (e.target as Element).setPointerCapture(e.pointerId)
                if (e.button === 2) {  // 우클릭 = 맵 평행이동
                  const cv = canvasRef.current!; const rect = cv.getBoundingClientRect()
                  const px = (e.clientX - rect.left) * (CW / rect.width)
                  const py = (e.clientY - rect.top) * (CH / rect.height)
                  panning.current = { sx: px, sy: py, px: view.panX, py: view.panY }
                  return
                }
                dragging.current = true; lastRx.current = null; addAt(e)
              }}
              onPointerMove={(e) => {
                if (panning.current) {
                  const cv = canvasRef.current!; const rect = cv.getBoundingClientRect()
                  const px = (e.clientX - rect.left) * (CW / rect.width)
                  const py = (e.clientY - rect.top) * (CH / rect.height)
                  const p = panning.current
                  setView((v) => ({ ...v, panX: p.px + (px - p.sx), panY: p.py - (py - p.sy) }))
                  return
                }
                if (dragging.current) addAt(e)
              }}
              onPointerUp={() => { dragging.current = false; panning.current = null }}
              onPointerLeave={() => { dragging.current = false; panning.current = null }}
            />
          ) : (
            <div className="text-slate-500 text-sm p-6">
              {loading ? 'RT 결과(USDA/OBJ) 로딩 중...' : 'batch RT 산출물을 불러오지 못했습니다. 먼저 6.Run 에서 batch 잡을 실행하세요.'}
            </div>
          )}
        </div>

        <div className="space-y-4">
          <section className="card">
            <h3 className="font-semibold mb-2">TX 선택</h3>
            <div className="flex flex-wrap gap-1">
              {(data?.tx ?? []).map((t) => (
                <button key={t.idx}
                  className={`btn text-xs ${t.idx === txIndex ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => setTxIndex(t.idx)}>TX{t.idx}</button>
              ))}
              {(!data || data.tx.length === 0) && <span className="text-xs text-slate-400">TX 없음</span>}
            </div>
          </section>

          <section className="card space-y-2">
            <h3 className="font-semibold">키프레임 설정</h3>
            <NumberRow label="FPS" v={fps} on={setFps} />
            <NumberRow label="FRAMES_PER_STEP" v={framesPerStep} on={setFramesPerStep} />
            <div className="text-xs text-slate-500">
              RX 당 {framesPerStep} 프레임 · {fps} FPS → RX 당 {(framesPerStep / fps).toFixed(2)}초
            </div>
          </section>

          <section className="card">
            <h3 className="font-semibold mb-2">경로 ({path.length} RX)</h3>
            <div className="text-xs font-mono max-h-32 overflow-auto text-slate-600">
              {path.length ? path.map((r) => `RX${r}`).join(' → ') : '(드래그로 선택)'}
            </div>
            <div className="flex gap-2 mt-2">
              <button className="btn btn-secondary text-xs" onClick={() => setPath((p) => p.slice(0, -1))}>실행취소(z)</button>
              <button className="btn btn-secondary text-xs" onClick={() => setPath([])}>초기화(r)</button>
            </div>
          </section>

          <button className="btn btn-primary w-full" disabled={busy || path.length === 0} onClick={generate}>
            {busy ? '생성 중...' : '시나리오 생성 →'}
          </button>
        </div>
      </div>
    </div>
  )
}

function drawStar(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number) {
  ctx.beginPath()
  for (let i = 0; i < 10; i++) {
    const ang = (Math.PI / 5) * i - Math.PI / 2
    const rr = i % 2 === 0 ? r : r * 0.45
    const x = cx + rr * Math.cos(ang), y = cy + rr * Math.sin(ang)
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
  }
  ctx.closePath(); ctx.fill()
}

function NumberRow({ label, v, on }: { label: string; v: number; on: (v: number) => void }) {
  return (
    <label className="flex justify-between items-center gap-2 text-sm">
      <span className="text-slate-600">{label}</span>
      <input type="number" className="input w-24 text-right" value={v}
        onChange={(e) => { const n = parseInt(e.target.value); if (!isNaN(n)) on(n) }} />
    </label>
  )
}

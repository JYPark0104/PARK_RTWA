import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { Coord3 } from '../lib/api'
import { apiClient } from '../lib/api'
import { SceneViewer } from '../components/SceneViewer'
import { useStore } from '../store/useStore'

type Mode = 'tx' | 'rx_click'

export function DevicesPage() {
  const navigate = useNavigate()
  const session = useStore((s) => s.session)
  const sceneInfo = useStore((s) => s.sceneInfo)
  const setSceneInfo = useStore((s) => s.setSceneInfo)
  const tx = useStore((s) => s.tx)
  const addTX = useStore((s) => s.addTX)
  const addTXFull = useStore((s) => s.addTXFull)
  const removeTX = useStore((s) => s.removeTX)
  const clearTX = useStore((s) => s.clearTX)
  const rx = useStore((s) => s.rx)
  const setRX = useStore((s) => s.setRX)
  const selectedMetrics = useStore((s) => s.selectedMetrics)
  const rt = useStore((s) => s.rt)

  const [mode, setMode] = useState<Mode>('tx')
  const [gridBusy, setGridBusy] = useState(false)
  const [snapBusy, setSnapBusy] = useState(false)
  const [showTxWarn, setShowTxWarn] = useState(false)
  // TX 좌표 직접 입력 (재현 가능한 정확 배치)
  const [showTxManual, setShowTxManual] = useState(false)
  const [manualTx, setManualTx] = useState<{ x: string; y: string; z: string }>({ x: '', y: '', z: '' })
  const [manualSnap, setManualSnap] = useState(false)

  async function addManualTX() {
    const x = parseFloat(manualTx.x), y = parseFloat(manualTx.y)
    if (!isFinite(x) || !isFinite(y)) { setShowTxWarn(false); return }
    if (manualSnap && session) {
      // 지면 스냅: X,Y 만 입력하면 Z 는 지면+offset 으로 자동 계산
      setSnapBusy(true)
      try {
        const r = await apiClient.txSnap(session.uuid, x, y, rt.tx_ground_offset_m)
        addTXFull({
          position: r.tx, orientation: [0, 0, 0], name: `tx${tx.length + 1}`,
          clicked: [x, y, r.ground_z ?? 0],
        })
      } catch {
        const z = parseFloat(manualTx.z)
        addTXFull({ position: [x, y, isFinite(z) ? z : 0], orientation: [0, 0, 0], name: `tx${tx.length + 1}`, clicked: null })
      } finally { setSnapBusy(false) }
    } else {
      // 정확 좌표 그대로 (재현용). Z 미입력 시 0.
      const z = parseFloat(manualTx.z)
      addTXFull({
        position: [x, y, isFinite(z) ? z : 0], orientation: [0, 0, 0],
        name: `tx${tx.length + 1}`, clicked: null,
      })
    }
    setManualTx({ x: '', y: '', z: '' })
  }

  function goNext() {
    if (tx.length === 0) { setShowTxWarn(true); return }
    navigate('/rt')
  }

  async function previewGroundGrid() {
    if (!session) return
    setGridBusy(true)
    try {
      const r = await apiClient.rxGroundGrid(session.uuid, {
        grid_n: rx.grid_n, margin: rx.margin, rx_height: rx.rx_height,
        raycasting_z: rx.raycasting_z, max_height: rx.max_height,
        x_start: rx.x_start, x_stop: rx.x_stop, y_start: rx.y_start, y_stop: rx.y_stop,
        spacing: rx.rx_layout === 'spacing' ? rx.spacing_m : null,
      })
      setRX({ ground_positions: r.positions })
    } catch { /* ignore */ } finally { setGridBusy(false) }
  }

  async function placeTX(p: Coord3) {
    if (!session) { addTX(p); return }
    setSnapBusy(true)
    try {
      const r = await apiClient.txSnap(session.uuid, p[0], p[1], rt.tx_ground_offset_m)
      addTXFull({
        position: r.tx, orientation: [0, 0, 0], name: `tx${tx.length + 1}`,
        clicked: [p[0], p[1], r.ground_z ?? p[2]],
      })
    } catch {
      addTX(p)  // 스냅 실패 시 클릭 지점 그대로
    } finally { setSnapBusy(false) }
  }

  useEffect(() => {
    if (!session || sceneInfo) return
    apiClient.sceneInfo(session.uuid).then(setSceneInfo).catch(() => {})
  }, [session?.uuid])

  // ground_grid 가 선택되면 씬 경계 → 레이캐스팅 지면 격자를 자동 미리보기.
  // (PARK_2 방식: boundary → 하늘에서 ↓ ray casting → 지면 hit 위치에만 RX, 지면+높이)
  useEffect(() => {
    if (!session || !sceneInfo) return
    if (rx.method !== 'ground_grid') return
    previewGroundGrid()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.uuid, sceneInfo, rx.method, rx.grid_n, rx.margin, rx.rx_height, rx.max_height,
      rx.x_start, rx.x_stop, rx.y_start, rx.y_stop])

  // grid / ground_grid 진입 시 X/Y 범위가 bbox 밖이면 맵 전체로 초기화 (슬라이더 기본값)
  useEffect(() => {
    if (!sceneInfo) return
    if (rx.method !== 'grid' && rx.method !== 'ground_grid') return
    const xMin = sceneInfo.aabb_min[0], xMax = sceneInfo.aabb_max[0]
    const yMin = sceneInfo.aabb_min[1], yMax = sceneInfo.aabb_max[1]
    const fix: any = {}
    if (rx.x_start < xMin || rx.x_start > xMax || rx.x_stop < xMin || rx.x_stop > xMax) {
      fix.x_start = Math.round(xMin); fix.x_stop = Math.round(xMax)
    }
    if (rx.y_start < yMin || rx.y_start > yMax || rx.y_stop < yMin || rx.y_stop > yMax) {
      fix.y_start = Math.round(yMin); fix.y_stop = Math.round(yMax)
    }
    if (Object.keys(fix).length) setRX(fix)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rx.method, sceneInfo])

  if (!session) return <div className="card">먼저 세션을 선택하세요.</div>
  if (!sceneInfo) return <div className="card">먼저 씬을 업로드하세요.</div>

  const rxPositions: Coord3[] = computeRX(rx, sceneInfo.aabb_min, sceneInfo.aabb_max)

  const isCoverageOnly = selectedMetrics.includes('coverage_map') &&
                        !selectedMetrics.some((m) => m !== 'coverage_map')

  return (
    <div className="space-y-4">
      {isCoverageOnly && (
        <div className="card border-emerald-300 bg-emerald-50 text-emerald-800 text-sm">
          Coverage Map only 모드: RX 배치 없이도 진행 가능. TX만 한 개 이상 배치하세요.
        </div>
      )}

      <section className="card">
        <div className="flex flex-wrap items-center gap-3">
          <h3 className="text-lg font-semibold mr-2">배치 모드:</h3>
          <button className={`btn ${mode === 'tx' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setMode('tx')}>
            TX 클릭 추가
          </button>
          <button className={`btn ${mode === 'rx_click' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setMode('rx_click')}>
            RX 클릭 추가 (snap-to-surface)
          </button>
          <button className="btn btn-secondary" onClick={clearTX}>TX 모두 삭제</button>
          <button className="btn btn-secondary" onClick={() => setRX({ click_positions: [] })}>RX click 초기화</button>
          {snapBusy && <span className="text-xs text-amber-600">TX 지면 스냅 중...</span>}
        </div>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <SceneViewer
            uuid={session.uuid}
            sceneInfo={sceneInfo}
            tx={tx}
            rx={rx.method === 'clicks' ? rx.click_positions : rx.method === 'ground_grid' ? rx.ground_positions : rxPositions}
            txClicked={tx.map((t) => t.clicked).filter(Boolean) as Coord3[]}
            coverageOverlay={
              rt.coverage_map.enabled || isCoverageOnly
                ? {
                    center: [
                      (sceneInfo.aabb_min[0] + sceneInfo.aabb_max[0]) / 2,
                      (sceneInfo.aabb_min[1] + sceneInfo.aabb_max[1]) / 2,
                    ],
                    size: [
                      sceneInfo.aabb_max[0] - sceneInfo.aabb_min[0],
                      sceneInfo.aabb_max[1] - sceneInfo.aabb_min[1],
                    ],
                    height: rt.coverage_map.height_m,
                  }
                : undefined
            }
            onPickSurface={(p) => {
              if (mode === 'tx') placeTX(p)
              else if (mode === 'rx_click') {
                setRX({ method: 'clicks', click_positions: [...rx.click_positions, p] })
              }
            }}
          />
        </div>

        <div className="space-y-4">
          <section className="card">
            <div className="flex items-start justify-between mb-2">
              <h3 className="font-semibold">TX 목록 ({tx.length})</h3>
              <button
                className={`btn text-xs px-2 py-1 ${showTxManual ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setShowTxManual((v) => !v)}
              >
                TX 좌표 직접 설정
              </button>
            </div>

            {showTxManual && (
              <div className="mb-3 p-2 rounded border border-slate-200 bg-slate-50 space-y-2">
                <div className="grid grid-cols-3 gap-2">
                  {(['x', 'y', 'z'] as const).map((k) => (
                    <label key={k} className="text-xs text-slate-600">
                      <span className="block mb-0.5 uppercase">{k} (m)</span>
                      <input
                        type="number" step="any" inputMode="decimal"
                        className="input w-full text-xs py-1"
                        placeholder={k}
                        value={manualTx[k]}
                        disabled={k === 'z' && manualSnap}
                        onChange={(e) => setManualTx((s) => ({ ...s, [k]: e.target.value }))}
                        onKeyDown={(e) => { if (e.key === 'Enter') addManualTX() }}
                      />
                    </label>
                  ))}
                </div>
                <label className="flex items-center gap-1.5 text-xs text-slate-600">
                  <input type="checkbox" checked={manualSnap} onChange={(e) => setManualSnap(e.target.checked)} />
                  지면 스냅 (Z 자동 = 지면 + {rt.tx_ground_offset_m}m 이격)
                </label>
                <div className="flex items-center gap-2">
                  <button className="btn btn-primary text-xs px-3 py-1" onClick={addManualTX} disabled={snapBusy}>
                    추가
                  </button>
                  <span className="text-[11px] text-slate-400">
                    {manualSnap ? 'X · Y 입력 → Z 자동' : 'X · Y · Z 정확 좌표로 추가 (재현용)'}
                  </span>
                </div>
              </div>
            )}

            <ul className="text-sm space-y-1 max-h-60 overflow-auto">
              {tx.map((t, i) => (
                <li key={i} className="flex justify-between items-center border-b py-1">
                  <span className="font-mono text-xs">
                    [{t.position.map((v) => v.toFixed(2)).join(', ')}]
                  </span>
                  <button className="btn btn-danger text-xs px-2 py-0.5" onClick={() => removeTX(i)}>삭제</button>
                </li>
              ))}
              {tx.length === 0 && <li className="text-slate-400 text-xs">TX 모드에서 표면을 클릭하세요.</li>}
            </ul>
          </section>

          <section className="card">
            <h3 className="font-semibold mb-2">RX 배치 방법</h3>
            <div className="flex flex-wrap gap-1 mb-3">
              {(['grid', 'ground_grid', 'explicit', 'radial', 'street', 'clicks'] as const).map((m) => (
                <button key={m}
                  className={`btn text-xs ${rx.method === m ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => setRX({ method: m })}>{m}</button>
              ))}
            </div>

            {rx.method === 'ground_grid' && (
              <div className="space-y-2 text-sm">
                <div className="text-xs text-slate-500">
                  씬 경계에서 하늘 → 지면 레이캐스팅으로 지면 위에만 RX를 격자 배치합니다. (PARK_2 방식)
                </div>
                <div>
                  <div className="flex justify-between text-xs text-slate-600">
                    <span>X 영역 (레이캐스팅 대상)</span>
                    <span className="font-mono">{rx.x_start.toFixed(0)} ~ {rx.x_stop.toFixed(0)} m</span>
                  </div>
                  <DualRange
                    min={Math.floor(sceneInfo.aabb_min[0])} max={Math.ceil(sceneInfo.aabb_max[0])}
                    lo={rx.x_start} hi={rx.x_stop}
                    onChange={(lo, hi) => setRX({ x_start: lo, x_stop: hi })}
                  />
                </div>
                <div>
                  <div className="flex justify-between text-xs text-slate-600">
                    <span>Y 영역 (레이캐스팅 대상)</span>
                    <span className="font-mono">{rx.y_start.toFixed(0)} ~ {rx.y_stop.toFixed(0)} m</span>
                  </div>
                  <DualRange
                    min={Math.floor(sceneInfo.aabb_min[1])} max={Math.ceil(sceneInfo.aabb_max[1])}
                    lo={rx.y_start} hi={rx.y_stop}
                    onChange={(lo, hi) => setRX({ y_start: lo, y_stop: hi })}
                  />
                </div>
                <button className="btn btn-secondary text-xs" onClick={() => setRX({
                  x_start: Math.round(sceneInfo.aabb_min[0]), x_stop: Math.round(sceneInfo.aabb_max[0]),
                  y_start: Math.round(sceneInfo.aabb_min[1]), y_stop: Math.round(sceneInfo.aabb_max[1]),
                })}>맵 전체(bbox)</button>
                {/* RX 배치 방식: 밀도(grid_n) vs 간격(m) */}
                <div className="flex gap-2 text-xs items-center">
                  <span className="text-slate-600">배치 방식:</span>
                  <button onClick={() => setRX({ rx_layout: 'density' })}
                    className={`px-2 py-1 rounded border ${rx.rx_layout !== 'spacing' ? 'bg-sky-600 text-white border-sky-600' : 'bg-white text-slate-600 border-slate-300'}`}>밀도(grid_n)</button>
                  <button onClick={() => setRX({ rx_layout: 'spacing' })}
                    className={`px-2 py-1 rounded border ${rx.rx_layout === 'spacing' ? 'bg-sky-600 text-white border-sky-600' : 'bg-white text-slate-600 border-slate-300'}`}>간격(m)</button>
                </div>
                {rx.rx_layout === 'spacing' ? (
                  <>
                    <NumberRow label="RX 간격 [m] (정사각)" v={rx.spacing_m} on={(v) => setRX({ spacing_m: Math.max(0.1, v) })} />
                    <div className="text-[11px] text-slate-400 -mt-1">X·Y 동일 간격으로 격자 배치. 간격이 너무 작으면 자동으로 안전 간격(축당 ≤200)으로 보정됩니다.</div>
                  </>
                ) : (
                  <>
                    <NumberRow label="grid_n (축당)" v={rx.grid_n} on={(v) => setRX({ grid_n: Math.max(2, Math.min(200, v)) })} integer />
                    <div className="text-[11px] text-slate-400 -mt-1">축당 2~200 (총 grid_n² 후보). 너무 크면 RX 폭증 주의.</div>
                  </>
                )}
                <NumberRow label="margin (0~0.5)" v={rx.margin} on={(v) => setRX({ margin: v })} />
                <NumberRow label="RX 높이 [m]" v={rx.rx_height} on={(v) => setRX({ rx_height: v })} />
                <label className="flex justify-between items-center gap-2 text-sm">
                  <span className="text-slate-600">설치 최대 높이 [m]</span>
                  <input
                    type="number" className="input w-28 text-right"
                    placeholder="제한 없음"
                    value={rx.max_height ?? ''}
                    onChange={(e) => {
                      const s = e.target.value.trim()
                      setRX({ max_height: s === '' ? null : parseFloat(s) })
                    }}
                  />
                </label>
                <div className="text-[11px] text-slate-400 -mt-1">
                  지면고도+RX높이가 이 값을 넘는 RX(건물 옥상 등)는 배치에서 제거됩니다. 비우면 제한 없음.
                </div>
                <button className="btn btn-secondary text-xs" onClick={previewGroundGrid} disabled={gridBusy}>
                  {gridBusy ? '계산 중...' : '지면 격자 미리보기'}
                </button>
                <div className="text-xs text-slate-500">
                  후보 {rx.grid_n * rx.grid_n}개 중 지면 RX: <span className="font-mono">{rx.ground_positions.length}</span>개
                  {rx.max_height != null && <span className="text-slate-400"> (≤{rx.max_height}m)</span>}
                </div>
              </div>
            )}

            {rx.method === 'grid' && (
              <div className="space-y-3 text-sm">
                <div className="text-xs text-slate-500">
                  맵 bbox 안에서 RX 직사각형 영역을 슬라이더 양끝 노브로 조절하세요. (3D 뷰에 실시간 반영)
                </div>
                <div>
                  <div className="flex justify-between text-xs text-slate-600">
                    <span>X 영역</span>
                    <span className="font-mono">{rx.x_start.toFixed(0)} ~ {rx.x_stop.toFixed(0)} m</span>
                  </div>
                  <DualRange
                    min={Math.floor(sceneInfo.aabb_min[0])} max={Math.ceil(sceneInfo.aabb_max[0])}
                    lo={rx.x_start} hi={rx.x_stop}
                    onChange={(lo, hi) => setRX({ x_start: lo, x_stop: hi })}
                  />
                </div>
                <div>
                  <div className="flex justify-between text-xs text-slate-600">
                    <span>Y 영역</span>
                    <span className="font-mono">{rx.y_start.toFixed(0)} ~ {rx.y_stop.toFixed(0)} m</span>
                  </div>
                  <DualRange
                    min={Math.floor(sceneInfo.aabb_min[1])} max={Math.ceil(sceneInfo.aabb_max[1])}
                    lo={rx.y_start} hi={rx.y_stop}
                    onChange={(lo, hi) => setRX({ y_start: lo, y_stop: hi })}
                  />
                </div>
                <button className="btn btn-secondary text-xs" onClick={() => setRX({
                  x_start: Math.round(sceneInfo.aabb_min[0]), x_stop: Math.round(sceneInfo.aabb_max[0]),
                  y_start: Math.round(sceneInfo.aabb_min[1]), y_stop: Math.round(sceneInfo.aabb_max[1]),
                })}>맵 전체(bbox)</button>
                <NumberRow label="X num" v={rx.x_num} on={(v) => setRX({ x_num: v })} integer />
                <NumberRow label="Y num" v={rx.y_num} on={(v) => setRX({ y_num: v })} integer />
                <NumberRow label="z"     v={rx.z_values[0] ?? 1.5} on={(v) => setRX({ z_values: [v] })} />
                <div className="text-xs text-slate-500">총 RX: {rxPositions.length}</div>
              </div>
            )}

            {rx.method === 'radial' && (
              <div className="space-y-2 text-sm">
                <div>center XY: <input className="input"
                    value={rx.center_xy?.join(',') ?? ''}
                    onChange={(e) => {
                      const v = e.target.value.split(',').map(Number)
                      if (v.length === 2 && !v.some(Number.isNaN)) setRX({ center_xy: [v[0], v[1]] })
                    }} placeholder="0,0" /></div>
                <NumberRow label="angle start" v={rx.angles_start} on={(v) => setRX({ angles_start: v })} />
                <NumberRow label="angle stop"  v={rx.angles_stop}  on={(v) => setRX({ angles_stop: v })} />
                <NumberRow label="angle num"   v={rx.angles_num}   on={(v) => setRX({ angles_num: v })} integer />
                <div className="text-xs text-slate-500">총 RX: {rxPositions.length}</div>
              </div>
            )}

            {rx.method === 'street' && (
              <div className="space-y-2 text-sm">
                <NumberRow label="num_points" v={rx.num_points} on={(v) => setRX({ num_points: v })} integer />
                <div>start: <input className="input"
                    value={rx.path_points[0]?.join(',') ?? ''}
                    onChange={(e) => {
                      const v = e.target.value.split(',').map(Number)
                      if (v.length === 2) setRX({ path_points: [[v[0], v[1]], rx.path_points[1]] })
                    }} /></div>
                <div>end: <input className="input"
                    value={rx.path_points[1]?.join(',') ?? ''}
                    onChange={(e) => {
                      const v = e.target.value.split(',').map(Number)
                      if (v.length === 2) setRX({ path_points: [rx.path_points[0], [v[0], v[1]]] })
                    }} /></div>
              </div>
            )}

            {rx.method === 'clicks' && (
              <div className="text-sm">
                <div className="text-slate-500">표면 클릭으로 추가. 현재 {rx.click_positions.length}개.</div>
              </div>
            )}
          </section>

          <button className="btn btn-primary w-full" onClick={goNext}>
            다음: RT 설정
          </button>
        </div>
      </div>

      {showTxWarn && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
             onClick={() => setShowTxWarn(false)}>
          <div className="card max-w-md w-full mx-4 bg-white shadow-xl" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-lg font-semibold mb-2">TX가 지정되지 않았습니다</h3>
            <p className="text-sm text-slate-600 mb-4">
              현재 배치된 TX가 한 대도 없습니다. TX 없이 진행하면 Ray Tracing 결과가
              비어 있을 수 있습니다. 어떻게 할까요?
            </p>
            <div className="flex justify-end gap-2">
              <button className="btn btn-secondary" onClick={() => { setShowTxWarn(false); setMode('tx') }}>
                돌아가기 (TX 설치)
              </button>
              <button className="btn btn-primary" onClick={() => { setShowTxWarn(false); navigate('/rt') }}>
                TX 없이 이대로 진행
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function NumberRow({ label, v, on, integer = false }: { label: string; v: number; on: (v: number) => void; integer?: boolean }) {
  return (
    <label className="flex justify-between items-center gap-2">
      <span className="text-slate-600">{label}</span>
      <input
        type="number"
        className="input w-32 text-right"
        value={v}
        onChange={(e) => {
          const n = integer ? parseInt(e.target.value) : parseFloat(e.target.value)
          if (!isNaN(n)) on(n)
        }}
      />
    </label>
  )
}

/** 듀얼 노브 범위 슬라이더 — (min)──[lo]////[hi]──(max). 양끝 노브를 드래그해 영역 지정. */
function DualRange({ min, max, lo, hi, onChange }: {
  min: number; max: number; lo: number; hi: number; onChange: (lo: number, hi: number) => void
}) {
  const trackRef = useRef<HTMLDivElement | null>(null)
  const span = (max - min) || 1
  const clamp = (v: number) => Math.min(max, Math.max(min, v))
  const loC = clamp(lo), hiC = clamp(hi)
  const pct = (v: number) => ((clamp(v) - min) / span) * 100

  function startDrag(which: 'lo' | 'hi') {
    return (e: React.PointerEvent) => {
      e.preventDefault()
      const move = (ev: PointerEvent) => {
        const rect = trackRef.current?.getBoundingClientRect()
        if (!rect) return
        const t = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width))
        const v = Math.round(min + t * span)
        if (which === 'lo') onChange(Math.min(v, hiC), hiC)
        else onChange(loC, Math.max(v, loC))
      }
      const up = () => {
        window.removeEventListener('pointermove', move)
        window.removeEventListener('pointerup', up)
      }
      window.addEventListener('pointermove', move)
      window.addEventListener('pointerup', up)
    }
  }

  return (
    <div ref={trackRef} className="relative h-6 select-none touch-none">
      <div className="absolute top-1/2 -translate-y-1/2 w-full h-1.5 bg-slate-200 rounded" />
      <div className="absolute top-1/2 -translate-y-1/2 h-1.5 bg-brand-600 rounded"
        style={{ left: `${pct(loC)}%`, width: `${Math.max(0, pct(hiC) - pct(loC))}%` }} />
      <div onPointerDown={startDrag('lo')}
        className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-4 h-4 rounded-full bg-white border-2 border-brand-600 shadow cursor-ew-resize"
        style={{ left: `${pct(loC)}%` }} />
      <div onPointerDown={startDrag('hi')}
        className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-4 h-4 rounded-full bg-white border-2 border-brand-600 shadow cursor-ew-resize"
        style={{ left: `${pct(hiC)}%` }} />
    </div>
  )
}

function computeRX(rx: ReturnType<typeof useStore.getState>['rx'], _aabbMin: number[], _aabbMax: number[]): Coord3[] {
  if (rx.method === 'clicks') return rx.click_positions
  if (rx.method === 'grid') {
    const xs = linspace(rx.x_start, rx.x_stop, rx.x_num)
    const ys = linspace(rx.y_start, rx.y_stop, rx.y_num)
    const out: Coord3[] = []
    for (const z of rx.z_values) for (const y of ys) for (const x of xs) out.push([x, y, z])
    return out
  }
  if (rx.method === 'radial' && rx.center_xy) {
    const angles = linspace(rx.angles_start * Math.PI / 180, rx.angles_stop * Math.PI / 180, rx.angles_num)
    const out: Coord3[] = []
    for (const z of rx.z_values) for (const r of rx.radii_m) for (const a of angles) {
      out.push([rx.center_xy[0] + r * Math.cos(a), rx.center_xy[1] + r * Math.sin(a), z])
    }
    return out
  }
  if (rx.method === 'street' && rx.path_points.length === 2) {
    const [s, e] = rx.path_points
    const ts = linspace(0, 1, rx.num_points)
    const out: Coord3[] = []
    for (const z of rx.z_values) for (const t of ts) out.push([s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1]), z])
    return out
  }
  return []
}

function linspace(a: number, b: number, n: number): number[] {
  if (n <= 1) return [a]
  const step = (b - a) / (n - 1)
  return Array.from({ length: n }, (_, i) => a + step * i)
}

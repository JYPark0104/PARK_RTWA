import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { Coord3, MaterialInfo, ExperimentPresetSummary } from '../lib/api'
import { apiClient, formatApiError } from '../lib/api'
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
  const setRT = useStore((s) => s.setRT)
  const antenna = useStore((s) => s.antenna)
  const setAntenna = useStore((s) => s.setAntenna)
  const setTXOrient = useStore((s) => s.setTXOrient)

  const [mode, setMode] = useState<Mode>('tx')
  const [gridBusy, setGridBusy] = useState(false)
  const [facadeBusy, setFacadeBusy] = useState(false)
  const [facadeCounting, setFacadeCounting] = useState(false)   // 예상 RX 계산 중 스피너
  const [showAntDir, setShowAntDir] = useState(false)          // 안테나 방향성 설정 패널
  const [orientTx, setOrientTx] = useState<number | null>(null) // 방향 조절 대상 TX
  const [gizmoScale, setGizmoScale] = useState(1)              // 방향 기즈모(화살표+구체) 크기 배율
  const [markerScale, setMarkerScale] = useState(1)   // 마커(구) 크기 배율 (시각용)
  const [selectedTx, setSelectedTx] = useState<number | null>(null)  // 리스트 클릭 선택(지속)
  const [hoverTx, setHoverTx] = useState<number | null>(null)        // 3D/리스트 호버(일시)
  const [showTxLabels, setShowTxLabels] = useState(true)             // 3D TX 번호 라벨 표시
  const [materials, setMaterials] = useState<MaterialInfo[]>([])  // 재질 특성 목록
  const [showLoad, setShowLoad] = useState(false)                 // 설정 불러오기 모달
  const [presets, setPresets] = useState<ExperimentPresetSummary[]>([])
  const [presetBusy, setPresetBusy] = useState(false)
  const [saveMsg, setSaveMsg] = useState<string | null>(null)
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

  async function previewFacade() {
    if (!session) return
    let cnt = rx.facade_count
    if (cnt == null) {
      try {
        const c = await apiClient.rxFacade(session.uuid, {
          z_min: rx.z_min, z_max: rx.z_max, z_distance: rx.z_distance,
          facade_spacing: rx.facade_spacing, facade_epsilon: rx.facade_epsilon, count_only: true,
          x_min: rx.facade_x_min, x_max: rx.facade_x_max,
          y_min: rx.facade_y_min, y_max: rx.facade_y_max,
        })
        cnt = c.count; setRX({ facade_count: c.count })
      } catch { /* ignore */ }
    }
    if (cnt != null && cnt > 20000) {
      const ok = window.confirm(`현재 ${cnt.toLocaleString()}개의 RX가 생성됩니다. 렌더링에 시간이 오래 걸리거나 브라우저가 느려질 수 있습니다. 띄우시겠습니까?`)
      if (!ok) return
    }
    setFacadeBusy(true)
    try {
      const r = await apiClient.rxFacade(session.uuid, {
        z_min: rx.z_min, z_max: rx.z_max, z_distance: rx.z_distance,
        facade_spacing: rx.facade_spacing, facade_epsilon: rx.facade_epsilon,
        x_min: rx.facade_x_min, x_max: rx.facade_x_max,
        y_min: rx.facade_y_min, y_max: rx.facade_y_max,
      })
      setRX({ facade_positions: r.positions ?? [], facade_count: r.count })
    } catch { /* ignore */ } finally { setFacadeBusy(false) }
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

  // 재질 특성 로드 (εr/σ 는 주파수 의존 → rt.frequency_ghz 변경 시 갱신).
  // 아직 값이 없는 재질은 산란계수 기본값(문헌/ITU=문헌, custom=XML값)으로 초기화.
  useEffect(() => {
    if (!session) return
    apiClient.sceneMaterials(session.uuid, rt.frequency_ghz).then((r) => {
      setMaterials(r.materials)
      const cur = useStore.getState().rt.material_scattering || {}
      const next: Record<string, number> = { ...cur }
      let changed = false
      for (const m of r.materials) {
        if (next[m.name] == null) { next[m.name] = defScat(m); changed = true }
      }
      if (changed) setRT({ material_scattering: next })
    }).catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.uuid, rt.frequency_ghz])

  const setMatS = (name: string, v: number) => {
    const cur = useStore.getState().rt.material_scattering || {}
    setRT({ material_scattering: { ...cur, [name]: v } })
  }
  const resetMatDefaults = () => {
    const next: Record<string, number> = {}
    for (const m of materials) next[m.name] = defScat(m)
    setRT({ material_scattering: next })
  }

  // ── 실험 설정(TX/안테나/RX) 저장·불러오기 ──
  async function saveSetting() {
    if (!session) return
    const name = window.prompt('이 실험 설정의 이름을 입력하세요',
      `${session.scene_name || 'scene'} 설정`)
    if (!name) return
    setPresetBusy(true); setSaveMsg(null)
    try {
      await apiClient.saveExperimentPreset({
        name, scene_name: session.scene_name || '',
        tx, antenna, rx,
        rt: {
          tx_pattern: rt.tx_pattern, tx_polarization: rt.tx_polarization,
          rx_pattern: rt.rx_pattern, rx_polarization: rt.rx_polarization,
          material_scattering: rt.material_scattering,
          scattering_pattern: rt.scattering_pattern,
          directive_alpha_r: rt.directive_alpha_r,
          backscattering_alpha_r: rt.backscattering_alpha_r,
          backscattering_alpha_i: rt.backscattering_alpha_i,
          backscattering_lambda: rt.backscattering_lambda,
          tx_ground_offset_m: rt.tx_ground_offset_m,
        },
      })
      setSaveMsg('✅ 설정을 저장했습니다.')
    } catch (e) { setSaveMsg('저장 실패: ' + formatApiError(e)) }
    finally { setPresetBusy(false) }
  }
  async function openLoad() {
    setShowLoad(true); setPresetBusy(true)
    try { const r = await apiClient.listExperimentPresets(); setPresets(r.presets) }
    catch { setPresets([]) } finally { setPresetBusy(false) }
  }
  async function applyPreset(id: string) {
    setPresetBusy(true)
    try {
      const d = await apiClient.getExperimentPreset(id)
      clearTX()
      for (const t of (d.tx || [])) addTXFull(t)
      if (d.antenna && Object.keys(d.antenna).length) setAntenna(d.antenna)
      if (d.rx && Object.keys(d.rx).length) setRX(d.rx)
      if (d.rt && Object.keys(d.rt).length) setRT(d.rt)
      setShowLoad(false)
      setSaveMsg(`📂 '${d.name}' 설정을 불러왔습니다.`)
    } catch { /* ignore */ } finally { setPresetBusy(false) }
  }
  async function deletePreset(id: string) {
    if (!window.confirm('이 설정을 삭제할까요?')) return
    try {
      await apiClient.deleteExperimentPreset(id)
      setPresets((ps) => ps.filter((x) => x.id !== id))
    } catch { /* ignore */ }
  }

  // ground_grid 가 선택되면 씬 경계 → 레이캐스팅 지면 격자를 자동 미리보기.
  // (PARK_2 방식: boundary → 하늘에서 ↓ ray casting → 지면 hit 위치에만 RX, 지면+높이)
  useEffect(() => {
    if (!session || !sceneInfo) return
    if (rx.method !== 'ground_grid') return
    const t = setTimeout(() => { previewGroundGrid() }, 250)  // 스윕 중 요청 폭주 방지(손 뗀 뒤 1회)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.uuid, sceneInfo, rx.method, rx.grid_n, rx.margin, rx.rx_height, rx.max_height,
      rx.x_start, rx.x_stop, rx.y_start, rx.y_stop])

  // facade(O2I): 파라미터 변경 시 이전 결과만 무효화(스테일 표시). 실제 개수 계산은 '벽면 RX 수 계산'
  //   버튼으로만 실행 → 슬라이더 드래그 중 수백번 백엔드 호출로 서버가 터지는 문제 방지. (2026-07-07)
  useEffect(() => {
    if (rx.method !== 'facade') return
    setRX({ facade_positions: [], facade_count: null })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rx.method, rx.z_min, rx.z_max, rx.z_distance, rx.facade_spacing, rx.facade_epsilon,
      rx.facade_x_min, rx.facade_x_max, rx.facade_y_min, rx.facade_y_max])

  async function countFacade() {
    if (!session) return
    setFacadeCounting(true)
    try {
      const r = await apiClient.rxFacade(session.uuid, {
        z_min: rx.z_min, z_max: rx.z_max, z_distance: rx.z_distance,
        facade_spacing: rx.facade_spacing, facade_epsilon: rx.facade_epsilon, count_only: true,
        x_min: rx.facade_x_min, x_max: rx.facade_x_max,
        y_min: rx.facade_y_min, y_max: rx.facade_y_max,
      })
      setRX({ facade_count: r.count })
    } catch { setRX({ facade_count: null }) }
    finally { setFacadeCounting(false) }
  }

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

  // facade(O2I) 진입 시 X/Y 경계가 bbox 밖이면 맵 전체로 초기화 (전용 필드)
  useEffect(() => {
    if (!sceneInfo) return
    if (rx.method !== 'facade') return
    const xMin = sceneInfo.aabb_min[0], xMax = sceneInfo.aabb_max[0]
    const yMin = sceneInfo.aabb_min[1], yMax = sceneInfo.aabb_max[1]
    const fix: any = {}
    if (rx.facade_x_min < xMin || rx.facade_x_min > xMax || rx.facade_x_max < xMin || rx.facade_x_max > xMax) {
      fix.facade_x_min = Math.round(xMin); fix.facade_x_max = Math.round(xMax)
    }
    if (rx.facade_y_min < yMin || rx.facade_y_min > yMax || rx.facade_y_max < yMin || rx.facade_y_max > yMax) {
      fix.facade_y_min = Math.round(yMin); fix.facade_y_max = Math.round(yMax)
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
          <label className="flex items-center gap-1.5 ml-auto text-xs text-slate-600 cursor-pointer">
            <input type="checkbox" checked={showTxLabels} onChange={(e) => setShowTxLabels(e.target.checked)} />
            TX 번호 라벨
          </label>
          <div className="flex items-center gap-2 text-xs text-slate-600">
            <span>마커 크기</span>
            <input type="range" min={0.3} max={5} step={0.1} value={markerScale}
              onChange={(e) => setMarkerScale(parseFloat(e.target.value))} className="w-28" />
            <span className="font-mono w-10">{markerScale.toFixed(1)}x</span>
          </div>
        </div>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <SceneViewer
            uuid={session.uuid}
            sceneInfo={sceneInfo}
            markerScale={markerScale}
            highlightTxIndex={hoverTx ?? selectedTx}
            onTxHover={setHoverTx}
            showTxLabels={showTxLabels}
            tx={tx}
            rx={rx.method === 'clicks' ? rx.click_positions : rx.method === 'ground_grid' ? rx.ground_positions : rx.method === 'facade' ? rx.facade_positions : rxPositions}
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
            facadeRegion={rx.method === 'facade' ? {
              x_min: rx.facade_x_min, x_max: rx.facade_x_max,
              y_min: rx.facade_y_min, y_max: rx.facade_y_max,
              z_min: rx.z_min, z_max: rx.z_max,
            } : undefined}
            regionOverlay={(rx.method === 'ground_grid' || rx.method === 'grid') ? {
              x_min: Math.min(rx.x_start, rx.x_stop), x_max: Math.max(rx.x_start, rx.x_stop),
              y_min: Math.min(rx.y_start, rx.y_stop), y_max: Math.max(rx.y_start, rx.y_stop),
              z: sceneInfo.aabb_min[2],
            } : undefined}
            orientTxIndex={showAntDir ? orientTx : null}
            onOrient={(idx, az, el) => setTXOrient(idx, az, el)}
            orientScale={gizmoScale}
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
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-sm">실험 설정</h3>
              <button className="btn btn-secondary text-xs px-2 py-1" onClick={openLoad} disabled={presetBusy}>
                📂 이전 실험 setting 불러오기
              </button>
            </div>
            {saveMsg && <div className="text-xs text-emerald-600 mt-1">{saveMsg}</div>}
          </section>

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

            <label className="flex justify-between items-center gap-2 text-sm">
              <span className="text-slate-600">TX 지면 이격 [m]</span>
              <input type="number" step="any" className="input w-24 text-right"
                value={rt.tx_ground_offset_m}
                onChange={(e) => { const v = parseFloat(e.target.value); if (!isNaN(v)) setRT({ tx_ground_offset_m: v }) }} />
            </label>
            <div className="text-[11px] text-slate-400 mb-2 mt-0.5">TX 클릭/좌표 추가 시 지면으로부터 이 높이만큼 띄워 배치합니다.</div>

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
              {tx.map((t, i) => {
                const on = i === selectedTx || i === hoverTx
                return (
                  <li key={i}
                    className={`flex justify-between items-center border-b py-1 px-1 rounded cursor-pointer ${on ? 'bg-amber-100 ring-1 ring-amber-400' : 'hover:bg-slate-50'}`}
                    onClick={() => setSelectedTx(selectedTx === i ? null : i)}
                    onMouseEnter={() => setHoverTx(i)}
                    onMouseLeave={() => setHoverTx(null)}>
                    <span className="flex items-center gap-2 min-w-0">
                      <span className={`inline-flex items-center justify-center shrink-0 w-6 h-5 rounded text-[10px] font-bold ${on ? 'bg-amber-400 text-white' : 'bg-slate-200 text-slate-600'}`}>TX{i + 1}</span>
                      <span className="font-mono text-xs truncate">[{t.position.map((v) => v.toFixed(2)).join(', ')}]</span>
                    </span>
                    <button className="btn btn-danger text-xs px-2 py-0.5 shrink-0"
                      onClick={(e) => { e.stopPropagation(); removeTX(i) }}>삭제</button>
                  </li>
                )
              })}
              {tx.length === 0 && <li className="text-slate-400 text-xs">TX 모드에서 표면을 클릭하세요.</li>}
            </ul>
          </section>

          <section className="card space-y-2 text-sm">
            <h3 className="font-semibold">안테나 설정</h3>
            <div className="flex gap-2">
              <button className="btn btn-secondary text-xs" onClick={() => setAntenna({ bs_rows: 1, bs_cols: 1, ue_rows: 1, ue_cols: 1 })}>SISO (1×1/1×1)</button>
              <button className="btn btn-secondary text-xs" onClick={() => setAntenna({ bs_rows: 32, bs_cols: 32, ue_rows: 4, ue_cols: 4 })}>기본 (32×32/4×4)</button>
            </div>
            <NumberRow label="BS rows" v={antenna.bs_rows} on={(v) => setAntenna({ bs_rows: Math.max(1, Math.round(v)) })} integer />
            <NumberRow label="BS cols" v={antenna.bs_cols} on={(v) => setAntenna({ bs_cols: Math.max(1, Math.round(v)) })} integer />
            <NumberRow label="UE rows" v={antenna.ue_rows} on={(v) => setAntenna({ ue_rows: Math.max(1, Math.round(v)) })} integer />
            <NumberRow label="UE cols" v={antenna.ue_cols} on={(v) => setAntenna({ ue_cols: Math.max(1, Math.round(v)) })} integer />
            <SelectInput label="tx_pattern" value={rt.tx_pattern} on={(v) => setRT({ tx_pattern: v })} options={PATTERN_OPTIONS} />
            <SelectInput label="tx_polarization" value={rt.tx_polarization} on={(v) => setRT({ tx_polarization: v })} options={POLARIZATION_OPTIONS} />
            <SelectInput label="rx_pattern" value={rt.rx_pattern} on={(v) => setRT({ rx_pattern: v })} options={PATTERN_OPTIONS} />
            <SelectInput label="rx_polarization" value={rt.rx_polarization} on={(v) => setRT({ rx_polarization: v })} options={POLARIZATION_OPTIONS} />
            {rt.tx_pattern !== 'iso' ? (
              <div className="space-y-2">
                <button className={`btn text-xs ${showAntDir ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => setShowAntDir((v) => !v)}>
                  {showAntDir ? '안테나 방향성 설정 닫기' : '📡 안테나 방향성 설정'}
                </button>
                {showAntDir && (
                  <div className="p-2 rounded border border-slate-200 bg-slate-50 space-y-2">
                    <div className="text-xs text-slate-600">
                      TX 선택 후 3D 뷰에서 방향 조절: <b>WASD</b>(A/D=방위, W/S=고각) 또는 화살표(반투명 구체) <b>드래그</b>.
                    </div>
                    <label className="flex items-center gap-2 text-xs text-slate-600">
                      <span className="whitespace-nowrap">화살표 크기</span>
                      <input type="range" min={0.1} max={3} step={0.1} value={gizmoScale}
                        onChange={(e) => setGizmoScale(parseFloat(e.target.value))} className="flex-1" />
                      <span className="font-mono w-8">{gizmoScale.toFixed(1)}x</span>
                    </label>
                    <div className="flex flex-wrap gap-1">
                      {tx.map((_, i) => (
                        <button key={i} className={`btn text-xs ${orientTx === i ? 'btn-primary' : 'btn-secondary'}`}
                          onClick={() => setOrientTx(orientTx === i ? null : i)}>TX{i + 1}</button>
                      ))}
                      {tx.length === 0 && <span className="text-xs text-slate-400">먼저 TX를 배치하세요.</span>}
                    </div>
                    {orientTx != null && tx[orientTx] && (
                      <div className="text-xs font-mono text-slate-600">
                        TX{orientTx + 1}: azimuth {(tx[orientTx].az_deg ?? 0).toFixed(0)}° · elevation {(tx[orientTx].el_deg ?? 0).toFixed(0)}°
                        <button className="btn btn-secondary text-[11px] ml-2 px-2 py-0.5" onClick={() => setTXOrient(orientTx, 0, 0)}>리셋</button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="text-[11px] text-slate-400">iso(등방성)은 방향성이 없어 방향 설정이 불필요합니다.</div>
            )}
          </section>

          <section className="card">
            <h3 className="font-semibold mb-2">RX 배치 방법</h3>
            <div className="flex flex-wrap gap-1 mb-3">
              {(['grid', 'ground_grid', 'facade', 'explicit', 'radial', 'street', 'clicks'] as const).map((m) => (
                <button key={m}
                  className={`btn text-xs ${rx.method === m ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => setRX({ method: m })}>{m === 'facade' ? 'O2I(건물벽면)' : m}</button>
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
                    <div className="text-[11px] text-slate-400 -mt-1">X·Y 동일 간격(m)으로 격자 배치. 총 후보가 100만 개를 넘을 만큼 촘촘하면 자동으로 간격을 상향합니다(그 이하면 입력 간격 그대로).</div>
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

            {rx.method === 'facade' && (
              <div className="space-y-2 text-sm">
                <div className="text-xs text-slate-500">
                  건물 외벽(수직면)에 RX를 배치합니다. z=k 평면과 건물 메시의 교선을 따라
                  벽 바깥으로 ε(m) 띄워 배치하며, host 건물·재질·법선을 기록합니다 (O2I penetration용).
                </div>
                <div>
                  <div className="flex justify-between text-xs text-slate-600">
                    <span>X 영역 (벽면 RX 대상)</span>
                    <span className="font-mono">{(rx.facade_x_min ?? sceneInfo.aabb_min[0]).toFixed(0)} ~ {(rx.facade_x_max ?? sceneInfo.aabb_max[0]).toFixed(0)} m</span>
                  </div>
                  <DualRange
                    min={Math.floor(sceneInfo.aabb_min[0])} max={Math.ceil(sceneInfo.aabb_max[0])}
                    lo={rx.facade_x_min ?? Math.floor(sceneInfo.aabb_min[0])} hi={rx.facade_x_max ?? Math.ceil(sceneInfo.aabb_max[0])}
                    onChange={(lo, hi) => setRX({ facade_x_min: lo, facade_x_max: hi })}
                  />
                </div>
                <div>
                  <div className="flex justify-between text-xs text-slate-600">
                    <span>Y 영역 (벽면 RX 대상)</span>
                    <span className="font-mono">{(rx.facade_y_min ?? sceneInfo.aabb_min[1]).toFixed(0)} ~ {(rx.facade_y_max ?? sceneInfo.aabb_max[1]).toFixed(0)} m</span>
                  </div>
                  <DualRange
                    min={Math.floor(sceneInfo.aabb_min[1])} max={Math.ceil(sceneInfo.aabb_max[1])}
                    lo={rx.facade_y_min ?? Math.floor(sceneInfo.aabb_min[1])} hi={rx.facade_y_max ?? Math.ceil(sceneInfo.aabb_max[1])}
                    onChange={(lo, hi) => setRX({ facade_y_min: lo, facade_y_max: hi })}
                  />
                </div>
                <button className="btn btn-secondary text-xs" onClick={() => setRX({
                  facade_x_min: Math.round(sceneInfo.aabb_min[0]), facade_x_max: Math.round(sceneInfo.aabb_max[0]),
                  facade_y_min: Math.round(sceneInfo.aabb_min[1]), facade_y_max: Math.round(sceneInfo.aabb_max[1]),
                })}>맵 전체(bbox)</button>
                <div className="text-[11px] text-slate-400 -mt-1">반투명 직육면체 = 이 영역·높이대(z 최저~최대)에 벽면 RX가 생성됩니다.</div>
                <NumberRow label="z 최저 [m]" v={rx.z_min} on={(v) => setRX({ z_min: v })} />
                <NumberRow label="z 최대 [m]" v={rx.z_max} on={(v) => setRX({ z_max: v })} />
                <NumberRow label="z 간격 [m]" v={rx.z_distance} on={(v) => setRX({ z_distance: Math.max(0.1, v) })} />
                <NumberRow label="벽면 RX 간격 [m]" v={rx.facade_spacing} on={(v) => setRX({ facade_spacing: Math.max(0.1, v) })} />
                <NumberRow label="바깥 이격 ε [m]" v={rx.facade_epsilon} on={(v) => setRX({ facade_epsilon: Math.max(0, v) })} />
                <div className="flex gap-2">
                  <button className="btn btn-secondary text-xs" onClick={countFacade} disabled={facadeCounting}>
                    {facadeCounting ? '계산 중...' : '벽면 RX 수 계산'}
                  </button>
                  <button className="btn btn-secondary text-xs" onClick={previewFacade} disabled={facadeBusy}>
                    {facadeBusy ? '계산 중...' : '벽면 RX 미리보기'}
                  </button>
                </div>
                <div className="text-xs text-slate-500">
                  높이층: <span className="font-mono">{Math.max(0, Math.floor((rx.z_max - rx.z_min) / Math.max(0.1, rx.z_distance)) + 1)}</span>개 ·
                  예상 RX: <span className={`font-mono ${(rx.facade_count ?? 0) > 20000 ? 'text-amber-600 font-semibold' : ''}`}>{rx.facade_count == null ? '—' : rx.facade_count.toLocaleString()}</span>개
                  {facadeCounting && <span className="inline-block ml-1.5 w-3 h-3 border-2 border-slate-300 border-t-sky-600 rounded-full animate-spin align-[-1px]" title="계산 중" />}
                  {rx.facade_positions.length > 0 && <> · 렌더됨: <span className="font-mono">{rx.facade_positions.length.toLocaleString()}</span>개</>}
                </div>
                <div className="text-[11px] text-slate-400 -mt-1">
                  값 변경 시 개수는 초기화됩니다(과부하 방지). '벽면 RX 수 계산'으로 개수를 구하고,
                  '벽면 RX 미리보기'로 실제 배치를 표시하세요. (2만 개 초과 시 경고)
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

          <button className="btn btn-secondary w-full mb-2" onClick={saveSetting} disabled={presetBusy}>
            💾 이 실험 setting 저장하기
          </button>
          <button className="btn btn-primary w-full" onClick={goNext}>
            다음: RT 설정
          </button>
        </div>
      </div>

      {/* ── Material Properties (재질 특성) — 페이지 맨 아래, 전체 너비 ── */}
      <section className="card">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
          <h3 className="text-lg font-semibold">Material Properties (재질 특성)</h3>
          <div className="flex items-center gap-3 text-xs text-slate-500">
            <span>εr·σ 는 <b>{rt.frequency_ghz} GHz</b> 기준 (4.RT의 주파수와 연동)</span>
            <button className="btn btn-secondary text-xs px-2 py-1" onClick={resetMatDefaults}>문헌 기본값으로 리셋</button>
          </div>
        </div>
        <p className="text-xs text-slate-500 mb-3 leading-relaxed">
          εr(유전율)·σ(전도율)은 ITU-R P.2040 재질 클래스가 <b>주파수로부터 자동 계산</b>합니다(수정 불가).
          산란계수 <b>S</b>는 표면 거칠기 파라미터로 ITU 표준에 정의되지 않아 <b>사용자가 지정</b>합니다.
          기본값은 문헌 근사치(콘크리트 0.4·유리 0.2 등)이며, 재질마다 다르게 조정할 수 있습니다.
          커스텀 재질(예: irr_glass)의 εr/σ는 scene.xml에 명시된 값을 그대로 사용합니다.
        </p>
        <div className="flex flex-wrap items-end gap-3 mb-3 pb-3 border-b">
          <label className="text-sm">
            <span className="block text-xs text-slate-600 mb-0.5">산란 패턴 (scattering pattern · 전역)</span>
            <select className="input w-40" value={rt.scattering_pattern}
              onChange={(e) => setRT({ scattering_pattern: e.target.value as any })}>
              <option value="lambertian">lambertian</option>
              <option value="directive">directive</option>
              <option value="backscattering">backscattering</option>
            </select>
          </label>
          {rt.scattering_pattern === 'directive' && (
            <label className="text-sm">
              <span className="block text-xs text-slate-600 mb-0.5">directive_alpha_r</span>
              <input type="number" className="input w-24" value={rt.directive_alpha_r}
                onChange={(e) => { const v = parseInt(e.target.value); if (!isNaN(v)) setRT({ directive_alpha_r: v }) }} />
            </label>
          )}
          {rt.scattering_pattern === 'backscattering' && (
            <>
              <label className="text-sm">
                <span className="block text-xs text-slate-600 mb-0.5">backscat_alpha_r</span>
                <input type="number" className="input w-24" value={rt.backscattering_alpha_r}
                  onChange={(e) => { const v = parseInt(e.target.value); if (!isNaN(v)) setRT({ backscattering_alpha_r: v }) }} />
              </label>
              <label className="text-sm">
                <span className="block text-xs text-slate-600 mb-0.5">backscat_alpha_i</span>
                <input type="number" className="input w-24" value={rt.backscattering_alpha_i}
                  onChange={(e) => { const v = parseInt(e.target.value); if (!isNaN(v)) setRT({ backscattering_alpha_i: v }) }} />
              </label>
              <label className="text-sm">
                <span className="block text-xs text-slate-600 mb-0.5">backscat_lambda</span>
                <input type="number" step="any" className="input w-24" value={rt.backscattering_lambda}
                  onChange={(e) => { const v = parseFloat(e.target.value); if (!isNaN(v)) setRT({ backscattering_lambda: v }) }} />
              </label>
            </>
          )}
        </div>
        {materials.length === 0 ? (
          <div className="text-sm text-slate-400">재질 정보를 불러오는 중이거나 씬에 재질이 없습니다.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-slate-500 border-b">
                  <th className="py-1.5 pr-3">재질</th>
                  <th className="py-1.5 pr-3">종류</th>
                  <th className="py-1.5 pr-3 text-right">shape 수</th>
                  <th className="py-1.5 pr-3 text-right">εr</th>
                  <th className="py-1.5 pr-3 text-right">σ [S/m]</th>
                  <th className="py-1.5 pr-3 text-right">XPD</th>
                  <th className="py-1.5 pr-3">산란계수 S (거칠기)</th>
                </tr>
              </thead>
              <tbody>
                {materials.map((m) => (
                  <tr key={m.name} className="border-b last:border-0">
                    <td className="py-1.5 pr-3 font-mono">
                      {m.name}
                      {m.kind === 'custom' && <span className="ml-1 text-[10px] text-amber-600">(custom)</span>}
                    </td>
                    <td className="py-1.5 pr-3 text-slate-500">{m.kind === 'itu' ? 'ITU' : 'Custom'}</td>
                    <td className="py-1.5 pr-3 text-right font-mono">{m.shape_count.toLocaleString()}</td>
                    <td className="py-1.5 pr-3 text-right font-mono">{m.eps_r != null ? m.eps_r.toFixed(3) : '—'}</td>
                    <td className="py-1.5 pr-3 text-right font-mono">{m.sigma != null ? m.sigma.toFixed(4) : '—'}</td>
                    <td className="py-1.5 pr-3 text-right font-mono">
                      {m.xpd_coefficient != null ? m.xpd_coefficient.toFixed(2) : (m.kind === 'itu' ? rt.itu_xpd_coeff.toFixed(2) : '—')}
                    </td>
                    <td className="py-1.5 pr-3">
                      <div className="flex items-center gap-2">
                        <input type="number" min={0} max={0.99} step={0.05}
                          className="input w-24 text-right py-0.5"
                          value={rt.material_scattering[m.name] ?? defScat(m)}
                          onChange={(e) => {
                            const v = parseFloat(e.target.value)
                            if (!isNaN(v)) setMatS(m.name, Math.min(0.99, Math.max(0, v)))
                          }} />
                        <span className="text-[10px] text-slate-400">기본 {defScat(m)}</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="text-[11px] text-slate-400 mt-2">
              S=0 → 완전 정반사(거울), S→1 → 확산 산란 강함. 반사 에너지를 정반사(√(1−S²)) vs 확산(S)으로 분배합니다.
              값 0 은 산란 없음(정반사만)으로 처리됩니다.
            </div>
          </div>
        )}
      </section>

      {showLoad && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
             onClick={() => setShowLoad(false)}>
          <div className="card max-w-3xl w-full mx-4 bg-white shadow-xl max-h-[80vh] overflow-auto"
               onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-semibold">이전 실험 setting 불러오기</h3>
              <button className="btn btn-secondary text-xs px-2 py-1" onClick={() => setShowLoad(false)}>닫기</button>
            </div>
            {presetBusy && <div className="text-xs text-slate-400 mb-2">불러오는 중...</div>}
            {presets.length === 0 ? (
              <div className="text-sm text-slate-400">저장된 설정이 없습니다.</div>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-slate-500 border-b">
                    <th className="py-1 pr-2">이름</th>
                    <th className="py-1 pr-2">Scene</th>
                    <th className="py-1 pr-2 text-right">TX</th>
                    <th className="py-1 pr-2">RX 방법</th>
                    <th className="py-1 pr-2">안테나(BS/UE)</th>
                    <th className="py-1 pr-2">저장시각</th>
                    <th className="py-1 pr-2"></th>
                  </tr>
                </thead>
                <tbody>
                  {presets.map((p) => (
                    <tr key={p.id} className="border-b last:border-0">
                      <td className="py-1.5 pr-2 font-medium">{p.name}</td>
                      <td className="py-1.5 pr-2 font-mono text-xs">{p.scene_name || '-'}</td>
                      <td className="py-1.5 pr-2 text-right font-mono">{p.n_tx}</td>
                      <td className="py-1.5 pr-2">{p.rx_method || '-'}</td>
                      <td className="py-1.5 pr-2 font-mono text-xs">
                        {p.antenna?.bs_rows ?? '?'}x{p.antenna?.bs_cols ?? '?'} / {p.antenna?.ue_rows ?? '?'}x{p.antenna?.ue_cols ?? '?'}
                      </td>
                      <td className="py-1.5 pr-2 text-xs text-slate-500">{p.created_at}</td>
                      <td className="py-1.5 pr-2 whitespace-nowrap">
                        <button className="btn btn-primary text-xs px-2 py-0.5 mr-1"
                          onClick={() => applyPreset(p.id)} disabled={presetBusy}>불러오기</button>
                        <button className="btn btn-danger text-xs px-2 py-0.5"
                          onClick={() => deletePreset(p.id)}>삭제</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}

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

const PATTERN_OPTIONS = ['iso', 'dipole', 'hw_dipole', 'tr38901']
const POLARIZATION_OPTIONS = ['V', 'H', 'VH', 'cross']

function SelectInput({ label, value, on, options }: {
  label: string; value: string; on: (v: string) => void; options: string[]
}) {
  const opts = options.includes(value) ? options : [value, ...options]
  return (
    <label className="flex justify-between items-center gap-2">
      <span className="text-slate-600">{label}</span>
      <select className="input w-40 text-right" value={value} onChange={(e) => on(e.target.value)}>
        {opts.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </label>
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

function defScat(m: MaterialInfo): number {
  return (m.kind === 'custom' && m.scattering_xml != null) ? m.scattering_xml : m.scattering_default
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

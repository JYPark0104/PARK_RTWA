import { Component, Suspense, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Canvas, useFrame, useLoader } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { apiClient } from '../lib/api'
import { useStore } from '../store/useStore'
import { ChannelStatePanel, type ChItem } from '../components/ChannelStatePanel'

type Ray = { pts: number[][]; los: boolean; width: number }
type Step = { name: string; pos: number[]; rays: Ray[] }

// 9.Scenario Results 표시 설정 (localStorage 'rtagent_viz')
type Viz = {
  rayThickness: number; rayFixed: boolean; rayFixedRadius: number; rayOpacity: number
  losColor: string; nlosColor: string
  pathColor: string; pathThickness: number; pathOpacity: number
  txColor: string; txSize: number; rxColor: string; rxSize: number
}
const VIZ_DEFAULT: Viz = {
  rayThickness: 1, rayFixed: false, rayFixedRadius: 0.4, rayOpacity: 1,
  losColor: '#3399ff', nlosColor: '#ffbf00',
  pathColor: '#b6f36a', pathThickness: 1, pathOpacity: 1,
  txColor: '#ff0000', txSize: 1, rxColor: '#22ff44', rxSize: 1,
}
function loadViz(): Viz {
  try { const r = localStorage.getItem('rtagent_viz'); return r ? { ...VIZ_DEFAULT, ...JSON.parse(r) } : VIZ_DEFAULT }
  catch { return VIZ_DEFAULT }
}

/**
 * 9. Scenario Results — blender_mobility.py 다운로드 + 사용법 + 웹 3D 키프레임 미리보기.
 * (Blender 없이 동일한 scenario_data 를 브라우저에서 직접 재생 = 검토 Q5(b))
 */
export function ScenarioResultsPage() {
  const session = useStore((s) => s.session)
  const result = useStore((s) => s.scenarioResult)
  const setScenarioResult = useStore((s) => s.setScenarioResult)

  const [playing, setPlaying] = useState(true)
  const [step, setStep] = useState(0)
  const [restoring, setRestoring] = useState(false)
  const [mapLoaded, setMapLoaded] = useState(false)
  const [playFps, setPlayFps] = useState<number | null>(null)   // 재생 속도(override). null=시나리오 기본값
  const [viz, setViz] = useState<Viz>(loadViz)
  useEffect(() => { try { localStorage.setItem('rtagent_viz', JSON.stringify(viz)) } catch { /* ignore */ } }, [viz])
  const [ch, setCh] = useState<(ChItem | undefined)[]>([])      // step별 채널상태 (병렬 로딩)
  const [chLoaded, setChLoaded] = useState(0)
  const [chTotal, setChTotal] = useState(0)
  const [chLoading, setChLoading] = useState(false)

  // 채널 상태(RSRP/PADP/Cov)를 경로 청크 단위로 병렬 로딩 (RT 키프레임은 먼저 재생됨)
  useEffect(() => {
    if (!session || !result || result.session_uuid !== session.uuid) return
    let cancelled = false
    setCh([]); setChLoaded(0); setChTotal(0); setChLoading(true)
    const CHUNK = 20
    ;(async () => {
      let start = 0, total = Infinity
      const acc: (ChItem | undefined)[] = []
      while (!cancelled && start < total) {
        try {
          const r = await apiClient.scenarioChannelState(session.uuid, start, CHUNK)
          if (cancelled) break
          total = r.total; setChTotal(total)
          r.items.forEach((it, k) => { acc[start + k] = it as ChItem })
          setCh([...acc])
          setChLoaded(Math.min(start + r.items.length, total))
          if (r.items.length === 0) break
          start += r.items.length
        } catch { break }
      }
      if (!cancelled) setChLoading(false)
    })()
    return () => { cancelled = true }
  }, [session?.uuid, result?.session_uuid])
  const [skin, setSkin] = useState<{ has_skin: boolean; url?: string; name?: string }>({ has_skin: false })
  const [textured, setTextured] = useState(false)   // false=회색 메시, true=텍스처 GLB

  // 세션의 시각화 스킨(텍스처 GLB) 유무 조회
  useEffect(() => {
    if (!session) { setSkin({ has_skin: false }); setTextured(false); return }
    let cancelled = false
    apiClient.sessionSkin(session.uuid)
      .then((s) => { if (!cancelled) setSkin(s) })
      .catch(() => { if (!cancelled) setSkin({ has_skin: false }) })
    return () => { cancelled = true }
  }, [session?.uuid])

  // 회색↔텍스처 전환 시 로딩 오버레이 리셋
  useEffect(() => { setMapLoaded(false) }, [textured])

  // 세션을 다시 열거나 새로고침해도 저장된 시나리오를 백엔드에서 복원.
  useEffect(() => {
    if (!session) return
    if (result && result.session_uuid === session.uuid) return
    let cancelled = false
    setRestoring(true)
    apiClient.scenarioResult(session.uuid)
      .then((r) => {
        if (cancelled) return
        if (r && r.exists) setScenarioResult({ ...r, session_uuid: session.uuid })
        else if (result && result.session_uuid !== session.uuid) setScenarioResult(null)
      })
      .catch(() => {})
      .finally(() => { if (!cancelled) setRestoring(false) })
    return () => { cancelled = true }
  }, [session?.uuid, result?.session_uuid])

  if (!session) return <div className="card">먼저 세션을 선택하세요.</div>
  if (!result) {
    return (
      <div className="card text-sm text-slate-500">
        {restoring
          ? '저장된 시나리오를 불러오는 중...'
          : '아직 생성된 시나리오가 없습니다. 8.Scenario 에서 경로를 그려 "시나리오 생성"을 누르세요.'}
      </div>
    )
  }

  const preview = result.preview as {
    tx_name: string; tx_pos: number[]; fps: number; frames_per_step: number
    total_frames: number; steps: Step[]
  }
  const steps = preview.steps ?? []
  const dlUrl = apiClient.fileUrl(session.uuid, result.script_rel)
  const unrealRel = (result as any).unreal_script_rel as string | undefined
  const unrealDlUrl = unrealRel ? apiClient.fileUrl(session.uuid, unrealRel) : null
  const effFps = playFps ?? preview.fps ?? 30
  const secPerStep = (preview.frames_per_step || 3) / Math.max(effFps, 0.5)

  return (
    <div className="space-y-4">
      <section className="card">
        <div className="flex justify-between items-center mb-2">
          <h2 className="text-lg font-semibold">9. Scenario Results</h2>
          <div className="flex items-center gap-2">
            <a className="btn btn-primary" href={dlUrl} download="blender_mobility.py">
              blender_mobility.py 다운로드
            </a>
            {unrealDlUrl && (
              <a className="btn btn-secondary" href={unrealDlUrl} download="unreal_scenario.py">
                unreal_scenario.py 다운로드
              </a>
            )}
          </div>
        </div>
        <div className="text-sm grid grid-cols-2 md:grid-cols-4 gap-2">
          <div>TX: <span className="font-mono">{result.tx_name}</span></div>
          <div>RX 경로: <span className="font-mono">{result.num_rx}개</span></div>
          <div>Ray: <span className="font-mono">{result.num_rays}개</span></div>
          <div>총 프레임: <span className="font-mono">{preview.total_frames} ({preview.fps} FPS)</span></div>
        </div>
        <details className="mt-2 text-sm">
          <summary className="cursor-pointer text-slate-600 hover:text-slate-900">도움말 : Blender 사용법</summary>
          <ol className="list-decimal pl-5 space-y-1 text-slate-700 text-xs mt-2">
            <li>위 버튼으로 <b>blender_mobility.py</b> 다운로드</li>
            <li>(선택) Blender에서 File → Import → Wavefront OBJ → <span className="font-mono">Map_Mesh.obj</span> (배경 지도)</li>
            <li><b>Scripting</b> 탭 → Open → blender_mobility.py → <b>▶ Run Script</b></li>
            <li><b>Layout</b> 탭 → <b>Space bar</b> 로 재생 (키프레임 흐름)</li>
            <li>색이 안 보이면 Z키 → Material Preview / Rendered</li>
          </ol>
          <p className="text-xs text-slate-500 mt-1">
            이 스크립트는 USDA 통째 import 없이, 선택한 TX/RX/Ray 만 생성하는 자체 완결형입니다.
          </p>
        </details>
        <details className="mt-2 text-sm">
          <summary className="cursor-pointer text-slate-600 hover:text-slate-900">도움말 : Unreal Engine 사용법</summary>
          <ol className="list-decimal pl-5 space-y-1 text-slate-700 text-xs mt-2">
            <li>위 버튼으로 <b>unreal_scenario.py</b> 다운로드 (데이터 포함, 파일 하나)</li>
            <li>Unreal에서 도시 메시(.glb 등)를 먼저 임포트해 배치</li>
            <li>플러그인: Edit → Plugins → <b>Python Editor Script</b> + <b>Sequencer Scripting</b> 활성화</li>
            <li><b>Tools → Execute Python Script...</b> → unreal_scenario.py 선택</li>
            <li>콘텐츠 브라우저 <span className="font-mono">RT_Rays / SEQ_RayPlayback</span> 더블클릭 → <b>재생</b></li>
            <li>도시와 안 맞으면 파일 상단 <span className="font-mono">FLIP_Y</span>(기본 True)/<span className="font-mono">FLIP_X</span>/<span className="font-mono">SCALE</span> 조정 후 재실행</li>
            <li>재생이 반대(전부 보임/숨김)면 <span className="font-mono">VIS_KEY_VISIBLE=False</span></li>
          </ol>
          <p className="text-xs text-slate-500 mt-1">
            blender_mobility.py 와 동일한 데이터를 Unreal에서 재생 - 파일 하나로 딸깍 실행.
          </p>
        </details>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        <section className="card lg:col-span-3">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-semibold">웹 미리보기 (키프레임 재생)</h3>
            <div className="flex items-center gap-2 text-sm">
              {skin.has_skin && (
                <button
                  className={`btn text-xs ${textured ? 'btn-primary' : 'btn-secondary'}`}
                  title="회색 기하 메시 ↔ 텍스처 GLB 맵 전환 (RT 결과는 동일, 시각화만 변경)"
                  onClick={() => setTextured((v) => !v)}>
                  {textured ? '🎨 텍스처 맵' : '◻ 회색 메시'}
                </button>
              )}
              <button className="btn btn-secondary text-xs" onClick={() => setPlaying((p) => !p)}>
                {playing ? '⏸ 일시정지' : '▶ 재생'}
              </button>
              <div className="flex items-center gap-1">
                <span className="text-xs text-slate-500">속도</span>
                <input type="range" min={1} max={60} step={1} value={effFps}
                  title="재생 FPS (낮을수록 느리게)"
                  onChange={(e) => setPlayFps(parseInt(e.target.value))}
                  className="w-24" />
                <span className="font-mono text-xs text-slate-600 w-14">{effFps} FPS</span>
              </div>
              <span className="font-mono text-slate-600">step {step + 1}/{steps.length}</span>
            </div>
          </div>
          <div className="h-[560px] bg-slate-950 rounded overflow-hidden relative">
            {!mapLoaded && (
              <div className="absolute top-2 left-1/2 -translate-x-1/2 z-10 px-3 py-1.5 rounded-full
                              bg-slate-800/85 text-slate-100 text-xs font-medium shadow flex items-center gap-2 pointer-events-none">
                <span className="inline-block w-3 h-3 rounded-full border-2 border-slate-400 border-t-emerald-400 animate-spin" />
                🗺 {session.scene_name || 'scene'} {textured ? '텍스처 맵(skin)' : '3D'} 로딩 중 …{textured ? ' 대용량이라 시간이 걸립니다' : ''}
              </div>
            )}
            <ScenarioViewer
              uuid={session.uuid}
              steps={steps}
              txPos={preview.tx_pos}
              step={step}
              playing={playing}
              secPerStep={secPerStep}
              onStep={setStep}
              onMapLoaded={() => setMapLoaded(true)}
              skinUrl={textured && skin.has_skin ? skin.url : undefined}
              viz={viz}
            />
          </div>
          <input
            type="range" min={0} max={Math.max(steps.length - 1, 0)} value={step}
            onChange={(e) => { setPlaying(false); setStep(parseInt(e.target.value)) }}
            className="w-full mt-2"
          />
          <details className="mt-2 text-sm">
            <summary className="cursor-pointer text-slate-600 hover:text-slate-900">🎨 표시 설정 (Ray / 경로 / TX·RX)</summary>
            <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-x-5 gap-y-2 text-xs">
              <label className="flex items-center gap-2 md:col-span-2">
                <input type="checkbox" checked={viz.rayFixed} onChange={(e) => setViz((v) => ({ ...v, rayFixed: e.target.checked }))} />
                Ray 굵기 고정 (power 차등 무시)
              </label>
              {viz.rayFixed
                ? <RangeRow label="Ray 고정 반지름" min={0.05} max={2} step={0.05} value={viz.rayFixedRadius} unit="m" onChange={(x) => setViz((v) => ({ ...v, rayFixedRadius: x }))} />
                : <RangeRow label="Ray 굵기 배율" min={0.2} max={5} step={0.1} value={viz.rayThickness} unit="x" onChange={(x) => setViz((v) => ({ ...v, rayThickness: x }))} />}
              <RangeRow label="Ray 투명도" min={0.1} max={1} step={0.05} value={viz.rayOpacity} onChange={(x) => setViz((v) => ({ ...v, rayOpacity: x }))} />
              <ColorRow label="LoS 색" value={viz.losColor} onChange={(c) => setViz((v) => ({ ...v, losColor: c }))} />
              <ColorRow label="NLoS 색" value={viz.nlosColor} onChange={(c) => setViz((v) => ({ ...v, nlosColor: c }))} />
              <ColorRow label="이동경로 색" value={viz.pathColor} onChange={(c) => setViz((v) => ({ ...v, pathColor: c }))} />
              <RangeRow label="경로 굵기" min={0.2} max={5} step={0.1} value={viz.pathThickness} unit="x" onChange={(x) => setViz((v) => ({ ...v, pathThickness: x }))} />
              <RangeRow label="경로 투명도" min={0.1} max={1} step={0.05} value={viz.pathOpacity} onChange={(x) => setViz((v) => ({ ...v, pathOpacity: x }))} />
              <ColorRow label="TX 색" value={viz.txColor} onChange={(c) => setViz((v) => ({ ...v, txColor: c }))} />
              <RangeRow label="TX 크기" min={0.3} max={4} step={0.1} value={viz.txSize} unit="x" onChange={(x) => setViz((v) => ({ ...v, txSize: x }))} />
              <ColorRow label="RX 색" value={viz.rxColor} onChange={(c) => setViz((v) => ({ ...v, rxColor: c }))} />
              <RangeRow label="RX 크기" min={0.3} max={4} step={0.1} value={viz.rxSize} unit="x" onChange={(x) => setViz((v) => ({ ...v, rxSize: x }))} />
              <button className="btn btn-secondary text-xs md:col-span-2" onClick={() => setViz(VIZ_DEFAULT)}>기본값으로 초기화</button>
            </div>
          </details>
        </section>

        <section className="card text-sm space-y-2">
          {(chLoading || (chTotal > 0 && chLoaded < chTotal)) && (
            <div>
              <div className="flex justify-between text-[10px] text-slate-500 mb-0.5">
                <span>채널 상태 로딩{chTotal === 0 ? ' (채널 데이터 여는 중…)' : ''}</span>
                <span>{chTotal > 0 ? `${chLoaded}/${chTotal}` : ''}</span>
              </div>
              <div className="w-full bg-slate-200 rounded h-1.5 overflow-hidden">
                {chTotal > 0 ? (
                  <div className="bg-emerald-500 h-1.5 transition-all"
                    style={{ width: `${(chLoaded / chTotal) * 100}%` }} />
                ) : (
                  // 총량 미정(npz 여는 중) → 불확정 펄스 바
                  <div className="bg-emerald-500 h-1.5 animate-pulse" style={{ width: '40%' }} />
                )}
              </div>
            </div>
          )}
          <ChannelStatePanel
            item={ch[step]}
            loading={chLoading || chLoaded < chTotal || chTotal === 0}
            stepLabel={`step ${step + 1}/${steps.length}`}
          />
        </section>
      </div>
    </div>
  )
}

function ScenarioViewer({
  uuid, steps, txPos, step, playing, secPerStep, onStep, onMapLoaded, skinUrl, viz,
}: {
  uuid: string; steps: Step[]; txPos: number[]; step: number; playing: boolean
  secPerStep: number; onStep: (s: number) => void; onMapLoaded?: () => void
  skinUrl?: string; viz: Viz
}) {
  // 카메라 fit: 스텝 위치 + TX 로 대략적 중심/스케일
  const center = useMemo<[number, number, number]>(() => {
    const pts = [txPos, ...steps.map((s) => s.pos)]
    const c = [0, 0, 0]
    pts.forEach((p) => { c[0] += p[0]; c[1] += p[1]; c[2] += p[2] })
    const n = Math.max(pts.length, 1)
    return [c[0] / n, c[1] / n, c[2] / n]
  }, [steps, txPos])
  const span = useMemo(() => {
    let m = 50
    const pts = [txPos, ...steps.map((s) => s.pos)]
    pts.forEach((p) => { m = Math.max(m, Math.abs(p[0] - center[0]), Math.abs(p[1] - center[1])) })
    return m * 2.2
  }, [steps, txPos, center])

  // 전체 레이 width(=power 기반) 범위 → 굵기 정규화용
  const wRange = useMemo<[number, number]>(() => {
    let lo = Infinity, hi = -Infinity
    for (const s of steps) for (const r of s.rays) {
      if (r.width < lo) lo = r.width
      if (r.width > hi) hi = r.width
    }
    if (!isFinite(lo)) return [0, 1]
    return [lo, hi]
  }, [steps])

  return (
    <Canvas
      camera={{ position: [center[0] + span * 0.6, center[1] - span * 0.6, center[2] + span * 0.6], fov: 40, up: [0, 0, 1] as any, near: span * 0.005, far: span * 50 }}
      onCreated={({ camera, gl }) => { camera.up.set(0, 0, 1); gl.toneMappingExposure = 1.25 }}
    >
      {/* 텍스처 GLB(PBR)가 너무 어둡지 않도록 전역광 보강 */}
      <ambientLight intensity={1.15} />
      <hemisphereLight args={['#ffffff', '#6b7280', 0.7]} />
      <directionalLight position={[span, span, span]} intensity={0.9} />
      <directionalLight position={[-span, -span, span]} intensity={0.4} />
      <OrbitControls target={center as any} makeDefault enableDamping />
      <SafeMap>
        {skinUrl
          ? <TexturedMap url={skinUrl} onLoaded={onMapLoaded} />
          : <MapMesh uuid={uuid} onLoaded={onMapLoaded} />}
      </SafeMap>
      {/* 이동 경로 (전체 RX 궤적) — 불투명 연한 연두색 굵은 선 */}
      <MobilityPath steps={steps} span={span} viz={viz} />
      {/* TX */}
      <mesh position={txPos as any}>
        <sphereGeometry args={[Math.max(span / 300, 1.5) * viz.txSize, 16, 16]} />
        <meshStandardMaterial color={viz.txColor} emissive={viz.txColor} emissiveIntensity={0.5} />
      </mesh>
      <Animator steps={steps} step={step} playing={playing} secPerStep={secPerStep} onStep={onStep} span={span} wRange={wRange} viz={viz} />
    </Canvas>
  )
}

function Animator({
  steps, step, playing, secPerStep, onStep, span, wRange, viz,
}: {
  steps: Step[]; step: number; playing: boolean; secPerStep: number
  onStep: (s: number) => void; span: number; wRange: [number, number]; viz: Viz
}) {
  const acc = useRef(0)
  useFrame((_s, dt) => {
    if (!playing || steps.length === 0) return
    acc.current += dt
    if (acc.current >= Math.max(secPerStep, 0.05)) {
      acc.current = 0
      onStep((step + 1) % steps.length)
    }
  })
  const cur = steps[step]
  if (!cur) return null
  const rxR = Math.max(span / 350, 1.2) * viz.rxSize
  return (
    <group>
      <mesh position={cur.pos as any}>
        <sphereGeometry args={[rxR, 16, 16]} />
        <meshStandardMaterial color={viz.rxColor} emissive={viz.rxColor} emissiveIntensity={0.6} />
      </mesh>
      {cur.rays.map((ray, i) => (
        <RayTube key={i} ray={ray} span={span} wRange={wRange} viz={viz} />
      ))}
    </group>
  )
}

function RayTube({ ray, span, wRange, viz }: { ray: Ray; span: number; wRange: [number, number]; viz: Viz }) {
  const [lo, hi] = wRange
  const norm = hi > lo ? (ray.width - lo) / (hi - lo) : 0.5   // 0=약 ~ 1=강 (굵기 차등)
  const geom = useMemo(() => {
    const pts = ray.pts.map((p) => new THREE.Vector3(p[0], p[1], p[2]))
    if (pts.length < 2) return null
    const path = new THREE.CurvePath<THREE.Vector3>()
    for (let i = 0; i < pts.length - 1; i++) path.add(new THREE.LineCurve3(pts[i], pts[i + 1]))
    const baseR = Math.min(span / 1100, 0.5)
    // 고정 굵기 토글 or (power 차등 × 사용자 배율)
    const radius = viz.rayFixed
      ? viz.rayFixedRadius
      : Math.min(baseR * (0.35 + 0.9 * norm), 0.6) * viz.rayThickness
    return new THREE.TubeGeometry(path, Math.max((pts.length - 1) * 2, 2), Math.max(radius, 0.02), 6, false)
  }, [ray, span, norm, viz.rayFixed, viz.rayFixedRadius, viz.rayThickness])
  if (!geom) return null
  const color = ray.los ? viz.losColor : viz.nlosColor
  return (
    <mesh geometry={geom}>
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.55}
        transparent={viz.rayOpacity < 1} opacity={viz.rayOpacity} />
    </mesh>
  )
}

function MobilityPath({ steps, span, viz }: { steps: Step[]; span: number; viz: Viz }) {
  const geom = useMemo(() => {
    if (steps.length < 2) return null
    const pts = steps.map((s) => new THREE.Vector3(s.pos[0], s.pos[1], s.pos[2]))
    const path = new THREE.CurvePath<THREE.Vector3>()
    for (let i = 0; i < pts.length - 1; i++) path.add(new THREE.LineCurve3(pts[i], pts[i + 1]))
    const radius = (span / 500) * viz.pathThickness
    return new THREE.TubeGeometry(path, Math.max((pts.length - 1) * 2, 2), Math.max(radius, 0.02), 8, false)
  }, [steps, span, viz.pathThickness])
  if (!geom) return null
  return (
    <mesh geometry={geom}>
      <meshStandardMaterial color={viz.pathColor} emissive={viz.pathColor} emissiveIntensity={0.4}
        transparent={viz.pathOpacity < 1} opacity={viz.pathOpacity} />
    </mesh>
  )
}

/**
 * 회색 배경 맵 메시.
 *  - 재질 부여 트윈(수천 PLY로 분할, scene_mesh.ply 없음)은 서버 병합 geometry(/scene/geometry)로 로드.
 *  - 단일 메시 씬(scene_mesh.ply)은 폴백 로드.
 *  - useLoader(Suspense) 대신 fetch+state 를 써서, 로드 실패 시에도 onLoaded 를 호출해
 *    '3D 로딩 중' 오버레이가 무한히 남지 않도록 한다. (2026-07-07 버그픽스)
 */
function MapMesh({ uuid, onLoaded }: { uuid: string; onLoaded?: () => void }) {
  const [geoms, setGeoms] = useState<THREE.BufferGeometry[]>([])
  useEffect(() => {
    let cancelled = false
    const done = () => { if (!cancelled) onLoaded?.() }
    async function loadMerged(): Promise<boolean> {
      try {
        const m = await fetch(`/api/sessions/${uuid}/scene/geometry`)
        if (!m.ok) return false
        const manifest = await m.json()
        const mats: { rel: string }[] = manifest?.materials ?? []
        if (!mats.length) return false
        const out: THREE.BufferGeometry[] = []
        for (const mt of mats) {
          if (cancelled) return true
          try {
            const r = await fetch(`/api/sessions/${uuid}/scene/merged/${encodeURIComponent(mt.rel)}`)
            if (!r.ok) continue
            const arr = new Float32Array(await r.arrayBuffer())
            if (arr.length >= 9) {
              const g = new THREE.BufferGeometry()
              g.setAttribute('position', new THREE.BufferAttribute(arr, 3))
              g.computeVertexNormals()
              out.push(g)
            }
          } catch { /* 이 재질만 스킵 */ }
        }
        if (!cancelled) { setGeoms(out); done() }
        return true
      } catch { return false }
    }
    async function loadSingle() {
      try {
        const r = await fetch(apiClient.meshUrl(uuid, 'scene_mesh.ply'))
        if (!r.ok) throw new Error('no scene_mesh.ply')
        const g = new PLYLoader().parse(await r.arrayBuffer()) as THREE.BufferGeometry
        g.computeVertexNormals()
        if (!cancelled) { setGeoms([g]); done() }
      } catch { done() /* 실패해도 오버레이는 닫아 무한 로딩 방지 */ }
    }
    ;(async () => {
      const ok = await loadMerged()
      if (!ok && !cancelled) await loadSingle()
    })()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uuid])
  if (!geoms.length) return null
  return (
    <>
      {geoms.map((g, i) => (
        // flatShading: 벽을 면 단위로 또렷하게. opacity 0.8 로 레이는 비치되 벽 통과 착시 완화.
        <mesh key={i} geometry={g}>
          <meshStandardMaterial color="#9aa3b2" roughness={0.95} metalness={0.0} flatShading transparent opacity={0.8} />
        </mesh>
      ))}
    </>
  )
}

/**
 * 텍스처 GLB 스킨 맵. RT 데이터(.ply)는 Z-up, GLB 는 glTF 표준 Y-up 이므로
 * X축 +90°(π/2) 회전으로 RT 좌표계에 정합한다. (bbox center 일치 검증 완료)
 */
function TexturedMap({ url, onLoaded }: { url: string; onLoaded?: () => void }) {
  const gltf = useLoader(GLTFLoader, url) as any
  useEffect(() => {
    if (!gltf?.scene) return
    // photogrammetry 텍스처는 빛이 이미 baked → diffuse맵을 emissive로도 사용해
    // 조명에 상관없이 밝고 충실하게 표시 (어두운 실내 느낌 제거)
    gltf.scene.traverse((o: any) => {
      if (!o.isMesh || !o.material) return
      const mats = Array.isArray(o.material) ? o.material : [o.material]
      mats.forEach((m: any) => {
        if (m.map) {
          m.emissive = new THREE.Color(0xffffff)
          m.emissiveMap = m.map
          m.emissiveIntensity = 0.55
        }
        if ('roughness' in m) m.roughness = 1.0
        if ('metalness' in m) m.metalness = 0.0
        m.needsUpdate = true
      })
    })
    onLoaded?.()
  }, [gltf])
  if (!gltf?.scene) return null
  return <primitive object={gltf.scene} rotation={[Math.PI / 2, 0, 0]} />
}

/** 맵 메시 로딩 실패/지연이 미리보기 전체를 막지 않도록 Suspense + ErrorBoundary 로 감싼다. */
class MapErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  constructor(props: { children: ReactNode }) { super(props); this.state = { failed: false } }
  static getDerivedStateFromError() { return { failed: true } }
  render() { return this.state.failed ? null : this.props.children }
}

function SafeMap({ children }: { children: ReactNode }) {
  return (
    <MapErrorBoundary>
      <Suspense fallback={null}>{children}</Suspense>
    </MapErrorBoundary>
  )
}

function RangeRow({ label, min, max, step, value, onChange, unit }: {
  label: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void; unit?: string
}) {
  return (
    <label className="flex items-center justify-between gap-2">
      <span className="text-slate-600 whitespace-nowrap">{label}</span>
      <span className="flex items-center gap-1">
        <input type="range" min={min} max={max} step={step} value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))} className="w-20" />
        <span className="font-mono text-slate-500 w-12 text-right">{value.toFixed(2)}{unit ?? ''}</span>
      </span>
    </label>
  )
}

function ColorRow({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <label className="flex items-center justify-between gap-2">
      <span className="text-slate-600 whitespace-nowrap">{label}</span>
      <input type="color" value={value} onChange={(e) => onChange(e.target.value)} className="w-10 h-6 p-0 border rounded" />
    </label>
  )
}

import { useMemo, useRef, useEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Html } from '@react-three/drei'
import * as THREE from 'three'

export type ChItem = {
  rx_idx: number
  rsrp: number | null
  los: boolean | null
  num_paths: number
  padp: { tau: number[]; aoa: number[]; power: number[] } | null
  r_rx: { m: number[]; disp: number; n: number } | null
  r_tx: { m: number[]; disp: number; n: number } | null
}

/** 0(약)~1(강) → 파랑→청록→초록→노랑→빨강 (viridis-lite). */
function viridis(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t))
  const stops = [[40, 50, 140], [30, 130, 160], [40, 170, 110], [180, 200, 60], [240, 90, 50]]
  const seg = x * (stops.length - 1)
  const i = Math.min(Math.floor(seg), stops.length - 2)
  const f = seg - i
  const a = stops[i], b = stops[i + 1]
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}

/** RSRP 반원형 게이지 (SVG). 범위 -140~-40 dBm, 높을수록 좋음. */
function RsrpGauge({ rsrp }: { rsrp: number | null }) {
  const MIN = -140, MAX = -40
  const W = 200, H = 150, cx = 100, cy = 100, r = 78
  const A0 = 210, A1 = -30   // 시작(좌하)→끝(우하), 시계방향 240°
  const polar = (deg: number, rad = r) => {
    const a = (deg * Math.PI) / 180
    return [cx + rad * Math.cos(a), cy - rad * Math.sin(a)]
  }
  const f2deg = (f: number) => A0 + (A1 - A0) * Math.max(0, Math.min(1, f))
  const v2f = (v: number) => (v - MIN) / (MAX - MIN)
  const arc = (f0: number, f1: number) => {
    const [x0, y0] = polar(f2deg(f0)); const [x1, y1] = polar(f2deg(f1))
    const large = Math.abs(f2deg(f1) - f2deg(f0)) > 180 ? 1 : 0
    return `M ${x0.toFixed(1)} ${y0.toFixed(1)} A ${r} ${r} 0 ${large} 1 ${x1.toFixed(1)} ${y1.toFixed(1)}`
  }
  // 색 구간 (dBm 경계): red <-110, orange -110~-95, light -95~-80, green >-80
  const bands = [
    { a: -140, b: -110, c: '#e2503a' },
    { a: -110, b: -95, c: '#f0a020' },
    { a: -95, b: -80, c: '#9ac61e' },
    { a: -80, b: -40, c: '#27ae60' },
  ]
  const ticks = [-140, -120, -100, -80, -60, -40]
  const hasV = rsrp != null && isFinite(rsrp)
  const vv = hasV ? Math.max(MIN, Math.min(MAX, rsrp as number)) : MIN
  const [nx, ny] = polar(f2deg(v2f(vv)), r - 10)
  const status = !hasV ? { t: 'Dead', c: '#64748b' }
    : vv >= -80 ? { t: 'Good', c: '#27ae60' }
    : vv >= -95 ? { t: 'Fair', c: '#f0a020' }
    : vv >= -110 ? { t: 'Poor', c: '#e2503a' }
    : { t: 'Very Poor', c: '#c0392b' }

  return (
    <div className="flex flex-col items-center">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-[220px]">
        {bands.map((bd, i) => (
          <path key={i} d={arc(v2f(bd.a), v2f(bd.b))} stroke={bd.c} strokeWidth={13}
            fill="none" strokeLinecap="butt" />
        ))}
        {ticks.map((tk) => {
          const [tx, ty] = polar(f2deg(v2f(tk)), r + 14)
          return <text key={tk} x={tx} y={ty} fontSize={9} fill="#64748b"
            textAnchor="middle" dominantBaseline="middle">{tk}</text>
        })}
        {/* 바늘 */}
        {hasV && <line x1={cx} y1={cy} x2={nx} y2={ny} stroke="#1e293b" strokeWidth={3} strokeLinecap="round" />}
        <circle cx={cx} cy={cy} r={6} fill="#1e293b" />
        <text x={cx} y={cy - 18} fontSize={26} fontWeight={700} fill="#0f2440" textAnchor="middle">
          {hasV ? (rsrp as number).toFixed(1) : '—'}
        </text>
        <text x={cx} y={cy - 4} fontSize={10} fill="#64748b" textAnchor="middle">dBm</text>
      </svg>
      <span className="px-2 py-0.5 rounded-full text-xs font-semibold mt-1"
        style={{ background: `${status.c}22`, color: status.c }}>● {status.t}</span>
      <div className="text-[10px] text-slate-400 mt-0.5">범위 -140 ~ -40 dBm · 높을수록 좋음</div>
    </div>
  )
}

/** PADP 3D 콩나물 — x=delay, y=AoA, z=정규화 power, 줄기는 바닥에서 위로. (three.js)
 *  padp 가 없으면(dead zone) 축/격자는 유지하고 데이터만 비운다 → 카메라/좌표 보존. */
function PadpMini({ padp }: { padp: ChItem['padp'] }) {
  const { stemGeo, headGeo } = useMemo(() => {
    const sg = new THREE.BufferGeometry()
    const hg = new THREE.BufferGeometry()
    if (!padp || padp.tau.length === 0) {
      sg.setAttribute('position', new THREE.Float32BufferAttribute([], 3))
      sg.setAttribute('color', new THREE.Float32BufferAttribute([], 3))
      hg.setAttribute('position', new THREE.Float32BufferAttribute([], 3))
      hg.setAttribute('color', new THREE.Float32BufferAttribute([], 3))
      return { stemGeo: sg, headGeo: hg }
    }
    const { tau, aoa, power } = padp
    const tmin = Math.min(...tau), tmax = Math.max(...tau)
    const amin = Math.min(...aoa), amax = Math.max(...aoa)
    const pmax = Math.max(...power, 1e-9)
    const nx = (v: number, a: number, b: number) => (b > a ? (v - a) / (b - a) : 0.5)
    const SX = 2, SY = 2, SZ = 1.6
    const sp: number[] = [], sc: number[] = [], hp: number[] = [], hc: number[] = []
    tau.forEach((t, i) => {
      const x = (nx(t, tmin, tmax) - 0.5) * SX
      const y = (nx(aoa[i], amin, amax) - 0.5) * SY
      const z = (power[i] / pmax) * SZ
      const [r, g, b] = viridis(power[i] / pmax).map((c) => c / 255) as [number, number, number]
      sp.push(x, y, 0, x, y, z); sc.push(r, g, b, r, g, b)
      hp.push(x, y, z); hc.push(r, g, b)
    })
    sg.setAttribute('position', new THREE.Float32BufferAttribute(sp, 3))
    sg.setAttribute('color', new THREE.Float32BufferAttribute(sc, 3))
    hg.setAttribute('position', new THREE.Float32BufferAttribute(hp, 3))
    hg.setAttribute('color', new THREE.Float32BufferAttribute(hc, 3))
    return { stemGeo: sg, headGeo: hg }
  }, [padp])

  return (
    <Canvas
      camera={{ position: [2.8, -2.8, 2.4], fov: 45, up: [0, 0, 1] as any }}
      onCreated={({ camera }) => camera.up.set(0, 0, 1)}
      style={{ height: 180 }}
    >
      <OrbitControls enablePan makeDefault screenSpacePanning />
      {/* 바닥 격자 (xy 평면) */}
      <gridHelper args={[2.4, 8, '#334155', '#1e293b']} rotation={[Math.PI / 2, 0, 0]} />
      {/* 축 (X=delay 빨강, Y=AoA 초록, Z=power 파랑) */}
      <AxisLine from={[-1.2, -1.2, 0]} to={[1.3, -1.2, 0]} color="#ef4444" />
      <AxisLine from={[-1.2, -1.2, 0]} to={[-1.2, 1.3, 0]} color="#22c55e" />
      <AxisLine from={[-1.2, -1.2, 0]} to={[-1.2, -1.2, 1.8]} color="#3b82f6" />
      <Html position={[1.45, -1.2, 0]} center style={lblStyle('#ef4444')}>delay</Html>
      <Html position={[-1.2, 1.5, 0]} center style={lblStyle('#22c55e')}>AoA</Html>
      <Html position={[-1.2, -1.2, 2.0]} center style={lblStyle('#3b82f6')}>power</Html>
      <lineSegments geometry={stemGeo}>
        <lineBasicMaterial vertexColors transparent opacity={0.9} />
      </lineSegments>
      <points geometry={headGeo}>
        <pointsMaterial vertexColors size={0.12} sizeAttenuation />
      </points>
    </Canvas>
  )
}

function AxisLine({ from, to, color }: { from: [number, number, number]; to: [number, number, number]; color: string }) {
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute([...from, ...to], 3))
    return g
  }, [from, to])
  return <lineSegments geometry={geo}><lineBasicMaterial color={color} /></lineSegments>
}

function lblStyle(color: string): React.CSSProperties {
  return {
    color, fontSize: '10px', fontWeight: 700, whiteSpace: 'nowrap',
    textShadow: '0 0 3px #000', pointerEvents: 'none', userSelect: 'none',
  }
}

/** 공분산 |R| heatmap (2D 캔버스). disp×disp 정규화 행렬. */
function CovHeatmap({ cov, label }: { cov: NonNullable<ChItem['r_rx']>; label: string }) {
  const ref = useRef<HTMLCanvasElement | null>(null)
  const SIZE = 130
  useEffect(() => {
    const cv = ref.current
    if (!cv) return
    const ctx = cv.getContext('2d')!
    const d = cov.disp
    const cell = SIZE / d
    for (let r = 0; r < d; r++) {
      for (let c = 0; c < d; c++) {
        const v = cov.m[r * d + c] ?? 0
        const [R, G, B] = viridis(v)
        ctx.fillStyle = `rgb(${R | 0},${G | 0},${B | 0})`
        ctx.fillRect(c * cell, r * cell, Math.ceil(cell), Math.ceil(cell))
      }
    }
  }, [cov])
  return (
    <div className="text-center">
      <canvas ref={ref} width={SIZE} height={SIZE} className="rounded border border-slate-200 mx-auto" />
      <div className="text-[10px] text-slate-500 mt-0.5">
        |{label}| {cov.n}×{cov.n}{cov.n > cov.disp ? ` (→${cov.disp} 표시)` : ''}
      </div>
    </div>
  )
}

/** 9.Scenario Results 우측 Channel State 패널 — 현재 step 의 RSRP/PADP/Cov. */
export function ChannelStatePanel({ item, loading, stepLabel }: {
  item: ChItem | null | undefined; loading: boolean; stepLabel: string
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold">Channel State</h3>
        <span className="text-xs text-slate-400 font-mono">{stepLabel}</span>
      </div>

      {!item ? (
        <div className="text-xs text-slate-400 py-6 text-center">
          {loading ? '채널 상태 로딩 중 …' : '이 step 의 채널 데이터가 없습니다.'}
        </div>
      ) : (
        <>
          {/* RSRP 게이지 + 경로수/LoS */}
          <div className="flex items-center gap-2">
            <div className="flex-1 min-w-0">
              <RsrpGauge rsrp={item.rsrp} />
            </div>
            <div className="shrink-0 w-16 space-y-2 text-sm text-center">
              <div>
                <div className="text-[10px] text-slate-400">경로 수</div>
                <div className="font-mono font-semibold">{item.num_paths}</div>
              </div>
              {item.los != null && (
                <span className={`inline-block w-full px-1 py-0.5 rounded text-[11px] font-medium ${item.los ? 'bg-blue-100 text-blue-700' : 'bg-amber-100 text-amber-700'}`}>
                  {item.los ? 'LoS' : 'NLoS'}
                </span>
              )}
            </div>
          </div>

          {/* PADP 3D 콩나물 — dead zone 에도 항상 마운트(카메라/좌표 보존), 데이터만 비움 */}
          <div>
            <div className="text-[10px] text-slate-400 mb-0.5">PADP — 회전: 좌클릭드래그 · 이동: 우클릭드래그 · 확대: 휠</div>
            <div className="bg-slate-950 rounded overflow-hidden relative" style={{ height: 180 }}>
              <PadpMini padp={item.padp} />
              {(!item.padp || item.padp.tau.length === 0) && (
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none text-xs text-slate-400">
                  dead zone — no path
                </div>
              )}
            </div>
            {item.padp && item.padp.tau.length > 0 ? (
              <div className="text-[10px] text-slate-400 mt-0.5 leading-tight">
                <span className="text-red-500">delay</span> {Math.min(...item.padp.tau).toFixed(0)}~{Math.max(...item.padp.tau).toFixed(0)} ns ·{' '}
                <span className="text-green-600">AoA</span> {Math.min(...item.padp.aoa).toFixed(0)}~{Math.max(...item.padp.aoa).toFixed(0)}° ·{' '}
                <span className="text-blue-600">power</span> 정규화(합=1)
              </div>
            ) : (
              <div className="text-[10px] text-slate-400 mt-0.5">유효 경로 없음 (Dead Zone)</div>
            )}
          </div>

          {/* Covariance */}
          <div>
            <div className="text-[10px] text-slate-400 mb-0.5">Spatial Covariance</div>
            <div className="grid grid-cols-2 gap-2">
              {item.r_rx ? <CovHeatmap cov={item.r_rx} label="R_RX" /> : <CovEmpty label="R_RX" />}
              {item.r_tx ? <CovHeatmap cov={item.r_tx} label="R_TX" /> : <CovEmpty label="R_TX" />}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

function CovEmpty({ label }: { label: string }) {
  return <div className="text-[10px] text-slate-400 h-[150px] flex items-center justify-center border border-slate-100 rounded">{label}: N/A</div>
}

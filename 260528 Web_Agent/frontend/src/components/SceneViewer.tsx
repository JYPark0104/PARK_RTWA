import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Canvas, ThreeEvent, useFrame } from '@react-three/fiber'
import { OrbitControls, Grid, Box, Text, Billboard } from '@react-three/drei'
import * as THREE from 'three'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type { SceneInfo, Coord3 } from '../lib/api'

// ITU 재질별 표시 색 (Geo-Radio Env. Twin 의 재질별 PLY 를 3D 에서 색으로 구분)
const MATERIAL_COLORS: Record<string, string> = {
  itu_concrete: '#9ca3af',
  itu_glass: '#7dd3fc',
  itu_marble: '#f1f5f9',
  itu_wood: '#b45309',
  itu_wet_ground: '#3f6212',
  itu_metal: '#cbd5e1',
  itu_brick: '#b91c1c',
  itu_ceiling_board: '#fde68a',
  itu_plasterboard: '#fcd34d',
  itu_floorboard: '#d97706',
  itu_chipboard: '#a16207',
  itu_very_dry_ground: '#ca8a04',
  itu_medium_dry_ground: '#65a30d',
}
const DEFAULT_MESH_COLOR = '#cbd5e1'

/** 메시 파일명/경로에서 재질명을 추출.
 *  · ITU 재질: itu_* 패턴 (예: ..._v1-itu_glass.ply → itu_glass)
 *  · 커스텀 재질: 파일명 마지막 __ 세그먼트 (예: b0001__Empire__irr_glass.ply → irr_glass) */
function materialFromName(name: string): string | null {
  const itu = name.match(/itu_[a-z0-9_]+/i)
  if (itu) return itu[0].toLowerCase()
  const base = (name.split('/').pop() ?? name).replace(/\.[a-z0-9]+$/i, '')
  const seg = base.split('__').pop() ?? ''
  return /^[a-z][a-z0-9_]*$/i.test(seg) ? seg.toLowerCase() : null
}
function colorForName(name: string): string {
  const mat = materialFromName(name)
  return (mat && MATERIAL_COLORS[mat]) || DEFAULT_MESH_COLOR
}

export interface SceneViewerProps {
  uuid: string
  sceneInfo: SceneInfo
  tx: { position: Coord3; name: string; az_deg?: number; el_deg?: number }[]
  rx: Coord3[]
  coverageOverlay?: {
    center: [number, number]
    size: [number, number]
    height: number
  }
  /** facade(O2I) 영역 미리보기: 선택 XY 사각형 × z_min~z_max 높이의 반투명 직육면체 */
  facadeRegion?: {
    x_min: number; x_max: number; y_min: number; y_max: number
    z_min: number; z_max: number
  }
  /** RX 영역(그리드/지면격자) 지면 높이 반투명 사각형 — 슬라이더에 즉시 동기화 */
  regionOverlay?: {
    x_min: number; x_max: number; y_min: number; y_max: number; z: number
  }
  /** TX 방향성 설정 중인 TX 인덱스 (null 이면 기즈모 비표시) */
  orientTxIndex?: number | null
  /** 방향 변경 콜백 (idx, azimuth[deg], elevation[deg]) */
  onOrient?: (idx: number, az_deg: number, el_deg: number) => void
  /** 방향성 기즈모(화살표+구체) 크기 배율 (기본 1) */
  orientScale?: number
  /** 클릭 원점(연한 마커) — TX 진짜 위치(빨강)와 함께 표시 */
  txClicked?: Coord3[]
  /** if defined, mouse click on mesh surface triggers this with snapped world position */
  onPickSurface?: (point: Coord3) => void
  /** 마커(구) 크기 배율 — 시각 렌더링 전용 (RT 물리와 무관) */
  markerScale?: number
  /** 리스트에서 선택/호버된 TX 인덱스 → 3D 에서 노란색·확대·맥동 강조 */
  highlightTxIndex?: number | null
  /** 3D TX 마커 호버 콜백 (idx|null) → 리스트 행 강조용 */
  onTxHover?: (idx: number | null) => void
  /** 3D 맵에 TX 번호 라벨 상시 표시 */
  showTxLabels?: boolean
}

export function SceneViewer({
  uuid, sceneInfo, tx, rx, coverageOverlay, facadeRegion, regionOverlay, orientTxIndex, onOrient, orientScale = 1, onPickSurface, txClicked, markerScale = 1,
  highlightTxIndex, onTxHover, showTxLabels,
}: SceneViewerProps) {
  const bbMin = sceneInfo.aabb_min
  const bbMax = sceneInfo.aabb_max
  const center: [number, number, number] = [
    (bbMin[0] + bbMax[0]) / 2,
    (bbMin[1] + bbMax[1]) / 2,
    (bbMin[2] + bbMax[2]) / 2,
  ]
  const dx = bbMax[0] - bbMin[0]
  const dy = bbMax[1] - bbMin[1]
  const dz = bbMax[2] - bbMin[2]
  const size = Math.max(dx, dy, dz, 10)
  // 화면을 거의 채우는 카메라 거리 (fov 35°). 도시 씬은 비스듬한 위쪽에서 내려다보는 각도.
  const fov = 35
  const fitDist = (size * 0.6) / Math.tan((fov * Math.PI) / 360)
  const camPos: [number, number, number] = [
    center[0] + fitDist * 0.7,
    center[1] - fitDist * 0.7,
    center[2] + fitDist * 0.5,
  ]
  const [aabbVisible, setAabbVisible] = useState(false)  // AABB 와이어프레임은 기본 OFF
  const [showGrid, setShowGrid] = useState(true)
  const [topView, setTopView] = useState(false)
  const [meshProg, setMeshProg] = useState<{ loaded: number; total: number; failed: number } | null>(null)
  const ctrlsRef = useRef<any>(null)

  // TX 방향성 설정: WASD 키로 방위/고각 조절 (A/D=azimuth, W/S=elevation)
  useEffect(() => {
    if (orientTxIndex == null || !onOrient) return
    const step = 5
    function onKey(e: KeyboardEvent) {
      const t = tx[orientTxIndex!]
      if (!t) return
      let az = t.az_deg ?? 0
      let el = t.el_deg ?? 0
      const k = e.key.toLowerCase()
      if (k === 'a') az -= step
      else if (k === 'd') az += step
      else if (k === 'w') el = Math.min(90, el + step)
      else if (k === 's') el = Math.max(-90, el - step)
      else return
      e.preventDefault()
      onOrient!(orientTxIndex!, az, el)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [orientTxIndex, onOrient, tx])

  // 방향 설정 대상 TX 선택 시 카메라 타깃을 그 TX 로 이동(확대 느낌)
  useEffect(() => {
    if (orientTxIndex == null) return
    const t = tx[orientTxIndex]
    const c = ctrlsRef.current
    if (t && c) { c.target.set(t.position[0], t.position[1], t.position[2]); c.update() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [orientTxIndex])

  const setView = (kind: 'iso' | 'top' | 'side' | 'front') => {
    const c = ctrlsRef.current
    if (!c) return
    const cam = c.object as THREE.PerspectiveCamera
    const d = fitDist
    if (kind === 'iso') cam.position.set(center[0] + d * 0.7, center[1] - d * 0.7, center[2] + d * 0.5)
    if (kind === 'top') cam.position.set(center[0], center[1], center[2] + d * 1.0)
    if (kind === 'side') cam.position.set(center[0] + d, center[1], center[2])
    if (kind === 'front') cam.position.set(center[0], center[1] - d, center[2])
    cam.up.set(0, 0, 1)
    c.target.set(center[0], center[1], center[2])
    c.update()
    setTopView(kind === 'top')
  }

  return (
    <div className="relative h-[calc(100vh-220px)] min-h-[500px] bg-slate-950 rounded-lg overflow-hidden">
      {/* 카메라 제어 패널 */}
      <div className="absolute top-2 left-2 z-10 flex gap-1 text-xs">
        <button onClick={() => setView('iso')} className="px-2 py-1 rounded bg-slate-800/90 text-slate-200 hover:bg-slate-700">3D</button>
        <button onClick={() => setView('top')} className="px-2 py-1 rounded bg-slate-800/90 text-slate-200 hover:bg-slate-700">Top</button>
        <button onClick={() => setView('side')} className="px-2 py-1 rounded bg-slate-800/90 text-slate-200 hover:bg-slate-700">Side</button>
        <button onClick={() => setView('front')} className="px-2 py-1 rounded bg-slate-800/90 text-slate-200 hover:bg-slate-700">Front</button>
        <button onClick={() => setAabbVisible(v => !v)} className={`px-2 py-1 rounded text-slate-200 hover:bg-slate-700 ${aabbVisible ? 'bg-sky-700' : 'bg-slate-800/90'}`}>AABB</button>
        <button onClick={() => setShowGrid(v => !v)} className={`px-2 py-1 rounded text-slate-200 hover:bg-slate-700 ${showGrid ? 'bg-sky-700' : 'bg-slate-800/90'}`}>Grid</button>
      </div>
      {/* 안내 */}
      <div className="absolute top-2 right-2 z-10 text-xs text-slate-400 bg-slate-800/70 px-2 py-1 rounded">
        좌클릭 회전 · 우클릭 패닝 · 휠 확대/축소
      </div>
      {meshProg && meshProg.loaded < meshProg.total && (
        <div className="absolute inset-x-0 top-12 z-10 flex justify-center pointer-events-none">
          <div className="text-xs bg-slate-900/85 text-slate-200 px-3 py-1.5 rounded flex items-center gap-2">
            <span className="inline-block w-3 h-3 border-2 border-slate-500 border-t-sky-400 rounded-full animate-spin" />
            메시 로딩 {meshProg.loaded.toLocaleString()} / {meshProg.total.toLocaleString()}
            {meshProg.failed > 0 && <span className="text-amber-400">(실패 {meshProg.failed})</span>}
          </div>
        </div>
      )}
      {/* 재질 범례 (Geo-Radio Env. Twin: 재질이 부여된 경우만).
          scene_info.materials 뿐 아니라 메시 파일명에서 추출한 재질(커스텀 irr_glass 등)도 합쳐 표시. */}
      {sceneInfo.material_assigned && (() => {
        const mats = Array.from(new Set([
          ...(sceneInfo.materials ?? []),
          ...((sceneInfo.mesh_files ?? []).map((f) => materialFromName(f)).filter(Boolean) as string[]),
        ]))
        if (mats.length === 0) return null
        return (
          <div className="absolute bottom-2 left-2 z-10 text-xs bg-slate-900/80 px-2.5 py-2 rounded space-y-1">
            <div className="text-slate-300 font-medium mb-1">재질 (Materials)</div>
            {mats.map((m) => (
              <div key={m} className="flex items-center gap-2">
                <span className="inline-block w-3 h-3 rounded-sm border border-slate-600"
                  style={{ background: MATERIAL_COLORS[m] || DEFAULT_MESH_COLOR }} />
                <span className="text-slate-200 font-mono">{m}</span>
                {!m.startsWith('itu_') && <span className="text-slate-400">(custom)</span>}
              </div>
            ))}
          </div>
        )
      })()}
      <Canvas
        frameloop="always"
        camera={{ position: camPos, fov, near: size * 0.001, far: size * 100, up: [0, 0, 1] as any }}
        onCreated={({ camera }) => { camera.up.set(0, 0, 1) }}
      >
        <ambientLight intensity={0.7} />
        <directionalLight position={[size, size, size]} intensity={1.0} />
        <directionalLight position={[-size, -size, size]} intensity={0.4} />
        <OrbitControls
          ref={ctrlsRef}
          target={center as any}
          makeDefault
          enableDamping
          dampingFactor={0.08}
          zoomSpeed={1.5}
          panSpeed={1.5}
          rotateSpeed={0.9}
          minDistance={size * 0.02}
          maxDistance={size * 8}
          screenSpacePanning
        />
        <axesHelper args={[size / 8]} />
        {showGrid && (
          <Grid
            position={[center[0], center[1], bbMin[2]]}
            args={[size * 4, size * 4]}
            cellSize={size / 40}
            sectionSize={size / 8}
            cellColor={'#334'}
            sectionColor={'#556'}
            infiniteGrid
            rotation={[Math.PI / 2, 0, 0]}
          />
        )}

        <Meshes uuid={uuid} files={sceneInfo.mesh_files} onPickSurface={onPickSurface}
          onProgress={(l, t, f) => setMeshProg({ loaded: l, total: t, failed: f })}
          version={`${sceneInfo.n_vertices}_${sceneInfo.n_faces}_${bbMin.join(',')}_${bbMax.join(',')}`} />

        {/* TX clicked-origin (faint) markers */}
        {(txClicked ?? []).map((p, i) => (
          <Marker key={`txc${i}`} position={p} color="orange" size={Math.max(size / 700, 1.0) * markerScale} faint />
        ))}
        {/* TX markers (real = ground+offset). 번호 라벨 + 선택/호버 강조 */}
        {tx.map((t, i) => (
          <TxMarker key={`tx${i}`} position={t.position}
            size={Math.max(size / 600, 1.2) * markerScale}
            index={i}
            highlighted={highlightTxIndex === i}
            showLabel={!!showTxLabels}
            onHover={onTxHover} />
        ))}
        {/* TX 방향성 기즈모 (선택된 TX) */}
        {orientTxIndex != null && tx[orientTxIndex] && (
          <TxGizmo
            position={tx[orientTxIndex].position}
            azDeg={tx[orientTxIndex].az_deg ?? 0}
            elDeg={tx[orientTxIndex].el_deg ?? 0}
            size={Math.max(size / 200, 0.5) * (orientScale ?? 1)}
            onOrient={(az, el) => onOrient?.(orientTxIndex!, az, el)}
            onDragToggle={(d) => { if (ctrlsRef.current) ctrlsRef.current.enabled = !d }}
          />
        )}
        {/* RX markers — instancedMesh (수천~수만 개도 1 draw call) */}
        <RXInstances points={rx} size={Math.max(size / 1500, 0.7) * markerScale} />

        {/* Coverage Map overlay */}
        {coverageOverlay && (
          <mesh position={[coverageOverlay.center[0], coverageOverlay.center[1], coverageOverlay.height]}>
            <planeGeometry args={[coverageOverlay.size[0], coverageOverlay.size[1]]} />
            <meshBasicMaterial color="lime" transparent opacity={0.15} side={THREE.DoubleSide} />
          </mesh>
        )}

        {/* RX 영역 미리보기 (지면 높이 반투명 사각형 — 슬라이더 스윕에 즉시 동기화) */}
        {regionOverlay && regionOverlay.x_max > regionOverlay.x_min && regionOverlay.y_max > regionOverlay.y_min && (
          <mesh position={[
            (regionOverlay.x_min + regionOverlay.x_max) / 2,
            (regionOverlay.y_min + regionOverlay.y_max) / 2,
            regionOverlay.z + 0.1,
          ] as any}>
            <planeGeometry args={[regionOverlay.x_max - regionOverlay.x_min, regionOverlay.y_max - regionOverlay.y_min]} />
            <meshBasicMaterial color="#22c55e" transparent opacity={0.16} side={THREE.DoubleSide} depthWrite={false} />
          </mesh>
        )}

        {/* facade(O2I) 영역 미리보기 (반투명 직육면체: XY 사각형 × z_min~z_max) */}
        {facadeRegion && facadeRegion.x_max > facadeRegion.x_min && facadeRegion.y_max > facadeRegion.y_min && (
          <Box
            position={[
              (facadeRegion.x_min + facadeRegion.x_max) / 2,
              (facadeRegion.y_min + facadeRegion.y_max) / 2,
              (facadeRegion.z_min + facadeRegion.z_max) / 2,
            ] as any}
            args={[
              facadeRegion.x_max - facadeRegion.x_min,
              facadeRegion.y_max - facadeRegion.y_min,
              Math.max(0.5, facadeRegion.z_max - facadeRegion.z_min),
            ]}
          >
            <meshBasicMaterial color="#38bdf8" transparent opacity={0.15} side={THREE.DoubleSide} depthWrite={false} />
          </Box>
        )}

        {/* AABB box outline (토글 가능) */}
        {aabbVisible && (
          <Box position={center as any} args={[dx, dy, dz]}>
            <meshBasicMaterial wireframe color="#7dd3fc" transparent opacity={0.6} />
          </Box>
        )}
      </Canvas>
    </div>
  )
}

/** azimuth(동=+x=0°, CCW+) · elevation(위=+z, 아래=−) → boresight 단위벡터 */
function dirFromAzEl(azDeg: number, elDeg: number): THREE.Vector3 {
  const az = (azDeg * Math.PI) / 180
  const el = (elDeg * Math.PI) / 180
  return new THREE.Vector3(Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el))
}

/** TX 방향성 기즈모: 반투명 구체(드래그 핸들) + boresight 화살표. WASD 는 상위에서 처리. */
function TxGizmo({ position, azDeg, elDeg, size, onOrient, onDragToggle }: {
  position: Coord3; azDeg: number; elDeg: number; size: number
  onOrient?: (az: number, el: number) => void
  onDragToggle?: (dragging: boolean) => void
}) {
  const drag = useRef<{ x: number; y: number; az: number; el: number } | null>(null)
  const len = size * 6
  const r = size * 0.16
  const quat = useMemo(() => {
    const q = new THREE.Quaternion()
    q.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dirFromAzEl(azDeg, elDeg).normalize())
    return q
  }, [azDeg, elDeg])

  useEffect(() => {
    if (!onOrient) return
    function move(e: PointerEvent) {
      const d = drag.current
      if (!d) return
      const az = d.az + (e.clientX - d.x) * 0.5           // 0.5°/px
      const el = Math.max(-90, Math.min(90, d.el - (e.clientY - d.y) * 0.5))
      onOrient!(az, el)
    }
    function up() { if (drag.current) { drag.current = null; onDragToggle?.(false) } }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    return () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up) }
  }, [onOrient, onDragToggle])

  return (
    <group position={position as any}>
      {/* 드래그 핸들 = 반투명 구체 (회전 span 시각화) */}
      <mesh
        onPointerDown={(e: any) => {
          e.stopPropagation()
          drag.current = { x: e.clientX, y: e.clientY, az: azDeg, el: elDeg }
          onDragToggle?.(true)
        }}
      >
        <sphereGeometry args={[len, 24, 24]} />
        <meshBasicMaterial color="#38bdf8" transparent opacity={0.12} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      {/* boresight 화살표: +Y 기준 제작 후 quaternion 으로 방향 회전 */}
      <group quaternion={quat as any}>
        <mesh position={[0, len * 0.4, 0]}>
          <cylinderGeometry args={[r, r, len * 0.8, 12]} />
          <meshStandardMaterial color="#f59e0b" emissive="#f59e0b" emissiveIntensity={0.5} />
        </mesh>
        <mesh position={[0, len * 0.9, 0]}>
          <coneGeometry args={[r * 2.4, len * 0.24, 16]} />
          <meshStandardMaterial color="#f59e0b" emissive="#f59e0b" emissiveIntensity={0.5} />
        </mesh>
      </group>
    </group>
  )
}

/** TX 마커: 구 + 번호 라벨(빌보드, 항상 카메라 향함·건물에 안 가려짐) +
 *  선택/호버 시 노란색·확대·맥동으로 강조. */
function TxMarker({ position, size, index, highlighted, showLabel, onHover }: {
  position: Coord3; size: number; index: number; highlighted: boolean; showLabel: boolean
  onHover?: (idx: number | null) => void
}) {
  const ref = useRef<THREE.Mesh>(null)
  useFrame(({ clock }) => {
    if (!ref.current) return
    ref.current.scale.setScalar(highlighted ? 1 + 0.3 * Math.sin(clock.elapsedTime * 6) : 1)
  })
  const color = highlighted ? '#fde047' : '#ef4444'
  const r = highlighted ? size * 1.7 : size
  return (
    <group position={position as any}>
      <mesh ref={ref}
        onPointerOver={(e: ThreeEvent<PointerEvent>) => { e.stopPropagation(); onHover?.(index) }}
        onPointerOut={() => onHover?.(null)}>
        <sphereGeometry args={[r, 16, 16]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={highlighted ? 0.9 : 0.4} />
      </mesh>
      {(showLabel || highlighted) && (
        <Billboard position={[0, 0, r * 2.4]}>
          <Text fontSize={size * 2.6} color={highlighted ? '#fde047' : '#ffffff'}
            anchorX="center" anchorY="middle"
            outlineWidth={size * 0.18} outlineColor="#0f172a"
            renderOrder={999} material-depthTest={false} material-depthWrite={false}>
            {String(index + 1)}
          </Text>
        </Billboard>
      )}
    </group>
  )
}

function Marker({
  position, color, size, faint = false,
}: {
  position: Coord3
  color: string
  size: number
  faint?: boolean
}) {
  const [px, py, pz] = position
  return (
    <group position={[px, py, pz]}>
      <mesh>
        <sphereGeometry args={[size, 16, 16]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={faint ? 0.15 : 0.4}
          transparent={faint}
          opacity={faint ? 0.35 : 1}
        />
      </mesh>
    </group>
  )
}

/** 다수 RX 를 InstancedMesh 로 렌더 — 개별 mesh 대비 draw call/삼각형 급감. */
function RXInstances({ points, size, color = 'dodgerblue' }: { points: Coord3[]; size: number; color?: string }) {
  const ref = useRef<THREE.InstancedMesh>(null)
  const count = points.length
  useLayoutEffect(() => {
    const mesh = ref.current
    if (!mesh || count === 0) return
    const m = new THREE.Matrix4()
    const q = new THREE.Quaternion()
    const s = new THREE.Vector3(size, size, size)
    const pos = new THREE.Vector3()
    for (let i = 0; i < count; i++) {
      pos.set(points[i][0], points[i][1], points[i][2])
      m.compose(pos, q, s)
      mesh.setMatrixAt(i, m)
    }
    mesh.instanceMatrix.needsUpdate = true
  }, [points, size, count])
  if (count === 0) return null
  return (
    <instancedMesh ref={ref} key={count} args={[undefined as any, undefined as any, count]}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.4} />
    </instancedMesh>
  )
}

interface MeshesProps {
  uuid: string
  files: string[]
  onPickSurface?: (point: Coord3) => void
  version?: string
  onProgress?: (loaded: number, total: number, failed: number) => void
}

/**
 * 다수 PLY(수천 개 가능) 를 견고하게 로드한다:
 *  (a) 개별 로드 실패는 비치명적으로 스킵 (한 파일 실패로 뷰어 전체가 죽지 않음)
 *  (b) 동시 fetch 를 CONCURRENCY 개로 제한 (연결 고갈=Failed to fetch 방지)
 *  (c) 로드된 지오메트리를 재질(색)별로 병합해 3~4개 메시로 렌더 (draw call 급감)
 * 시각 결과는 개별 렌더와 동일하되, 수천 shape 씬에서도 안정적으로 동작한다.
 */
function Meshes({ uuid, files, onPickSurface, version = '', onProgress }: MeshesProps) {
  const [groups, setGroups] = useState<{ color: string; geometry: THREE.BufferGeometry }[]>([])
  const downPos = useRef<{ x: number; y: number } | null>(null)

  useEffect(() => {
    let cancelled = false
    const loader = new PLYLoader()
    setGroups([])
    onProgress?.(0, 1, 0)

    // (신규) 서버 병합 geometry — 재질당 1요청. 수천 PLY 개별요청(요청 폭주=실패 누적) 제거.
    async function loadMerged(): Promise<boolean> {
      try {
        const mResp = await fetch(`/api/sessions/${uuid}/scene/geometry?v=${encodeURIComponent(version)}`)
        if (!mResp.ok) return false
        const manifest = await mResp.json()
        const mats: { name: string; rel: string }[] = manifest?.materials ?? []
        if (!mats.length) return false
        const out: { color: string; geometry: THREE.BufferGeometry }[] = []
        let done = 0
        for (const mt of mats) {
          if (cancelled) return true
          try {
            const r = await fetch(`/api/sessions/${uuid}/scene/merged/${encodeURIComponent(mt.rel)}?v=${encodeURIComponent(version)}`)
            if (!r.ok) throw new Error('http ' + r.status)
            const arr = new Float32Array(await r.arrayBuffer())
            if (arr.length >= 9) {
              const g = new THREE.BufferGeometry()
              g.setAttribute('position', new THREE.BufferAttribute(arr, 3))
              g.computeVertexNormals()
              g.computeBoundingBox()
              g.computeBoundingSphere()
              out.push({ color: colorForName(mt.name), geometry: g })
            }
          } catch { /* 이 재질만 스킵(비치명적) */ }
          done++
          if (!cancelled) onProgress?.(done, mats.length, 0)
        }
        if (!cancelled) setGroups(out)
        return true
      } catch {
        return false
      }
    }

    // (폴백) 예전 방식: 파일당 개별 로드 + 클라이언트 병합 (병합 API 실패 시에만)
    const byColor = new Map<string, THREE.BufferGeometry[]>()
    let done = 0, failed = 0
    const CONCURRENCY = 8
    async function fetchGeom(url: string, ms: number): Promise<THREE.BufferGeometry> {
      const ctrl = new AbortController()
      const to = setTimeout(() => ctrl.abort(), ms)
      try {
        const resp = await fetch(url, { signal: ctrl.signal })
        if (!resp.ok) throw new Error('http ' + resp.status)
        const buf = await resp.arrayBuffer()
        return loader.parse(buf) as THREE.BufferGeometry
      } finally {
        clearTimeout(to)
      }
    }
    async function loadOne(rel: string) {
      const base = rel.split('/').pop()!
      const url = `/api/sessions/${uuid}/scene/mesh/${encodeURIComponent(base)}?v=${encodeURIComponent(version)}`
      let raw: THREE.BufferGeometry | null = null
      for (let attempt = 0; attempt < 3 && !cancelled && !raw; attempt++) {
        try { raw = await fetchGeom(url, 12000) }
        catch { raw = null; await new Promise((r) => setTimeout(r, 300 * (attempt + 1))) }
      }
      if (raw && !cancelled) {
        try {
          const ni = raw.index ? raw.toNonIndexed() : raw
          const pos = ni.getAttribute('position')
          if (pos) {
            const g = new THREE.BufferGeometry()
            g.setAttribute('position', pos)
            const color = colorForName(rel)
            const arr = byColor.get(color) ?? []
            arr.push(g); byColor.set(color, arr)
          }
        } catch { failed++ }
      } else if (!cancelled) { failed++ }
      done++
      if (!cancelled) onProgress?.(done, files.length, failed)
    }
    async function loadPerFile() {
      onProgress?.(0, files.length, 0)
      let idx = 0
      const worker = async () => {
        while (!cancelled) {
          const i = idx++
          if (i >= files.length) break
          await loadOne(files[i])
        }
      }
      await Promise.all(Array.from({ length: Math.min(CONCURRENCY, Math.max(1, files.length)) }, worker))
      if (cancelled) return
      const merged: { color: string; geometry: THREE.BufferGeometry }[] = []
      for (const [color, geoms] of byColor) {
        if (!geoms.length) continue
        try {
          const m = geoms.length === 1 ? geoms[0] : mergeGeometries(geoms, false)
          if (m) { m.computeVertexNormals(); m.computeBoundingBox(); m.computeBoundingSphere(); merged.push({ color, geometry: m }) }
        } catch { /* skip */ }
      }
      if (!cancelled) setGroups(merged)
    }

    ;(async () => {
      const ok = await loadMerged()
      if (!ok && !cancelled) await loadPerFile()
    })()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uuid, version, files.length])

  return (
    <>
      {groups.map((g, i) => (
        <mesh
          key={`${g.color}|${i}`}
          geometry={g.geometry}
          onPointerDown={onPickSurface ? (e: ThreeEvent<PointerEvent>) => { downPos.current = { x: e.clientX, y: e.clientY } } : undefined}
          onPointerUp={onPickSurface ? (e: ThreeEvent<PointerEvent>) => {
            if (!downPos.current) return
            const dx = e.clientX - downPos.current.x
            const dy = e.clientY - downPos.current.y
            downPos.current = null
            if (Math.hypot(dx, dy) > 5) return  // 드래그 → pick 무시
            const p = e.point
            onPickSurface([p.x, p.y, p.z])
          } : undefined}
        >
          <meshStandardMaterial color={g.color} roughness={0.85} metalness={0.05} side={THREE.DoubleSide} />
        </mesh>
      ))}
    </>
  )
}

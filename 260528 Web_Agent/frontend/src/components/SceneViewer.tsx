import { useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, ThreeEvent, useLoader } from '@react-three/fiber'
import { OrbitControls, Grid, Box } from '@react-three/drei'
import * as THREE from 'three'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
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

/** 메시 파일명/경로에서 itu_* 재질명을 추출 (예: ..._v1-itu_glass.ply → itu_glass) */
function materialFromName(name: string): string | null {
  const m = name.match(/itu_[a-z0-9_]+/i)
  return m ? m[0].toLowerCase() : null
}
function colorForName(name: string): string {
  const mat = materialFromName(name)
  return (mat && MATERIAL_COLORS[mat]) || DEFAULT_MESH_COLOR
}

export interface SceneViewerProps {
  uuid: string
  sceneInfo: SceneInfo
  tx: { position: Coord3; name: string }[]
  rx: Coord3[]
  coverageOverlay?: {
    center: [number, number]
    size: [number, number]
    height: number
  }
  /** 클릭 원점(연한 마커) — TX 진짜 위치(빨강)와 함께 표시 */
  txClicked?: Coord3[]
  /** if defined, mouse click on mesh surface triggers this with snapped world position */
  onPickSurface?: (point: Coord3) => void
}

export function SceneViewer({
  uuid, sceneInfo, tx, rx, coverageOverlay, onPickSurface, txClicked,
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
  const ctrlsRef = useRef<any>(null)

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
      {/* 재질 범례 (Geo-Radio Env. Twin: 재질이 부여된 경우만) */}
      {sceneInfo.material_assigned && sceneInfo.materials?.length > 0 && (
        <div className="absolute bottom-2 left-2 z-10 text-xs bg-slate-900/80 px-2.5 py-2 rounded space-y-1">
          <div className="text-slate-300 font-medium mb-1">재질 (Materials)</div>
          {sceneInfo.materials.map((m) => (
            <div key={m} className="flex items-center gap-2">
              <span className="inline-block w-3 h-3 rounded-sm border border-slate-600"
                style={{ background: MATERIAL_COLORS[m] || DEFAULT_MESH_COLOR }} />
              <span className="text-slate-200 font-mono">{m}</span>
            </div>
          ))}
        </div>
      )}
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
          version={`${sceneInfo.n_vertices}_${sceneInfo.n_faces}_${bbMin.join(',')}_${bbMax.join(',')}`} />

        {/* TX clicked-origin (faint) markers */}
        {(txClicked ?? []).map((p, i) => (
          <Marker key={`txc${i}`} position={p} color="orange" size={Math.max(size / 700, 1.0)} faint />
        ))}
        {/* TX markers (real = ground+offset, red) */}
        {tx.map((t, i) => (
          <Marker key={`tx${i}`} position={t.position} color="red" size={Math.max(size / 600, 1.2)} />
        ))}
        {/* RX markers */}
        {rx.map((p, i) => (
          <Marker key={`rx${i}`} position={p} color="dodgerblue" size={Math.max(size / 1500, 0.7)} />
        ))}

        {/* Coverage Map overlay */}
        {coverageOverlay && (
          <mesh position={[coverageOverlay.center[0], coverageOverlay.center[1], coverageOverlay.height]}>
            <planeGeometry args={[coverageOverlay.size[0], coverageOverlay.size[1]]} />
            <meshBasicMaterial color="lime" transparent opacity={0.15} side={THREE.DoubleSide} />
          </mesh>
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

interface MeshesProps {
  uuid: string
  files: string[]
  onPickSurface?: (point: Coord3) => void
  version?: string
}

function Meshes({ uuid, files, onPickSurface, version = '' }: MeshesProps) {
  const v = encodeURIComponent(version)
  return (
    <>
      {files.map((rel) => (
        <PLYMesh
          key={`${rel}|${version}`}
          url={`/api/sessions/${uuid}/scene/mesh/${encodeURIComponent(rel.split('/').pop()!)}?v=${v}`}
          color={colorForName(rel)}
          onPick={onPickSurface}
        />
      ))}
    </>
  )
}

function PLYMesh({ url, color = DEFAULT_MESH_COLOR, onPick }: { url: string; color?: string; onPick?: (p: Coord3) => void }) {
  const geom = useLoader(PLYLoader, url) as THREE.BufferGeometry
  const meshRef = useRef<THREE.Mesh>(null)
  // 드래그 vs 클릭 구분 — 5px 이상 움직이면 드래그(회전/패닝)로 보고 pick 무시
  const downPos = useRef<{ x: number; y: number } | null>(null)

  useEffect(() => {
    geom.computeVertexNormals()
    // mesh-side raycast 비용을 줄이기 위해 bounding box/sphere 계산
    geom.computeBoundingBox()
    geom.computeBoundingSphere()
  }, [geom])

  // onPick이 *없을 때*는 raycast 자체를 비활성 — OrbitControls 응답성 ↑
  useEffect(() => {
    if (!meshRef.current) return
    if (!onPick) {
      // no-op raycast → pointer 이벤트 무시
      ;(meshRef.current as any).raycast = () => {}
    }
  }, [onPick])

  return (
    <mesh
      ref={meshRef}
      geometry={geom}
      onPointerDown={(e: ThreeEvent<PointerEvent>) => {
        if (!onPick) return
        downPos.current = { x: e.clientX, y: e.clientY }
      }}
      onPointerUp={(e: ThreeEvent<PointerEvent>) => {
        if (!onPick || !downPos.current) return
        const dx = e.clientX - downPos.current.x
        const dy = e.clientY - downPos.current.y
        downPos.current = null
        if (Math.hypot(dx, dy) > 5) return  // 드래그였음 → pick 무시
        const p = e.point
        onPick([p.x, p.y, p.z])
      }}
    >
      <meshStandardMaterial color={color} roughness={0.85} metalness={0.05} side={THREE.DoubleSide} />
    </mesh>
  )
}

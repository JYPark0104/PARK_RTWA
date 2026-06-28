import { create } from 'zustand'
import type { Coord3, JobStatus, SceneInfo, SessionMeta } from '../lib/api'

export type AntennaMode = 'simple' | 'advanced'

export interface AntennaSimple {
  bs_rows: number
  bs_cols: number
  ue_rows: number
  ue_cols: number
}

export interface CoverageMapOpts {
  enabled: boolean
  cell_size_x: number
  cell_size_y: number
  height_m: number
  samples_per_tx: number
  max_depth: number
  specular_reflection: boolean
  diffuse_reflection: boolean
  refraction: boolean
}

export interface RTOpts {
  mode: 'simple' | 'advanced'
  engine: 'p1a' | 'batch'
  batch_size: number
  random_batch: boolean   // batch RT: RX 순서를 무작위로 섞어 배치 구성 (가로줄 클러스터 완화)
  frequency_ghz: number
  max_depth: number
  seed: number
  num_interesting_paths: number
  max_rays_per_pair: number
  pathsolver_los: boolean
  pathsolver_specular_reflection: boolean
  pathsolver_diffuse_reflection: boolean
  pathsolver_refraction: boolean
  pathsolver_synthetic_array: boolean
  itu_scattering_coeff: number
  itu_xpd_coeff: number
  // TX 배치 (PARK_2 RT 이식)
  tx_ground_offset_m: number
  // 고급 옵션 (PARK_2 config.yaml 이식)
  pathsolver_diffraction: boolean
  pathsolver_edge_diffraction: boolean
  pathsolver_diffraction_lit_region: boolean
  num_samples: number
  max_num_paths: number
  relative_permittivity: number
  conductivity: number
  material_thickness: number
  scattering_pattern: 'lambertian' | 'directive' | 'backscattering'
  directive_alpha_r: number
  backscattering_alpha_r: number
  backscattering_alpha_i: number
  backscattering_lambda: number
  tx_pattern: string
  tx_polarization: string
  rx_pattern: string
  rx_polarization: string
  coverage_map: CoverageMapOpts
}

export interface TXItem {
  position: Coord3
  orientation: Coord3
  name: string
  clicked?: Coord3 | null   // 클릭 원점(연한 마커). position = 지면+offset 스냅 위치
}

export type RXMethod = 'grid' | 'explicit' | 'radial' | 'street' | 'clicks' | 'ground_grid'

export interface RXState {
  method: RXMethod
  // grid
  x_start: number
  x_stop: number
  x_num: number
  y_start: number
  y_stop: number
  y_num: number
  z_values: number[]
  // radial
  center_xy: [number, number] | null
  radii_m: number[]
  angles_start: number
  angles_stop: number
  angles_num: number
  // street
  path_points: [number, number][]
  num_points: number
  // clicks (snap-to-surface)
  click_positions: Coord3[]
  // ground_grid (PARK_2: 경계+레이캐스팅 지면 격자)
  grid_n: number
  margin: number
  rx_height: number
  raycasting_z: number | null
  max_height: number | null   // 이 높이(지면고도+rx_height, m) 초과 RX 는 배치 단계에서 제거
  rx_layout: 'density' | 'spacing'   // ground_grid 배치 방식: 밀도(grid_n) vs 간격(m)
  spacing_m: number            // 간격(미터, 정사각). rx_layout==='spacing' 일 때 사용
  ground_positions: Coord3[]   // 미리보기/제출용으로 백엔드가 계산한 지면 RX
}

export interface AppState {
  session: SessionMeta | null
  setSession(s: SessionMeta | null): void

  sceneInfo: SceneInfo | null
  setSceneInfo(s: SceneInfo | null): void

  antennaMode: AntennaMode
  antenna: AntennaSimple
  setAntenna(p: Partial<AntennaSimple>): void

  tx: TXItem[]
  addTX(p: Coord3): void
  addTXFull(item: TXItem): void
  removeTX(idx: number): void
  clearTX(): void

  rx: RXState
  setRX(p: Partial<RXState>): void

  rt: RTOpts
  setRT(p: Partial<RTOpts>): void

  selectedMetrics: string[]
  setSelectedMetrics(m: string[]): void
  toggleMetric(id: string): void

  currentJob: JobStatus | null
  setJob(j: JobStatus | null): void
  jobEvents: any[]
  appendEvent(e: any): void
  clearEvents(): void

  scenarioResult: any | null
  setScenarioResult(r: any | null): void

  user: { id: string; name: string } | null
  setUser(u: { id: string; name: string } | null): void
}

const defaultRT: RTOpts = {
  mode: 'simple',
  engine: 'p1a',
  batch_size: 50,
  random_batch: false,
  frequency_ghz: 7.5,
  max_depth: 5,
  seed: 41,
  num_interesting_paths: 20,
  max_rays_per_pair: 400,
  pathsolver_los: true,
  pathsolver_specular_reflection: true,
  pathsolver_diffuse_reflection: true,
  pathsolver_refraction: true,
  pathsolver_synthetic_array: false,
  itu_scattering_coeff: 0.2,
  itu_xpd_coeff: 0.5,
  tx_ground_offset_m: 2.0,
  pathsolver_diffraction: false,
  pathsolver_edge_diffraction: false,
  pathsolver_diffraction_lit_region: true,
  num_samples: 100000,
  max_num_paths: 10000,
  relative_permittivity: 5.24,
  conductivity: 0.0462,
  material_thickness: 0.1,
  scattering_pattern: 'lambertian',
  directive_alpha_r: 10,
  backscattering_alpha_r: 20,
  backscattering_alpha_i: 30,
  backscattering_lambda: 0.7,
  tx_pattern: 'iso',
  tx_polarization: 'V',
  rx_pattern: 'dipole',
  rx_polarization: 'V',
  coverage_map: {
    enabled: false,
    cell_size_x: 1.0,
    cell_size_y: 1.0,
    height_m: 1.5,
    samples_per_tx: 1e8,
    max_depth: 5,
    specular_reflection: true,
    diffuse_reflection: true,
    refraction: true,
  },
}

const defaultRX: RXState = {
  method: 'ground_grid',
  x_start: -100, x_stop: 100, x_num: 20,
  y_start: -100, y_stop: 100, y_num: 20,
  z_values: [1.5],
  center_xy: null,
  radii_m: [50, 100, 150],
  angles_start: 0, angles_stop: 360, angles_num: 12,
  path_points: [[0, -50], [0, 50]],
  num_points: 20,
  click_positions: [],
  grid_n: 20,
  margin: 0.05,
  rx_height: 1.5,
  raycasting_z: null,
  max_height: null,
  rx_layout: 'density',
  spacing_m: 10,
  ground_positions: [],
}

export const useStore = create<AppState>((set) => ({
  session: null,
  setSession: (s) => set({ session: s }),

  sceneInfo: null,
  setSceneInfo: (s) => set({ sceneInfo: s }),

  antennaMode: 'simple',
  antenna: { bs_rows: 32, bs_cols: 32, ue_rows: 4, ue_cols: 4 },
  setAntenna: (p) => set((st) => ({ antenna: { ...st.antenna, ...p } })),

  tx: [],
  addTX: (p) => set((st) => ({
    tx: [...st.tx, { position: p, orientation: [0, 0, 0], name: `tx${st.tx.length + 1}` }],
  })),
  addTXFull: (item) => set((st) => ({ tx: [...st.tx, item] })),
  removeTX: (idx) => set((st) => ({ tx: st.tx.filter((_, i) => i !== idx) })),
  clearTX: () => set({ tx: [] }),

  rx: defaultRX,
  setRX: (p) => set((st) => ({ rx: { ...st.rx, ...p } })),

  rt: defaultRT,
  setRT: (p) => set((st) => ({ rt: { ...st.rt, ...p } })),

  selectedMetrics: [],
  setSelectedMetrics: (m) => set({ selectedMetrics: m }),
  toggleMetric: (id) => set((st) => ({
    selectedMetrics: st.selectedMetrics.includes(id)
      ? st.selectedMetrics.filter((x) => x !== id)
      : [...st.selectedMetrics, id],
  })),

  currentJob: null,
  setJob: (j) => set({ currentJob: j }),
  jobEvents: [],
  appendEvent: (e) => set((st) => ({ jobEvents: [...st.jobEvents.slice(-499), e] })),
  clearEvents: () => set({ jobEvents: [] }),
  scenarioResult: null,
  setScenarioResult: (r) => set({ scenarioResult: r }),

  user: (() => {
    try {
      const raw = localStorage.getItem('rtagent_user')
      return raw ? JSON.parse(raw) : null
    } catch { return null }
  })(),
  setUser: (u) => {
    try {
      if (u) localStorage.setItem('rtagent_user', JSON.stringify(u))
      else localStorage.removeItem('rtagent_user')
    } catch { /* ignore */ }
    set({ user: u })
  },
}))

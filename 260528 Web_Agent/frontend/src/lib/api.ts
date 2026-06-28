import axios from 'axios'

export const api = axios.create({
  baseURL: '/api',
  timeout: 300_000,
})

export type Coord3 = [number, number, number]

/** API 에러를 항상 '사람이 읽는 문자열'로 변환한다.
 * FastAPI/Pydantic 422 의 detail 은 [{type,loc,msg,input,ctx}] 배열(또는 객체)이라
 * React 자식으로 그대로 렌더하면 "Objects are not valid as a React child" 로 크래시한다. */
export function formatApiError(ex: any): string {
  const detail = ex?.response?.data?.detail
  if (detail == null) return ex?.message ?? '알 수 없는 오류'
  if (typeof detail === 'string') return detail
  const fmt1 = (d: any): string => {
    if (d == null) return ''
    if (typeof d === 'string') return d
    const loc = Array.isArray(d.loc) ? d.loc.join('.') : (d.loc ?? '')
    const msg = d.msg ?? d.message ?? JSON.stringify(d)
    return loc ? `${loc}: ${msg}` : msg
  }
  if (Array.isArray(detail)) return detail.map(fmt1).join(' / ')
  return fmt1(detail)
}

export interface MetricSpec {
  id: string
  label: string
  description: string
  stages: string[]
  derived: boolean
  coverage_map: boolean
  needs_rx: boolean
  output_kinds: string[]
}

export interface SessionMeta {
  uuid: string
  label: string
  created_at_kst: string
  scene_name: string
  display_name?: string
  bs_rows: number
  bs_cols: number
  ue_rows: number
  ue_cols: number
  metrics: string[]
  notes: string
  user_id?: string
  user_name?: string
  status?: string
  progress?: number
  current_stage?: string
  engine?: string
  job_id?: string
  queued_at?: string
  started_at?: string
  finished_at?: string
  last_error?: string
}

export interface User { id: string; name: string }

export interface QueueScatter {
  points: [number, number, number | null][]
  bounds: [number, number, number, number]
  rsrp_min: number | null; rsrp_max: number | null
  tx_index: number; num_tx: number
  total_rx: number; done_rx: number; updated: number
}

export interface QueueItem {
  uuid: string; label: string; user_id: string; user_name: string; scene_name: string; display_name?: string
  status: string; progress: number; current_stage: string; engine: string
  queued_at: string; started_at: string; finished_at: string; last_error: string
  elapsed_sec: number | null
  bs_rows: number; bs_cols: number; ue_rows: number; ue_cols: number
  scatter?: QueueScatter
}

export interface QueueDashboard {
  queueing: QueueItem[]
  processing: QueueItem | null
  done: QueueItem[]
}

export interface SceneInfo {
  source_type: 'obj' | 'ply' | 'xml_zip' | 'xml_bundle'
  mesh_files: string[]
  materials: string[]
  aabb_min: number[]
  aabb_max: number[]
  center: number[]
  size: number[]
  n_vertices: number
  n_faces: number
  units: string
  material_assigned?: boolean
}

export interface JobStatus {
  job_id: string
  session_uuid: string
  state: 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled'
  progress: number
  current_stage: string
  stages_done: string[]
  started_at: string | null
  finished_at: string | null
  error: string | null
  output_paths: Record<string, string>
}

export const apiClient = {
  async health() {
    const { data } = await api.get('/health')
    return data
  },
  async metricsCatalog(): Promise<{ metrics: MetricSpec[] }> {
    const { data } = await api.get('/metrics/catalog')
    return data
  },
  async metricsPresets(): Promise<{ presets: Record<string, any> }> {
    const { data } = await api.get('/metrics/presets')
    return data
  },
  async resolveMetrics(metrics: string[]) {
    const { data } = await api.post('/metrics/resolve', { metrics })
    return data
  },
  async createSession(payload: any): Promise<SessionMeta> {
    const { data } = await api.post('/sessions', payload)
    return data
  },
  async listSessions(): Promise<{ sessions: SessionMeta[] }> {
    const { data } = await api.get('/sessions')
    return data
  },
  async getSession(uuid: string): Promise<SessionMeta> {
    const { data } = await api.get(`/sessions/${uuid}`)
    return data
  },
  async updateLabel(uuid: string, label: string): Promise<SessionMeta> {
    const { data } = await api.patch(`/sessions/${uuid}/label`, { label })
    return data
  },
  async deleteSession(uuid: string): Promise<{ deleted: boolean }> {
    const { data } = await api.delete(`/sessions/${uuid}`)
    return data
  },
  async uploadScene(uuid: string, file: File, material = 'itu_concrete') {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('material', material)
    const { data } = await api.post(`/sessions/${uuid}/scene`, fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return data as { scene_info: SceneInfo; scene_xml: string }
  },
  async uploadSceneBundle(uuid: string, files: File[]) {
    const fd = new FormData()
    for (const f of files) fd.append('files', f)
    const { data } = await api.post(`/sessions/${uuid}/scene_bundle`, fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 0,  // 다수 PLY(대용량) 업로드 → 타임아웃 해제
    })
    return data as { scene_info: SceneInfo; scene_xml: string }
  },
  async sceneInfo(uuid: string): Promise<SceneInfo> {
    const { data } = await api.get(`/sessions/${uuid}/scene_info`)
    return data
  },
  meshUrl(uuid: string, name: string) {
    return `/api/sessions/${uuid}/scene/mesh/${encodeURIComponent(name)}`
  },
  // --- Scene Library (PARK_2 RT 이식) ---
  async listSceneLibrary(): Promise<{ scenes: { name: string; ply: string; has_xml: boolean; has_skin?: boolean; skin_size?: number; size: number }[]; library_path: string }> {
    const { data } = await api.get('/scene_library')
    return data
  },
  async sceneFromLibrary(uuid: string, name: string, material = 'itu_concrete') {
    const { data } = await api.post(`/sessions/${uuid}/scene_from_library`, { name, material })
    return data as { scene_info: SceneInfo; scene_xml: string; source: string; name: string }
  },
  async uploadSceneLibrary(file: File, name: string): Promise<{ ok: boolean; name: string; ply: string; size: number }> {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('name', name)
    const { data } = await api.post('/scene_library', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
    return data
  },
  async deleteSceneLibrary(name: string): Promise<{ deleted: string[] }> {
    const { data } = await api.delete(`/scene_library/${encodeURIComponent(name)}`)
    return data
  },
  // --- Scene Library (Geo-Radio 번들: .xml + meshes/*.ply) ---
  async listSceneLibraryRadio(): Promise<{ scenes: { name: string; n_meshes: number; materials: string[]; n_faces: number; size: number; has_xml: boolean }[]; library_path: string }> {
    const { data } = await api.get('/scene_library_radio')
    return data
  },
  async uploadSceneLibraryRadio(files: File[], name: string): Promise<{ ok: boolean; name: string; n_meshes?: number; materials?: string[]; size?: number }> {
    const fd = new FormData()
    for (const f of files) fd.append('files', f)
    fd.append('name', name)
    const { data } = await api.post('/scene_library_radio', fd, {
      headers: { 'Content-Type': 'multipart/form-data' }, timeout: 0,
    })
    return data
  },
  async deleteSceneLibraryRadio(name: string): Promise<{ deleted: string[] }> {
    const { data } = await api.delete(`/scene_library_radio/${encodeURIComponent(name)}`)
    return data
  },
  async sceneFromLibraryRadio(uuid: string, name: string) {
    const { data } = await api.post(`/sessions/${uuid}/scene_from_library_radio`, { name })
    return data as { scene_info: SceneInfo; scene_xml: string; source: string; name: string }
  },
  // --- Geo-Radio 청크 스테이징 업로드 (.xml + meshes 폴더 분할 전송) ---
  async sceneStageReset(uuid: string): Promise<{ ok: boolean }> {
    const { data } = await api.post(`/sessions/${uuid}/scene_stage/reset`)
    return data
  },
  async sceneStageChunk(uuid: string, filename: string, chunkIndex: number, blob: Blob): Promise<{ ok: boolean; file: string; size: number }> {
    const fd = new FormData()
    fd.append('filename', filename)
    fd.append('chunk_index', String(chunkIndex))
    fd.append('data', blob, filename)
    const { data } = await api.post(`/sessions/${uuid}/scene_stage/chunk`, fd, {
      headers: { 'Content-Type': 'multipart/form-data' }, timeout: 0,
    })
    return data
  },
  async sceneStageBuild(uuid: string) {
    const { data } = await api.post(`/sessions/${uuid}/scene_stage/build`)
    return data as { scene_info: SceneInfo; scene_xml: string }
  },
  async sceneStageSaveLibrary(uuid: string, name: string): Promise<{ ok: boolean; name: string; n_meshes?: number; materials?: string[]; size?: number }> {
    const { data } = await api.post(`/sessions/${uuid}/scene_stage/save_library`, { name })
    return data
  },
  // --- Scene Skin (시각화용 텍스처 GLB, 라이브러리 1:1) ---
  async uploadSceneSkin(name: string, file: File): Promise<{ ok: boolean; name: string; skin: string; size: number }> {
    const fd = new FormData()
    fd.append('file', file)
    const { data } = await api.post(`/scene_library/${encodeURIComponent(name)}/skin`, fd, {
      headers: { 'Content-Type': 'multipart/form-data' }, timeout: 0,
    })
    return data
  },
  async deleteSceneSkin(name: string): Promise<{ deleted: string[] }> {
    const { data } = await api.delete(`/scene_library/${encodeURIComponent(name)}/skin`)
    return data
  },
  async sessionSkin(uuid: string): Promise<{ has_skin: boolean; name?: string; url?: string }> {
    const { data } = await api.get(`/sessions/${uuid}/skin`)
    return data
  },
  // --- 지면 레이캐스팅 (TX 스냅 / RX 지면 격자) ---
  async txSnap(uuid: string, x: number, y: number, offset_m = 2.0): Promise<{
    clicked: [number, number, number | null]; ground_z: number | null;
    tx: Coord3; offset_m: number; in_bounds: boolean
  }> {
    const { data } = await api.post(`/sessions/${uuid}/tx_snap`, { x, y, offset_m })
    return data
  },
  async rxGroundGrid(uuid: string, params: { grid_n?: number; margin?: number; rx_height?: number; raycasting_z?: number | null; max_height?: number | null; x_start?: number; x_stop?: number; y_start?: number; y_stop?: number; spacing?: number | null }): Promise<{
    positions: Coord3[]; count: number; candidates: number
  }> {
    const { data } = await api.post(`/sessions/${uuid}/rx_ground_grid`, params)
    return data
  },
  async submitJob(uuid: string, payload: any): Promise<JobStatus> {
    const { data } = await api.post(`/sessions/${uuid}/job`, payload)
    return data
  },
  async getJob(jobId: string): Promise<JobStatus> {
    const { data } = await api.get(`/jobs/${jobId}`)
    return data
  },
  async cancelJob(jobId: string) {
    const { data } = await api.post(`/jobs/${jobId}/cancel`)
    return data
  },
  // --- Users (이름표) ---
  async listUsers(): Promise<{ users: User[] }> {
    const { data } = await api.get('/users')
    return data
  },
  async addUser(name: string): Promise<User> {
    const { data } = await api.post('/users', { name })
    return data
  },
  async renameUser(id: string, name: string): Promise<User> {
    const { data } = await api.patch(`/users/${id}`, { name })
    return data
  },
  async deleteUser(id: string): Promise<{ deleted: boolean }> {
    const { data } = await api.delete(`/users/${id}`)
    return data
  },
  // --- Queue dashboard / session control ---
  async queue(): Promise<QueueDashboard> {
    const { data } = await api.get('/queue')
    return data
  },
  async cancelSession(uuid: string): Promise<{ ok: boolean; status: string }> {
    const { data } = await api.post(`/sessions/${uuid}/cancel`)
    return data
  },
  async getSessionConfig(uuid: string): Promise<{ exists: boolean; config?: any }> {
    const { data } = await api.get(`/sessions/${uuid}/config`)
    return data
  },
  async putSessionConfig(uuid: string, payload: any): Promise<{ ok: boolean }> {
    const { data } = await api.put(`/sessions/${uuid}/config`, payload)
    return data
  },
  async listFiles(uuid: string): Promise<{ files: { path: string; size: number; kind: string }[] }> {
    const { data } = await api.get(`/sessions/${uuid}/files`)
    return data
  },
  fileUrl(uuid: string, path: string) {
    return `/api/sessions/${uuid}/files/${path.split('/').map(encodeURIComponent).join('/')}`
  },
  fileDownloadUrl(uuid: string, path: string) {
    // ?download=1 → 서버가 Content-Disposition: attachment 를 붙여 첫 클릭에 바로 다운로드
    return `${this.fileUrl(uuid, path)}?download=1`
  },
  zipUrl(uuid: string) {
    return `/api/sessions/${uuid}/zip`
  },
  // --- Scenario Generator (Mobility) ---
  async scenarioData(uuid: string): Promise<{
    rx: { idx: number; x: number; y: number; z: number }[]
    tx: { idx: number; x: number; y: number; z: number }[]
    edges: number[][][]
    num_rx: number
    num_tx: number
  }> {
    const { data } = await api.get(`/sessions/${uuid}/scenario/data`)
    return data
  },
  async scenarioGenerate(uuid: string, body: {
    path: number[]; tx_index: number; fps: number; frames_per_step: number
  }): Promise<{
    script_path: string; script_rel: string; num_rx: number; num_rays: number; tx_name: string
    preview: {
      tx_name: string; tx_pos: number[]; fps: number; frames_per_step: number; total_frames: number
      steps: { name: string; pos: number[]; rays: { pts: number[][]; los: boolean; width: number }[] }[]
    }
  }> {
    const { data } = await api.post(`/sessions/${uuid}/scenario/generate`, body)
    return data
  },
  async scenarioChannelState(uuid: string, start: number, count: number): Promise<{
    start: number; count: number; total: number; tx_index: number
    items: {
      rx_idx: number; rsrp: number | null; los: boolean | null; num_paths: number
      valid_code?: number | null
      padp: { tau: number[]; aoa: number[]; power: number[] } | null
      r_rx: { m: number[]; disp: number; n: number } | null
      r_tx: { m: number[]; disp: number; n: number } | null
    }[]
  }> {
    const { data } = await api.get(`/sessions/${uuid}/scenario/channel_state`, { params: { start, count } })
    return data
  },
  async scenarioResult(uuid: string): Promise<{
    exists: boolean
    session_uuid?: string
    path?: number[]; tx_index?: number; fps?: number; frames_per_step?: number
    script_path?: string; script_rel?: string; num_rx?: number; num_rays?: number; tx_name?: string
    preview?: {
      tx_name: string; tx_pos: number[]; fps: number; frames_per_step: number; total_frames: number
      steps: { name: string; pos: number[]; rays: { pts: number[][]; los: boolean; width: number }[] }[]
    }
  }> {
    const { data } = await api.get(`/sessions/${uuid}/scenario/result`)
    return data
  },
  async rxInspect(uuid: string, rx: number, tx: number): Promise<{
    rx_idx: number; tx_index: number; rsrp_dbm: number | null; num_paths: number
    valid_code?: number | null
    padp_png_rel: string; pdp_png_rel: string; cov_png_rel: string
  }> {
    const { data } = await api.get(`/sessions/${uuid}/rx_inspect`, { params: { rx, tx } })
    return data
  },
}

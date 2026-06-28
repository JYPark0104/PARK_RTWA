import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiClient } from '../lib/api'
import { useStore } from '../store/useStore'
import { RXInspector } from '../components/RXInspector'
import { QueueDashboard } from '../components/QueueDashboard'

type FileItem = { path: string; size: number; kind: string }

export function ResultsPage() {
  const navigate = useNavigate()
  const session = useStore((s) => s.session)
  const setSession = useStore((s) => s.setSession)
  const setSceneInfo = useStore((s) => s.setSceneInfo)
  const setRT = useStore((s) => s.setRT)
  const setAntenna = useStore((s) => s.setAntenna)
  const setRX = useStore((s) => s.setRX)
  const clearTX = useStore((s) => s.clearTX)
  const addTXFull = useStore((s) => s.addTXFull)
  const setSelectedMetrics = useStore((s) => s.setSelectedMetrics)
  const user = useStore((s) => s.user)
  const [files, setFiles] = useState<FileItem[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [reloadKey, setReloadKey] = useState(0)   // 같은 세션 재오픈 시에도 상세 영역 강제 갱신
  const [opening, setOpening] = useState(false)   // 세션 열기/복원 중 로딩 오버레이

  async function refresh() {
    if (!session) return
    const { files } = await apiClient.listFiles(session.uuid)
    setFiles(files)
    setReloadKey((k) => k + 1)   // 결과 이미지 캐시버스트(재실행 후 최신 PNG 강제 로드)
  }
  useEffect(() => { refresh() }, [session?.uuid])

  // done 세션을 열어 store/config 복원 (열기 / scenario 생성 공용)
  async function openSession(uuid: string, goScenario: boolean) {
    setOpening(true)
    try {
      const meta = await apiClient.getSession(uuid)
      setSession(meta); setSceneInfo(null); setSelected(null)
      const r = await apiClient.getSessionConfig(uuid)
      const cfg = r.exists ? r.config : null
      if (cfg) {
        if (cfg.rt) setRT(cfg.rt)
        if (cfg.antenna?.simple) setAntenna(cfg.antenna.simple)
        clearTX()
        for (const t of cfg.tx_list ?? []) addTXFull(t)
        if (cfg.rx_clicks?.positions) setRX({ method: 'clicks', click_positions: cfg.rx_clicks.positions })
        else if (cfg.rx_grid) setRX(cfg.rx_grid)
        setSelectedMetrics(cfg.metrics?.metrics ?? [])
      }
      if (!goScenario) {
        // uuid 가 같아도(동일 세션 재오픈) 결과 파일을 즉시 재로딩 → F5 불필요
        try { const { files } = await apiClient.listFiles(uuid); setFiles(files) } catch { /* ignore */ }
        setReloadKey((k) => k + 1)   // RX Inspector 등 상세 영역 강제 remount
      }
    } catch { /* ignore */ } finally {
      setOpening(false)
    }
    navigate(goScenario ? '/scenario' : '/results')
  }

  const hero = useMemo(() => {
    const cov = files.find((f) => f.path.startsWith('Coverage_Map_Results/') && f.kind === 'png')
    if (cov) return { type: 'coverage' as const, file: cov }
    const rsrp = files.find((f) => f.path.startsWith('Derived_RSRP_Results/') && f.kind === 'png')
    if (rsrp) return { type: 'rsrp' as const, file: rsrp }
    return null
  }, [files])

  const tree = useMemo(() => buildTree(files), [files])

  const review = useMemo(() => {
    // batch RT 산출물 우선. 다중 TX면 *_allTX.png 를 우선 표시, 없으면 단일 TX(_TX0) / 임의.
    const pick = (kind: string) => {
      const cands = files.filter(
        (f) => f.kind === 'png' && f.path.includes('Batch_RT_Results') && f.path.includes(kind)
      )
      if (cands.length === 0) return undefined
      return (
        cands.find((f) => f.path.includes('allTX')) ??
        cands.find((f) => /_TX0\b/.test(f.path) || f.path.includes('_TX0.')) ??
        cands[0]
      )
    }
    const batchRsrp = pick('rsrp_heatmap')
    const batchLos = pick('los_map')
    if (batchRsrp || batchLos) {
      return { rsrp: batchRsrp, los: batchLos, has: true, source: 'batch' as const }
    }
    const rev = files.filter((f) => f.path.startsWith('Review_Results/') && f.kind === 'png')
    const rsrp = rev.find((f) => f.path.includes('RSRP'))
    const los = rev.find((f) => f.path.includes('LOS_NLOS'))
    return { rsrp, los, has: Boolean(rsrp || los), source: 'p1a' as const }
  }, [files])

  // 모든 hook 호출 이후에 분기 (hooks 순서 보존 — 흰 화면 방지)
  const dashboard = (
    <QueueDashboard
      title="7. RT Results — 작업 현황"
      currentUserId={user?.id}
      openLabel="RT Results 열기"
      onScenario={(uuid) => openSession(uuid, true)}
      onOpen={(uuid) => openSession(uuid, false)}
    />
  )

  if (!session) {
    return (
      <div className="space-y-4">
        {opening && <LoadingOverlay label="세션 여는 중…" />}
        {dashboard}
        <div className="card text-sm text-slate-500">
          세부 결과를 보려면 위 Done 목록에서 세션을 열거나, 1.Sessions 에서 선택하세요.
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {opening && <LoadingOverlay label="세션 여는 중…" />}
      {dashboard}
      <section className="card">
        <div className="flex justify-between items-center mb-3">
          <h2 className="text-lg font-semibold">결과 — {session.label}</h2>
          <div className="space-x-2">
            <button className="btn btn-secondary" onClick={refresh}>새로고침</button>
            <a className="inline-flex items-center gap-1.5 rounded-lg bg-brand-600 hover:bg-brand-700 text-white px-3 py-1.5 text-sm font-medium shadow-sm transition-colors"
               href={apiClient.zipUrl(session.uuid)} download>
              <DownloadIcon /> 전체 zip 다운로드
            </a>
          </div>
        </div>
        {hero ? (
          <div>
            <div className="text-xs text-slate-500 mb-1">
              {hero.type === 'coverage' ? 'Coverage Map (TX 기반)' : 'RSRP 2D map (RX scatter)'} — {hero.file.path}
            </div>
            <ImageWithSpinner src={`${apiClient.fileUrl(session.uuid, hero.file.path)}?v=${reloadKey}`} className="max-h-[600px] mx-auto" />
          </div>
        ) : (
          <div className="text-slate-500 text-sm">
            아직 Coverage Map 또는 RSRP 결과가 없습니다. metric 체크 후 잡을 실행하세요.
          </div>
        )}
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <section className="card lg:col-span-1">
          <h3 className="text-lg font-semibold mb-3">결과 파일 트리</h3>
          <div className="space-y-2">
            {Object.entries(tree).map(([dir, items]) => (
              <details key={dir} open className="group rounded-lg border border-slate-200 overflow-hidden">
                <summary className="cursor-pointer select-none flex items-center gap-2 px-3 py-2 bg-slate-50 hover:bg-slate-100 text-sm font-medium text-slate-700">
                  <span className="text-slate-400 transition-transform group-open:rotate-90">▶</span>
                  <span className="text-amber-500">📁</span>
                  <span className="font-mono truncate flex-1">{dir}</span>
                  <span className="text-xs px-1.5 py-0.5 rounded-full bg-slate-200 text-slate-600">{items.length}</span>
                </summary>
                <ul className="divide-y divide-slate-100">
                  {items.map((f) => {
                    const name = f.path.split('/').slice(-1)[0]
                    const isSel = selected === f.path
                    return (
                      <li key={f.path} className={`flex items-center ${isSel ? 'bg-brand-50' : 'hover:bg-slate-50'}`}>
                        <button
                          onClick={() => setSelected(f.path)}
                          className={`flex-1 min-w-0 flex items-center gap-2 px-3 py-1.5 text-left text-sm transition-colors ${isSel ? 'text-brand-700' : 'text-slate-600'}`}
                        >
                          <span className="shrink-0">{kindIcon(f.kind)}</span>
                          <span className={`font-mono truncate flex-1 ${isSel ? 'font-semibold' : ''}`}>{name}</span>
                          <span className="shrink-0 text-[11px] tabular-nums text-slate-400">{humanSize(f.size)}</span>
                        </button>
                        <a
                          href={apiClient.fileDownloadUrl(session.uuid, f.path)}
                          title="다운로드"
                          className="shrink-0 px-2.5 py-1.5 text-slate-400 hover:text-brand-600"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <DownloadIcon />
                        </a>
                      </li>
                    )
                  })}
                </ul>
              </details>
            ))}
            {files.length === 0 && <div className="text-slate-400 text-sm px-1 py-4 text-center">파일 없음</div>}
          </div>
        </section>

        <section className="card lg:col-span-2">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold">미리보기</h3>
            {selected && <DownloadButton uuid={session.uuid} path={selected} />}
          </div>
          {!selected && <div className="text-slate-500 text-sm">왼쪽에서 파일을 선택하세요.</div>}
          {selected && <Preview uuid={session.uuid} path={selected} version={reloadKey} />}
        </section>
      </div>

      {review.has && (
        <section className="card">
          <h3 className="text-lg font-semibold mb-3">Review — RT 결과 요약 (all TX){review.source === 'batch' ? ' · batch 엔진' : ''}</h3>
          <div className="grid grid-cols-1 gap-4">
            <figure className="space-y-1">
              <figcaption className="text-sm font-medium text-slate-700">RSRP 맵</figcaption>
              {review.rsrp ? (
                <ImageWithSpinner src={`${apiClient.fileUrl(session.uuid, review.rsrp.path)}?v=${reloadKey}`} className="w-full rounded border border-slate-200" minHeight={280} />
              ) : (
                <div className="text-slate-400 text-sm">RSRP 맵 없음</div>
              )}
            </figure>
            <figure className="space-y-1">
              <figcaption className="text-sm font-medium text-slate-700">LoS / NLoS 맵</figcaption>
              {review.los ? (
                <ImageWithSpinner src={`${apiClient.fileUrl(session.uuid, review.los.path)}?v=${reloadKey}`} className="w-full rounded border border-slate-200" minHeight={280} />
              ) : (
                <div className="text-slate-400 text-sm">LoS/NLoS 맵 없음</div>
              )}
            </figure>
          </div>
        </section>
      )}

      <section className="card">
        <h3 className="text-lg font-semibold mb-2">RX Inspector (인터랙티브)</h3>
        <p className="text-xs text-slate-500 mb-3">
          batch RT 결과에서 RX 를 클릭해 해당 RX 의 PADP / RSRP / Covariance 를 확인합니다.
        </p>
        <RXInspector key={`${session.uuid}:${reloadKey}`} uuid={session.uuid} />
      </section>
    </div>
  )
}

function Preview({ uuid, path, version = 0 }: { uuid: string; path: string; version?: number }) {
  const ext = path.split('.').slice(-1)[0].toLowerCase()
  const url = `${apiClient.fileUrl(uuid, path)}?v=${version}`
  if (ext === 'png' || ext === 'jpg' || ext === 'jpeg' || ext === 'webp') {
    return <ImageWithSpinner src={url} className="max-h-[600px] mx-auto" minHeight={300} />
  }
  if (ext === 'csv') {
    return <CsvPreview url={url} />
  }
  if (ext === 'json') {
    return <JsonPreview url={url} />
  }
  return (
    <div className="text-sm space-y-3">
      <div className="text-slate-500">이 형식({ext})은 미리보기를 지원하지 않습니다. 아래 버튼으로 다운로드하세요.</div>
      <DownloadButton uuid={uuid} path={path} label={`${path.split('/').slice(-1)[0]} 다운로드`} />
    </div>
  )
}

/** 다운로드 아이콘 (arrow-down-tray) */
function DownloadIcon({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 3v12" />
      <path d="m7 12 5 5 5-5" />
      <path d="M5 21h14" />
    </svg>
  )
}

/** 예쁜 다운로드 버튼 (서버 attachment 강제 → 첫 클릭에 바로 받아짐) */
function DownloadButton({ uuid, path, label = '다운로드' }: { uuid: string; path: string; label?: string }) {
  return (
    <a
      href={apiClient.fileDownloadUrl(uuid, path)}
      className="inline-flex items-center gap-1.5 rounded-lg bg-brand-600 hover:bg-brand-700 active:bg-brand-800 text-white px-3 py-1.5 text-sm font-medium shadow-sm transition-colors"
    >
      <DownloadIcon /> {label}
    </a>
  )
}

/** 작은 회전 스피너 */
function Spinner({ size = 28 }: { size?: number }) {
  return (
    <span
      className="inline-block animate-spin rounded-full border-2 border-slate-300 border-t-brand-600"
      style={{ width: size, height: size }}
    />
  )
}

/** 화면 중앙 로딩 오버레이 (세션 열기 등) */
function LoadingOverlay({ label }: { label: string }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-[1px]">
      <div className="flex flex-col items-center gap-3 bg-white rounded-xl px-8 py-6 shadow-xl">
        <Spinner size={36} />
        <div className="text-sm text-slate-600">{label}</div>
      </div>
    </div>
  )
}

/** 로딩 중 스피너를 겹쳐 보여주는 이미지 (PNG 로딩이 느릴 때 진행 표시) */
function ImageWithSpinner({ src, className, minHeight = 200 }: { src: string; className?: string; minHeight?: number }) {
  const [state, setState] = useState<'loading' | 'done' | 'error'>('loading')
  // src 가 바뀌면 다시 로딩 상태로
  useEffect(() => { setState('loading') }, [src])
  return (
    <div className="relative w-full" style={state === 'loading' ? { minHeight } : undefined}>
      {state === 'loading' && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-slate-50/70 rounded">
          <Spinner />
          <span className="text-xs text-slate-500">이미지 불러오는 중…</span>
        </div>
      )}
      {state === 'error' && (
        <div className="absolute inset-0 flex items-center justify-center text-xs text-red-500">이미지 로드 실패</div>
      )}
      <img
        src={src}
        loading="lazy"
        decoding="async"
        onLoad={() => setState('done')}
        onError={() => setState('error')}
        className={className}
        style={state === 'loading' ? { opacity: 0 } : { opacity: 1, transition: 'opacity .15s' }}
      />
    </div>
  )
}

/** 파일 종류별 아이콘 */
function kindIcon(kind: string): string {
  switch (kind) {
    case 'png':
    case 'jpg':
    case 'jpeg':
    case 'webp': return '🖼️'
    case 'csv': return '📊'
    case 'json': return '🧾'
    case 'npz':
    case 'npy': return '📦'
    case 'usda':
    case 'obj':
    case 'ply': return '🧊'
    case 'log':
    case 'txt': return '📄'
    default: return '📄'
  }
}

function CsvPreview({ url }: { url: string }) {
  const [rows, setRows] = useState<string[][]>([])
  useEffect(() => {
    fetch(url).then((r) => r.text()).then((t) => {
      const lines = t.split(/\r?\n/).filter(Boolean).slice(0, 100)
      setRows(lines.map((ln) => ln.split(',')))
    })
  }, [url])
  if (rows.length === 0) return <div className="text-slate-500 text-sm">로딩…</div>
  const head = rows[0]
  return (
    <table className="text-xs w-full">
      <thead><tr>{head.map((h, i) => <th key={i} className="text-left border-b py-1">{h}</th>)}</tr></thead>
      <tbody>
        {rows.slice(1).map((r, i) => (
          <tr key={i} className="border-b">{r.map((c, j) => <td key={j} className="py-0.5 pr-2 font-mono">{c}</td>)}</tr>
        ))}
      </tbody>
    </table>
  )
}

function JsonPreview({ url }: { url: string }) {
  const [text, setText] = useState('')
  useEffect(() => {
    fetch(url).then((r) => r.text()).then(setText)
  }, [url])
  return <pre className="text-xs whitespace-pre-wrap max-h-96 overflow-auto">{text}</pre>
}

function buildTree(files: FileItem[]) {
  const tree: Record<string, FileItem[]> = {}
  for (const f of files) {
    const parts = f.path.split('/')
    const dir = parts.length > 1 ? parts.slice(0, -1).join('/') : '(root)'
    if (!tree[dir]) tree[dir] = []
    tree[dir].push(f)
  }
  return tree
}

function humanSize(b: number) {
  if (b < 1024) return `${b} B`
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} kB`
  if (b < 1024 ** 3) return `${(b / 1024 ** 2).toFixed(1)} MB`
  return `${(b / 1024 ** 3).toFixed(1)} GB`
}

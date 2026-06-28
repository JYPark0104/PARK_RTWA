import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiClient } from '../lib/api'
import { SceneViewer } from '../components/SceneViewer'
import { useStore } from '../store/useStore'

export function ScenePage() {
  const navigate = useNavigate()
  const session = useStore((s) => s.session)
  const sceneInfo = useStore((s) => s.sceneInfo)
  const setSceneInfo = useStore((s) => s.setSceneInfo)
  const [material, setMaterial] = useState<'itu_concrete' | 'itu_ceiling_board' | 'itu_glass'>('itu_concrete')
  const [twinMode, setTwinMode] = useState<'geo' | 'geo_radio'>('geo')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [library, setLibrary] = useState<{ name: string; ply: string; has_xml: boolean; has_skin?: boolean; skin_size?: number; size: number }[]>([])
  const [libSel, setLibSel] = useState<string>('')
  const [editLib, setEditLib] = useState(false)
  const [newLibName, setNewLibName] = useState('')
  const [libBusy, setLibBusy] = useState(false)
  // Geo-Radio 번들 라이브러리
  const [radioLib, setRadioLib] = useState<{ name: string; n_meshes: number; materials: string[]; n_faces: number; size: number; has_xml: boolean }[]>([])
  const [radioLibSel, setRadioLibSel] = useState<string>('')
  const [editRadioLib, setEditRadioLib] = useState(false)
  const [newRadioLibName, setNewRadioLibName] = useState('')
  // Geo-Radio 선택 파일(분리 입력) + 청크 업로드 진행률
  const [xmlFile, setXmlFile] = useState<File | null>(null)
  const [plyFiles, setPlyFiles] = useState<File[]>([])
  const [uploadPct, setUploadPct] = useState<number | null>(null)

  function refreshRadioLibrary() {
    apiClient.listSceneLibraryRadio().then((r) => {
      setRadioLib(r.scenes)
      if (r.scenes.length && !radioLibSel) setRadioLibSel(r.scenes[0].name)
    }).catch(() => {})
  }

  function refreshLibrary() {
    apiClient.listSceneLibrary().then((r) => {
      setLibrary(r.scenes)
      if (r.scenes.length && !libSel) setLibSel(r.scenes[0].name)
    }).catch(() => {})
  }

  useEffect(() => {
    if (!session) return
    apiClient.sceneInfo(session.uuid).then(setSceneInfo).catch(() => setSceneInfo(null))
  }, [session?.uuid])

  useEffect(() => {
    apiClient.listSceneLibrary().then((r) => {
      setLibrary(r.scenes)
      if (r.scenes.length && !libSel) setLibSel(r.scenes[0].name)
    }).catch(() => {})
    apiClient.listSceneLibraryRadio().then((r) => {
      setRadioLib(r.scenes)
      if (r.scenes.length && !radioLibSel) setRadioLibSel(r.scenes[0].name)
    }).catch(() => {})
  }, [])

  async function onUploadLibrary(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    const nm = (newLibName || file.name.replace(/\.[^.]+$/, '')).trim()
    if (!nm) { setErr('라이브러리 이름을 입력하세요.'); e.target.value = ''; return }
    setLibBusy(true); setErr(null)
    try {
      await apiClient.uploadSceneLibrary(file, nm)
      setNewLibName('')
      refreshLibrary()
      setLibSel(nm)
    } catch (ex: any) {
      setErr(ex?.response?.data?.detail ?? ex.message)
    } finally { setLibBusy(false); e.target.value = '' }
  }

  async function onDeleteLibrary(name: string) {
    if (!window.confirm(`라이브러리 "${name}" 를 삭제할까요? (.ply/.xml/.glb 영구 삭제)`)) return
    setLibBusy(true); setErr(null)
    try {
      await apiClient.deleteSceneLibrary(name)
      if (libSel === name) setLibSel('')
      refreshLibrary()
    } catch (ex: any) {
      setErr(ex?.response?.data?.detail ?? ex.message)
    } finally { setLibBusy(false) }
  }

  async function onUploadSkin(name: string, e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    setLibBusy(true); setErr(null)
    try {
      await apiClient.uploadSceneSkin(name, file)   // 대용량 GLB (timeout 0)
      refreshLibrary()
    } catch (ex: any) {
      setErr(ex?.response?.data?.detail ?? ex.message)
    } finally { setLibBusy(false); e.target.value = '' }
  }

  async function onDeleteSkin(name: string) {
    if (!window.confirm(`"${name}" 의 텍스처 스킨(.glb)을 삭제할까요?`)) return
    setLibBusy(true); setErr(null)
    try {
      await apiClient.deleteSceneSkin(name)
      refreshLibrary()
    } catch (ex: any) {
      setErr(ex?.response?.data?.detail ?? ex.message)
    } finally { setLibBusy(false) }
  }

  if (!session) return <div className="card">먼저 세션을 선택하세요.</div>

  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    if (!session) return
    const file = e.target.files?.[0]
    if (!file) return
    setBusy(true); setErr(null)
    try {
      const res = await apiClient.uploadScene(session.uuid, file, material)
      setSceneInfo(res.scene_info)
    } catch (ex: any) {
      setErr(ex?.response?.data?.detail ?? ex.message)
    } finally {
      setBusy(false)
      e.target.value = ''
    }
  }

  async function onLoadLibrary() {
    if (!session || !libSel) return
    setBusy(true); setErr(null)
    try {
      const res = await apiClient.sceneFromLibrary(session.uuid, libSel, material)
      setSceneInfo(res.scene_info)
    } catch (ex: any) {
      setErr(ex?.response?.data?.detail ?? ex.message)
    } finally {
      setBusy(false)
    }
  }

  // 분리 입력 핸들러: .xml 1개 / meshes 폴더(여러 .ply)
  function onSelectXml(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (f) { setXmlFile(f); setErr(null) }
  }
  function onSelectMeshes(e: React.ChangeEvent<HTMLInputElement>) {
    const list = e.target.files
    if (!list) return
    const plys = Array.from(list).filter((f) => f.name.toLowerCase().endsWith('.ply'))
    if (plys.length === 0) { setErr('선택한 폴더에 .ply 가 없습니다.'); return }
    setPlyFiles(plys); setErr(null)
  }

  // 청크 업로드: 큰 단일 요청 대신 4MB 단위로 나눠 스테이징에 이어붙임 (프록시 한도 회피)
  const CHUNK = 4 * 1024 * 1024
  async function stageUpload(uuid: string, files: File[]) {
    await apiClient.sceneStageReset(uuid)
    const totalChunks = files.reduce((a, f) => a + Math.max(1, Math.ceil(f.size / CHUNK)), 0)
    let done = 0
    setUploadPct(0)
    for (const f of files) {
      const n = Math.max(1, Math.ceil(f.size / CHUNK))
      for (let i = 0; i < n; i++) {
        const blob = f.slice(i * CHUNK, Math.min((i + 1) * CHUNK, f.size))
        await apiClient.sceneStageChunk(uuid, f.name, i, blob)
        done += 1
        setUploadPct(Math.round((done / totalChunks) * 100))
      }
    }
  }

  function validateBundleSelection(): boolean {
    if (!xmlFile || plyFiles.length === 0) {
      setErr('scene .xml 1개와 meshes 폴더(.ply)를 모두 선택하세요.')
      return false
    }
    return true
  }

  async function onApplyBundle() {
    if (!session || !validateBundleSelection()) return
    setBusy(true); setErr(null)
    try {
      await stageUpload(session.uuid, [xmlFile!, ...plyFiles])
      const res = await apiClient.sceneStageBuild(session.uuid)
      setSceneInfo(res.scene_info)
    } catch (ex: any) {
      setErr(ex?.response?.data?.detail ?? ex.message ?? '업로드 실패')
    } finally { setBusy(false); setUploadPct(null) }
  }

  async function onSaveBundleToLibrary() {
    if (!session || !validateBundleSelection()) return
    const nm = (newRadioLibName || xmlFile!.name.replace(/\.[^.]+$/, '')).trim()
    if (!nm) { setErr('라이브러리 이름을 입력하세요.'); return }
    setLibBusy(true); setErr(null)
    try {
      await stageUpload(session.uuid, [xmlFile!, ...plyFiles])
      await apiClient.sceneStageSaveLibrary(session.uuid, nm)
      setNewRadioLibName('')
      refreshRadioLibrary()
      setRadioLibSel(nm)
    } catch (ex: any) {
      setErr(ex?.response?.data?.detail ?? ex.message ?? '저장 실패')
    } finally { setLibBusy(false); setUploadPct(null) }
  }

  async function onLoadRadioLibrary() {
    if (!session || !radioLibSel) return
    setBusy(true); setErr(null)
    try {
      const res = await apiClient.sceneFromLibraryRadio(session.uuid, radioLibSel)
      setSceneInfo(res.scene_info)
    } catch (ex: any) {
      setErr(ex?.response?.data?.detail ?? ex.message)
    } finally {
      setBusy(false)
    }
  }

  async function onDeleteRadioLibrary(name: string) {
    if (!window.confirm(`Geo-Radio 라이브러리 "${name}" 를 삭제할까요? (번들 폴더 영구 삭제)`)) return
    setLibBusy(true); setErr(null)
    try {
      await apiClient.deleteSceneLibraryRadio(name)
      if (radioLibSel === name) setRadioLibSel('')
      refreshRadioLibrary()
    } catch (ex: any) {
      setErr(ex?.response?.data?.detail ?? ex.message)
    } finally { setLibBusy(false) }
  }

  return (
    <div className="space-y-4">
      <section className="card">
        <h2 className="text-lg font-semibold mb-3">씬 선택</h2>

        {/* Twin 모드 분기: 기하 전용 vs 재질 부여 완료 */}
        <div className="mb-3 flex flex-wrap gap-2">
          <button
            onClick={() => setTwinMode('geo')}
            className={`px-3 py-1.5 rounded text-sm border ${twinMode === 'geo' ? 'bg-sky-600 text-white border-sky-600' : 'bg-white text-slate-600 border-slate-300 hover:bg-slate-50'}`}
          >
            Geo Env. Twin (기하만 · 기본재질)
          </button>
          <button
            onClick={() => setTwinMode('geo_radio')}
            className={`px-3 py-1.5 rounded text-sm border ${twinMode === 'geo_radio' ? 'bg-emerald-600 text-white border-emerald-600' : 'bg-white text-slate-600 border-slate-300 hover:bg-slate-50'}`}
          >
            Geo-Radio Env. Twin (재질 부여됨)
          </button>
        </div>
        <p className="text-xs text-slate-500 mb-3">
          {twinMode === 'geo'
            ? '기하학적 맵(.obj/.ply)을 올리고 단일 기본재질을 부여합니다. (MGA 산출물)'
            : 'MAA 가 재질을 부여한 씬(scene .xml + meshes 폴더의 여러 .ply)을 업로드합니다. XML 의 ITU 재질이 그대로 RT 에 사용됩니다.'}
        </p>

        {twinMode === 'geo' && (<>
        {/* Scene Library (서버 내 사전 보관 .ply 맵) */}
        <div className="flex flex-wrap items-center gap-3 mb-3 pb-3 border-b border-slate-200">
          <span className="text-sm font-medium text-slate-700">라이브러리에서 불러오기:</span>
          <select className="input w-72" value={libSel} onChange={(e) => setLibSel(e.target.value)} disabled={busy || library.length === 0}>
            {library.length === 0 && <option value="">(scene_library 비어있음)</option>}
            {library.map((s) => (
              <option key={s.name} value={s.name}>{s.name}{s.has_xml ? '' : ' (xml 없음)'}</option>
            ))}
          </select>
          <button className="btn btn-primary" onClick={onLoadLibrary} disabled={busy || !libSel}>불러오기</button>
          <button className="btn btn-secondary" onClick={() => setEditLib((v) => !v)}>
            {editLib ? '편집 닫기' : '라이브러리 편집'}
          </button>
        </div>

        {editLib && (
          <div className="mb-3 pb-3 border-b border-slate-200 bg-slate-50 rounded p-3 space-y-3">
            <div>
              <h4 className="text-sm font-semibold text-slate-700 mb-1">기존 라이브러리</h4>
              {library.length === 0 && <div className="text-xs text-slate-400">비어 있습니다.</div>}
              <ul className="space-y-1">
                {library.map((s) => (
                  <li key={s.name} className="flex items-center justify-between text-sm border-b border-slate-100 py-1 gap-2">
                    <span className="font-mono text-slate-600 min-w-0 truncate">
                      {s.name}
                      <span className="text-slate-400 ml-2 text-xs">{(s.size / 1024 / 1024).toFixed(1)} MB{s.has_xml ? '' : ' · xml없음'}</span>
                      {s.has_skin
                        ? <span className="ml-2 text-xs px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700">🎨 스킨 {((s.skin_size ?? 0) / 1024 / 1024).toFixed(0)}MB</span>
                        : <span className="ml-2 text-xs text-slate-400">스킨 없음</span>}
                    </span>
                    <span className="flex items-center gap-1 shrink-0">
                      {s.has_skin ? (
                        <button className="btn text-xs btn-secondary" disabled={libBusy} onClick={() => onDeleteSkin(s.name)}>스킨삭제</button>
                      ) : (
                        <label className="btn text-xs btn-secondary cursor-pointer">
                          스킨(.glb) 업로드
                          <input type="file" accept=".glb" className="hidden" disabled={libBusy}
                            onChange={(e) => onUploadSkin(s.name, e)} />
                        </label>
                      )}
                      <button className="btn text-xs bg-red-50 text-red-600 border border-red-200 hover:bg-red-100"
                        disabled={libBusy} onClick={() => onDeleteLibrary(s.name)}>삭제</button>
                    </span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="flex flex-wrap items-end gap-2">
              <label className="text-sm">
                <span className="block text-xs text-slate-600 mb-0.5">새 라이브러리 이름</span>
                <input className="input w-56" value={newLibName} onChange={(e) => setNewLibName(e.target.value)} placeholder="예: MyCity_v1" disabled={libBusy} />
              </label>
              <label className="text-sm">
                <span className="block text-xs text-slate-600 mb-0.5">.ply 업로드</span>
                <input type="file" accept=".ply" onChange={onUploadLibrary} disabled={libBusy} />
              </label>
              {libBusy && <span className="text-xs text-slate-500">처리 중...</span>}
            </div>
            <p className="text-xs text-slate-500">※ 이름 미입력 시 파일명이 사용됩니다. 현재 .ply 만 지원합니다.</p>
          </div>
        )}

        <h3 className="text-sm font-medium text-slate-700 mb-2">또는 직접 업로드 (.obj / .ply / .zip)</h3>
        <div className="flex flex-wrap items-center gap-3">
          <input type="file" accept=".obj,.ply,.zip" onChange={onUpload} disabled={busy} />
          <label className="text-sm text-slate-600">
            기본 재질:
            <select className="input ml-2 w-44" value={material} onChange={(e) => setMaterial(e.target.value as any)}>
              <option value="itu_concrete">itu_concrete</option>
              <option value="itu_ceiling_board">itu_ceiling_board</option>
              <option value="itu_glass">itu_glass</option>
            </select>
          </label>
          {busy && <span className="text-sm text-slate-500">변환 중...</span>}
          {err && <span className="text-sm text-red-600">{err}</span>}
        </div>
        </>)}

        {twinMode === 'geo_radio' && (
          <div className="space-y-3">
            {/* Geo-Radio 번들 라이브러리에서 불러오기 */}
            <div className="flex flex-wrap items-center gap-3 pb-3 border-b border-slate-200">
              <span className="text-sm font-medium text-slate-700">라이브러리에서 불러오기:</span>
              <select className="input w-72" value={radioLibSel} onChange={(e) => setRadioLibSel(e.target.value)} disabled={busy || radioLib.length === 0}>
                {radioLib.length === 0 && <option value="">(Geo-Radio 라이브러리 비어있음)</option>}
                {radioLib.map((s) => (
                  <option key={s.name} value={s.name}>{s.name} · {s.n_meshes}개 메시 · {s.materials.length}종 재질</option>
                ))}
              </select>
              <button className="btn btn-primary" onClick={onLoadRadioLibrary} disabled={busy || !radioLibSel}>불러오기</button>
              <button className="btn btn-secondary" onClick={() => setEditRadioLib((v) => !v)}>
                {editRadioLib ? '편집 닫기' : '라이브러리 편집'}
              </button>
            </div>

            {editRadioLib && (
              <div className="pb-3 border-b border-slate-200 bg-slate-50 rounded p-3">
                <h4 className="text-sm font-semibold text-slate-700 mb-1">기존 Geo-Radio 라이브러리</h4>
                {radioLib.length === 0 && <div className="text-xs text-slate-400">비어 있습니다.</div>}
                <ul className="space-y-1">
                  {radioLib.map((s) => (
                    <li key={s.name} className="flex items-center justify-between text-sm border-b border-slate-100 py-1 gap-2">
                      <span className="font-mono text-slate-600 min-w-0 truncate">
                        {s.name}
                        <span className="text-slate-400 ml-2 text-xs">
                          {(s.size / 1024 / 1024).toFixed(1)} MB · {s.n_meshes} 메시 · {s.n_faces.toLocaleString()} faces
                        </span>
                        <span className="ml-2 text-xs px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700">
                          {s.materials.join(', ')}
                        </span>
                      </span>
                      <button className="btn text-xs bg-red-50 text-red-600 border border-red-200 hover:bg-red-100 shrink-0"
                        disabled={libBusy} onClick={() => onDeleteRadioLibrary(s.name)}>삭제</button>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* 또는 직접 선택: .xml 1개 + meshes 폴더 (분리 입력 → 4MB 청크 업로드) */}
            <div className="bg-slate-50 rounded p-3 space-y-2">
              <h3 className="text-sm font-medium text-slate-700">또는 직접 업로드 (scene .xml + meshes 폴더)</h3>
              <div className="flex flex-wrap items-end gap-5">
                <label className="text-sm">
                  <span className="block text-xs text-slate-600 mb-0.5">① scene .xml 파일</span>
                  <input type="file" accept=".xml" onChange={onSelectXml} disabled={busy || libBusy} />
                  {xmlFile && <div className="text-xs text-emerald-600 mt-0.5">✓ {xmlFile.name}</div>}
                </label>
                <label className="text-sm">
                  <span className="block text-xs text-slate-600 mb-0.5">② meshes 폴더 (통째로 선택)</span>
                  <input
                    type="file"
                    multiple
                    ref={(el) => { if (el) { el.setAttribute('webkitdirectory', ''); el.setAttribute('directory', '') } }}
                    onChange={onSelectMeshes}
                    disabled={busy || libBusy}
                  />
                  {plyFiles.length > 0 && <div className="text-xs text-emerald-600 mt-0.5">✓ {plyFiles.length}개 .ply 선택됨</div>}
                </label>
              </div>

              <div className="flex flex-wrap items-center gap-3 pt-1">
                <button className="btn btn-primary" onClick={onApplyBundle}
                  disabled={busy || libBusy || !xmlFile || plyFiles.length === 0}>
                  이 세션에 적용
                </button>
                <span className="text-slate-300">|</span>
                <input className="input w-48" value={newRadioLibName} onChange={(e) => setNewRadioLibName(e.target.value)}
                  placeholder="라이브러리 이름 (예: GHMTwin_v1)" disabled={busy || libBusy} />
                <button className="btn btn-secondary" onClick={onSaveBundleToLibrary}
                  disabled={busy || libBusy || !xmlFile || plyFiles.length === 0}>
                  라이브러리에 저장
                </button>
              </div>

              <div className="flex items-center gap-3">
                {uploadPct !== null && (
                  <div className="flex items-center gap-2">
                    <div className="w-40 h-2 bg-slate-200 rounded overflow-hidden">
                      <div className="h-full bg-emerald-500" style={{ width: `${uploadPct}%` }} />
                    </div>
                    <span className="text-xs text-slate-600">업로드 {uploadPct}%</span>
                  </div>
                )}
                {(busy || libBusy) && uploadPct === null && <span className="text-sm text-slate-500">빌드/검증 중...</span>}
                {err && <span className="text-sm text-red-600">{err}</span>}
              </div>

              <p className="text-xs text-slate-500 leading-relaxed">
                ① <span className="font-mono">scene .xml</span> 1개를 선택하고, ② <span className="font-mono">meshes</span> 폴더를{' '}
                <b>통째로</b> 선택하면 폴더 안 <span className="font-mono">.ply</span> 들이 자동 포함됩니다.<br />
                업로드는 4MB 청크로 나눠 전송되어 대용량 PLY 도 안정적입니다 (Network Error 회피).
                XML 에 지정된 ITU 재질(itu_glass, itu_marble, itu_wood …)이 그대로 보존되어 RT 에 사용됩니다.
              </p>
            </div>
          </div>
        )}
        {sceneInfo && (
          <div className="mt-3 text-xs text-slate-500 grid grid-cols-2 lg:grid-cols-4 gap-2">
            <div>source_type: <span className="font-mono">{sceneInfo.source_type}</span></div>
            <div>
              mode:{' '}
              <span className={`font-mono ${sceneInfo.material_assigned ? 'text-emerald-600' : 'text-sky-600'}`}>
                {sceneInfo.material_assigned ? 'Geo-Radio (재질부여)' : 'Geo (기본재질)'}
              </span>
            </div>
            <div>vertices: <span className="font-mono">{sceneInfo.n_vertices}</span></div>
            <div>faces: <span className="font-mono">{sceneInfo.n_faces}</span></div>
            <div>meshes: <span className="font-mono">{sceneInfo.mesh_files.length}</span></div>
            <div>AABB min: <span className="font-mono">[{sceneInfo.aabb_min.map((v) => v.toFixed(1)).join(', ')}]</span></div>
            <div>AABB max: <span className="font-mono">[{sceneInfo.aabb_max.map((v) => v.toFixed(1)).join(', ')}]</span></div>
            <div>size: <span className="font-mono">{sceneInfo.size.map((v) => v.toFixed(1)).join(' × ')}</span></div>
            <div className="col-span-2 lg:col-span-4">materials: <span className="font-mono">{sceneInfo.materials.join(', ')}</span></div>
          </div>
        )}
      </section>

      {sceneInfo && (
        <section className="card">
          <div className="flex justify-between items-center mb-2">
            <h3 className="font-semibold">3D 미리보기</h3>
            <button className="btn btn-primary" onClick={() => navigate('/devices')}>다음: TX/RX 배치</button>
          </div>
          <SceneViewer uuid={session.uuid} sceneInfo={sceneInfo} tx={[]} rx={[]} />
        </section>
      )}
    </div>
  )
}

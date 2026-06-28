import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiClient, type SessionMeta, type User } from '../lib/api'
import { useStore } from '../store/useStore'
import { QueueDashboard } from '../components/QueueDashboard'

export function SessionsPage() {
  const navigate = useNavigate()
  const setSession = useStore((s) => s.setSession)
  const setSceneInfo = useStore((s) => s.setSceneInfo)
  const setRT = useStore((s) => s.setRT)
  const setAntenna = useStore((s) => s.setAntenna)
  const setRX = useStore((s) => s.setRX)
  const clearTX = useStore((s) => s.clearTX)
  const addTXFull = useStore((s) => s.addTXFull)
  const setSelectedMetrics = useStore((s) => s.setSelectedMetrics)
  const antenna = useStore((s) => s.antenna)
  const user = useStore((s) => s.user)
  const setUser = useStore((s) => s.setUser)

  const [users, setUsers] = useState<User[]>([])
  const [editUsers, setEditUsers] = useState(false)
  const [newUser, setNewUser] = useState('')
  const [sessions, setSessions] = useState<SessionMeta[]>([])
  const [name, setName] = useState('Jonggak')
  const [notes, setNotes] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  async function refreshUsers() {
    try { const { users } = await apiClient.listUsers(); setUsers(users) } catch { /* ignore */ }
  }
  async function refreshSessions() {
    try { const { sessions } = await apiClient.listSessions(); setSessions(sessions) } catch { /* ignore */ }
  }

  useEffect(() => { refreshUsers(); refreshSessions() }, [])
  // idle(구성 중) 세션 갱신용 가벼운 폴링
  useEffect(() => {
    const t = window.setInterval(refreshSessions, 2000)
    return () => window.clearInterval(t)
  }, [])

  // ── User 관리 ──────────────────────────────────────────────
  async function addUser() {
    const nm = newUser.trim()
    if (!nm) return
    setErr(null)
    try {
      const u = await apiClient.addUser(nm)
      setNewUser('')
      await refreshUsers()
      setUser(u)
    } catch (ex: any) { setErr(ex?.response?.data?.detail ?? ex.message) }
  }
  async function renameUser(u: User) {
    const nl = prompt('새 user 이름', u.name)
    if (!nl || nl === u.name) return
    await apiClient.renameUser(u.id, nl)
    await refreshUsers()
    if (user?.id === u.id) setUser({ ...u, name: nl })
  }
  async function deleteUser(u: User) {
    if (!window.confirm(`user "${u.name}" 를 삭제할까요? (세션은 삭제되지 않음)`)) return
    await apiClient.deleteUser(u.id)
    await refreshUsers()
    if (user?.id === u.id) setUser(null)
  }

  // ── 세션 생성 ──────────────────────────────────────────────
  async function createSession() {
    if (!user) return
    setBusy(true); setErr(null)
    try {
      const meta = await apiClient.createSession({
        scene_name: name,
        antenna: { mode: 'simple', simple: antenna },
        metrics: { metrics: [] },
        notes,
        user_id: user.id,
        user_name: user.name,
      })
      setSession(meta)
      setSceneInfo(null)
      await refreshSessions()
      navigate('/scene')
    } finally { setBusy(false) }
  }

  // ── 세션 열기 (config 복원) ────────────────────────────────
  function restoreConfig(cfg: any) {
    if (!cfg) return
    if (cfg.rt) setRT(cfg.rt)
    if (cfg.antenna?.simple) setAntenna(cfg.antenna.simple)
    clearTX()
    for (const t of cfg.tx_list ?? []) addTXFull(t)
    if (cfg.rx_clicks?.positions) setRX({ method: 'clicks', click_positions: cfg.rx_clicks.positions })
    else if (cfg.rx_grid) setRX(cfg.rx_grid)
    setSelectedMetrics(cfg.metrics?.metrics ?? [])
  }
  async function openSession(uuid: string) {
    try {
      const meta = await apiClient.getSession(uuid)
      setSession(meta)
      setSceneInfo(null)
      const r = await apiClient.getSessionConfig(uuid)
      if (r.exists && r.config) restoreConfig(r.config)
    } catch { /* ignore */ }
    navigate('/scene')
  }
  async function renameSession(uuid: string, label: string) {
    await apiClient.updateLabel(uuid, label); refreshSessions()
  }
  async function deleteSession(uuid: string) {
    if (!window.confirm('이 세션을 삭제할까요? 폴더의 모든 결과가 영구 삭제됩니다.')) return
    await apiClient.deleteSession(uuid); refreshSessions()
  }
  async function cancelSession(uuid: string) {
    await apiClient.cancelSession(uuid); refreshSessions()
  }

  // 구성 중(아직 실행 안 한) 세션 = status 가 비었거나 idle
  const idleSessions = sessions.filter((s) => !s.status || s.status === 'idle')

  return (
    <div className="space-y-6">
      {/* ── User 선택 ── */}
      <section className="card">
        <div className="flex justify-between items-center mb-3">
          <h2 className="text-lg font-semibold">User 선택</h2>
          <button className="btn btn-secondary text-xs" onClick={() => setEditUsers((v) => !v)}>
            {editUsers ? '완료' : 'user 편집'}
          </button>
        </div>
        <div className="flex flex-wrap items-center gap-2 mb-3">
          {users.map((u) => (
            <div key={u.id} className="flex items-center gap-1">
              <button
                className={`btn text-sm ${user?.id === u.id ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setUser(u)}>
                👤 {u.name}
              </button>
              {editUsers && (
                <>
                  <button className="text-xs text-slate-500 hover:text-slate-800" onClick={() => renameUser(u)}>✎</button>
                  <button className="text-xs text-red-500 hover:text-red-700" onClick={() => deleteUser(u)}>🗑</button>
                </>
              )}
            </div>
          ))}
          {users.length === 0 && <span className="text-sm text-slate-400">등록된 user가 없습니다. 새로 추가하세요.</span>}
        </div>
        <div className="flex items-center gap-2">
          <input className="input w-48" placeholder="새 user 이름" value={newUser}
            onChange={(e) => setNewUser(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') addUser() }} />
          <button className="btn btn-secondary" onClick={addUser}>새 user 추가</button>
          {user && <span className="text-sm text-emerald-700 ml-2">선택됨: <b>{user.name}</b></span>}
        </div>
        {err && <div className="text-sm text-red-600 mt-2">{err}</div>}
      </section>

      {/* ── 새 세션 만들기 (user 선택 시 활성화) ── */}
      <section className={`card ${user ? '' : 'opacity-50 pointer-events-none'}`}>
        <h2 className="text-lg font-semibold mb-3">
          새 세션 만들기 {user ? <span className="text-sm text-slate-500">— {user.name}</span> : <span className="text-sm text-slate-400">(먼저 user를 선택하세요)</span>}
        </h2>
        <div className="flex flex-wrap gap-3 items-end">
          <div>
            <label className="block text-xs font-medium text-slate-600">Scene name</label>
            <input className="input w-48" value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <div className="flex-1 min-w-[200px]">
            <label className="block text-xs font-medium text-slate-600">Notes (optional)</label>
            <input className="input" value={notes} onChange={(e) => setNotes(e.target.value)} />
          </div>
          <button className="btn btn-primary" disabled={busy || !user} onClick={createSession}>
            {busy ? '...' : '세션 생성'}
          </button>
        </div>
      </section>

      {/* ── 구성 중 세션 (아직 실행 전) ── */}
      {idleSessions.length > 0 && (
        <section className="card">
          <h2 className="text-lg font-semibold mb-2">구성 중 세션 (미실행 {idleSessions.length})</h2>
          <table className="w-full text-sm">
            <thead className="text-slate-400 text-xs">
              <tr className="border-b">
                <th className="text-left py-1 pr-2">User</th>
                <th className="text-left pr-2">Scene</th>
                <th className="text-left pr-2">Antenna(BS/UE)</th>
                <th className="text-left pr-2">Label</th>
                <th className="text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {idleSessions.map((m) => {
                const mine = !!user && (!m.user_id || m.user_id === user.id)
                return (
                  <tr key={m.uuid} className="border-b hover:bg-slate-50">
                    <td className="py-1.5 pr-2 font-medium text-slate-700 whitespace-nowrap">👤 {m.user_name || '—'}</td>
                    <td className="pr-2 text-slate-600 whitespace-nowrap">{m.display_name || m.scene_name || '—'}</td>
                    <td className="pr-2 font-mono text-xs text-slate-500 whitespace-nowrap">{m.bs_rows}×{m.bs_cols} / {m.ue_rows}×{m.ue_cols}</td>
                    <td className="pr-2 font-mono text-xs text-slate-500 max-w-[260px] truncate" title={m.label}>{m.label}</td>
                    <td className="text-right space-x-1 whitespace-nowrap">
                      {mine ? (
                        <>
                          <button className="btn btn-secondary text-xs" onClick={() => openSession(m.uuid)}>열기</button>
                          <button className="btn btn-secondary text-xs" onClick={() => {
                            const nl = prompt('새 라벨', m.label); if (nl && nl !== m.label) renameSession(m.uuid, nl)
                          }}>이름변경</button>
                          <button className="btn text-xs bg-red-50 text-red-600 border border-red-200 hover:bg-red-100"
                            onClick={() => deleteSession(m.uuid)}>삭제</button>
                        </>
                      ) : (
                        <span className="text-xs text-slate-400">다른 user 세션</span>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </section>
      )}

      {/* ── 큐 대시보드 (Queueing / Processing / Done) ── */}
      <QueueDashboard
        title="세션 큐"
        currentUserId={user?.id}
        onOpen={openSession}
        onCancel={cancelSession}
        onDelete={deleteSession}
        onRename={renameSession}
      />
    </div>
  )
}

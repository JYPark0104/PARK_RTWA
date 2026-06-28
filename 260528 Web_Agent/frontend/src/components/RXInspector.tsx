import { useEffect, useMemo, useRef, useState } from 'react'
import { apiClient } from '../lib/api'

type RX = { idx: number; x: number; y: number; z: number }
type TX = { idx: number; x: number; y: number; z: number }
type SData = { rx: RX[]; tx: TX[]; edges: number[][][]; num_rx: number; num_tx: number }
type Inspect = {
  rx_idx: number; tx_index: number; rsrp_dbm: number | null; num_paths: number
  valid_code?: number | null
  padp_png_rel: string; pdp_png_rel: string; cov_png_rel: string
}

/**
 * 7.RT Results 인터랙티브 RX 인스펙터 — 지도에서 RX 클릭 → 해당 (TX,RX) 의
 * PADP / PDP(RSRP) / Covariance 결과를 로딩해 표시. (batch RT channel_data 필요)
 */
export function RXInspector({ uuid }: { uuid: string }) {
  const [data, setData] = useState<SData | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [txIndex, setTxIndex] = useState(0)
  const [selRx, setSelRx] = useState<number | null>(null)
  const [insp, setInsp] = useState<Inspect | null>(null)
  const [loading, setLoading] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    apiClient.scenarioData(uuid)
      .then(setData)
      .catch((e) => setErr(e?.response?.data?.detail ?? e.message))
  }, [uuid])

  const bounds = useMemo(() => {
    if (!data) return null
    let a = Infinity, b = Infinity, c = -Infinity, d = -Infinity
    const u = (x: number, y: number) => { if (x < a) a = x; if (x > c) c = x; if (y < b) b = y; if (y > d) d = y }
    for (const r of data.rx) u(r.x, r.y)
    for (const e of data.edges) { u(e[0][0], e[0][1]); u(e[1][0], e[1][1]) }
    if (!isFinite(a)) return null
    const px = (c - a) * 0.03 + 1, py = (d - b) * 0.03 + 1
    return { minX: a - px, minY: b - py, maxX: c + px, maxY: d + py }
  }, [data])

  const CW = 760, CH = 560
  const tf = useMemo(() => {
    if (!bounds) return null
    const scale = Math.min(CW / (bounds.maxX - bounds.minX), CH / (bounds.maxY - bounds.minY))
    const ox = (CW - (bounds.maxX - bounds.minX) * scale) / 2
    const oy = (CH - (bounds.maxY - bounds.minY) * scale) / 2
    return {
      toScreen: (x: number, y: number): [number, number] => [(x - bounds.minX) * scale + ox, CH - ((y - bounds.minY) * scale + oy)],
      toWorld: (px: number, py: number): [number, number] => [(px - ox) / scale + bounds.minX, (CH - py - oy) / scale + bounds.minY],
    }
  }, [bounds])

  useEffect(() => {
    const cv = canvasRef.current
    if (!cv || !data || !tf) return
    const ctx = cv.getContext('2d')!
    ctx.fillStyle = '#0f0f1a'; ctx.fillRect(0, 0, CW, CH)
    ctx.strokeStyle = 'rgba(74,144,217,0.45)'; ctx.lineWidth = 0.4; ctx.beginPath()
    for (const e of data.edges) {
      const [x1, y1] = tf.toScreen(e[0][0], e[0][1]); const [x2, y2] = tf.toScreen(e[1][0], e[1][1])
      ctx.moveTo(x1, y1); ctx.lineTo(x2, y2)
    }
    ctx.stroke()
    const rad = data.rx.length > 1500 ? 1.4 : data.rx.length > 600 ? 2 : 3
    ctx.fillStyle = 'rgba(0,229,255,0.7)'
    for (const r of data.rx) { const [sx, sy] = tf.toScreen(r.x, r.y); ctx.beginPath(); ctx.arc(sx, sy, rad, 0, Math.PI * 2); ctx.fill() }
    // 선택 RX 강조
    if (selRx != null) {
      const r = data.rx.find((rr) => rr.idx === selRx)
      if (r) { const [sx, sy] = tf.toScreen(r.x, r.y); ctx.strokeStyle = 'orange'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(sx, sy, 7, 0, Math.PI * 2); ctx.stroke() }
    }
    data.tx.forEach((t) => {
      const [sx, sy] = tf.toScreen(t.x, t.y); const sel = t.idx === txIndex
      ctx.fillStyle = sel ? '#ff2d2d' : '#888'
      ctx.beginPath(); ctx.arc(sx, sy, sel ? 7 : 5, 0, Math.PI * 2); ctx.fill()
      ctx.fillStyle = sel ? '#ff7777' : '#aaa'; ctx.font = 'bold 11px sans-serif'; ctx.fillText(`TX${t.idx}`, sx + 8, sy - 8)
    })
  }, [data, tf, selRx, txIndex])

  async function onClick(e: React.MouseEvent) {
    if (!data || !tf) return
    const cv = canvasRef.current!; const rect = cv.getBoundingClientRect()
    const px = (e.clientX - rect.left) * (CW / rect.width)
    const py = (e.clientY - rect.top) * (CH / rect.height)
    const [wx, wy] = tf.toWorld(px, py)
    let best = -1, bestD = Infinity
    for (const r of data.rx) { const dd = (r.x - wx) ** 2 + (r.y - wy) ** 2; if (dd < bestD) { bestD = dd; best = r.idx } }
    if (best < 0) return
    setSelRx(best); setLoading(true); setInsp(null)
    try {
      const res = await apiClient.rxInspect(uuid, best, txIndex)
      setInsp(res)
    } catch (ex: any) {
      setErr(ex?.response?.data?.detail ?? ex.message)
    } finally { setLoading(false) }
  }

  if (err && !data) {
    return <div className="text-sm text-slate-500">RX 인스펙터: batch RT 결과(channel_data)가 필요합니다. ({err})</div>
  }
  if (!data) return <div className="text-sm text-slate-500">RX 맵 로딩 중...</div>

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <div>
        <div className="flex items-center gap-2 mb-2">
          <span className="text-sm text-slate-600">TX:</span>
          {data.tx.map((t) => (
            <button key={t.idx} className={`btn text-xs ${t.idx === txIndex ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => { setTxIndex(t.idx); if (selRx != null) { setLoading(true); apiClient.rxInspect(uuid, selRx, t.idx).then(setInsp).finally(() => setLoading(false)) } }}>
              TX{t.idx}
            </button>
          ))}
        </div>
        <canvas ref={canvasRef} width={CW} height={CH} onClick={onClick}
          className="w-full rounded bg-slate-950 cursor-pointer" />
        <div className="text-xs text-slate-500 mt-1">
          지도에서 RX(파란 점)를 클릭하면 해당 RX의 PADP/RSRP/Covariance 가 표시됩니다.
          {selRx != null && <> · 선택: <b className="font-mono">RX{selRx}</b></>}
        </div>
      </div>

      <div className="space-y-3">
        {loading && (
          <div className="space-y-2">
            <div className="text-sm text-slate-600">RX{selRx} 결과 계산 중...</div>
            <div className="w-full h-2 bg-slate-200 rounded overflow-hidden">
              <div className="h-2 bg-brand-600 rounded animate-[indet_1.1s_ease-in-out_infinite]"
                   style={{ width: '40%' }} />
            </div>
            <style>{`@keyframes indet{0%{margin-left:-40%}50%{margin-left:60%}100%{margin-left:100%}}`}</style>
          </div>
        )}
        {insp && !loading && (
          <>
            <div className="text-sm">
              <b>RX{insp.rx_idx}</b> / TX{insp.tx_index} ·
              RSRP <span className="font-mono">{insp.rsrp_dbm != null ? `${insp.rsrp_dbm.toFixed(2)} dBm` : 'Dead'}</span> ·
              경로 <span className="font-mono">{insp.num_paths}개</span>
              {insp.valid_code != null && insp.valid_code !== 0 && (
                <span className={`ml-2 px-1.5 py-0.5 rounded text-[11px] ${insp.valid_code === 1 ? 'bg-slate-200 text-slate-600' : 'bg-rose-100 text-rose-700'}`}>
                  {insp.valid_code === 1 ? 'dead (유효경로 0)' : 'rt_fail (음수지연/효율미달)'}
                </span>
              )}
            </div>
            <Img uuid={uuid} rel={insp.pdp_png_rel} caption="PDP (delay vs power) + RSRP" />
            <Img uuid={uuid} rel={insp.padp_png_rel} caption="PADP (AoA vs delay, 크기=power)" />
            <Img uuid={uuid} rel={insp.cov_png_rel} caption="Covariance (|R_RX|, |R_TX|)" />
          </>
        )}
        {!insp && !loading && <div className="text-sm text-slate-500">RX를 선택하세요.</div>}
      </div>
    </div>
  )
}

function Img({ uuid, rel, caption }: { uuid: string; rel: string; caption: string }) {
  return (
    <figure className="space-y-1">
      <figcaption className="text-xs font-medium text-slate-600">{caption}</figcaption>
      <img src={`${apiClient.fileUrl(uuid, rel)}?t=${Date.now()}`} className="w-full rounded border border-slate-200" />
    </figure>
  )
}

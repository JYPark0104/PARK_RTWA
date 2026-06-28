import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useStore } from '../store/useStore'

export function RTConfigPage() {
  const navigate = useNavigate()
  const rt = useStore((s) => s.rt)
  const setRT = useStore((s) => s.setRT)
  const antenna = useStore((s) => s.antenna)
  const setAntenna = useStore((s) => s.setAntenna)
  const session = useStore((s) => s.session)

  const [adv, setAdv] = useState(false)
  const [covAdv, setCovAdv] = useState(false)

  if (!session) return <div className="card text-sm text-slate-500">먼저 1.Sessions 에서 세션을 선택하세요.</div>

  return (
    <div className="space-y-4">
      <section className="card">
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <h2 className="text-lg font-semibold">안테나 (rows × cols)</h2>
          <div className="flex gap-2">
            <button
              className="btn btn-secondary text-xs"
              title="SISO: BS 1×1, UE 1×1 로 설정"
              onClick={() => setAntenna({ bs_rows: 1, bs_cols: 1, ue_rows: 1, ue_cols: 1 })}>
              SISO mode (1×1 / 1×1)
            </button>
            <button
              className="btn btn-secondary text-xs"
              title="기본값 BS 32×32, UE 4×4 로 복원"
              onClick={() => setAntenna({ bs_rows: 32, bs_cols: 32, ue_rows: 4, ue_cols: 4 })}>
              기본 (32×32 / 4×4)
            </button>
          </div>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <NumberInput label="BS rows" value={antenna.bs_rows} on={(v) => setAntenna({ bs_rows: v })} integer />
          <NumberInput label="BS cols" value={antenna.bs_cols} on={(v) => setAntenna({ bs_cols: v })} integer />
          <NumberInput label="UE rows" value={antenna.ue_rows} on={(v) => setAntenna({ ue_rows: v })} integer />
          <NumberInput label="UE cols" value={antenna.ue_cols} on={(v) => setAntenna({ ue_cols: v })} integer />
        </div>
        <p className="text-xs text-slate-500 mt-2">
          patch 4×4 + grid 나머지로 자동 분해됩니다. (예: 64×64 → patch 4×4, grid 16×16 = 4096 AE)
        </p>
      </section>

      <section className="card">
        <h2 className="text-lg font-semibold mb-2">Ray Tracing 엔진</h2>
        <div className="flex flex-wrap gap-2">
          <button
            className={`btn ${rt.engine === 'p1a' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setRT({ engine: 'p1a' })}>
            P1A Ray Tracing
          </button>
          <button
            className={`btn ${rt.engine === 'batch' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setRT({ engine: 'batch' })}>
            batch Ray Tracing
          </button>
        </div>
        {rt.engine === 'p1a' ? (
          <p className="text-xs text-slate-500 mt-2">
            기존 P1A 파이프라인 (251218 표준 NPZ → metric 단계 호환). 선택한 metric이 그대로 실행됩니다.
          </p>
        ) : (
          <div className="mt-2 space-y-2">
            <p className="text-xs text-amber-600">
              PARK_2 batch_rx 방식 RT — TX별 배치 진행 로그 + RSRP/LoS/PDP/PADP/공분산/USD 산출.
              (이 모드에선 metric 단계는 건너뜁니다.)
            </p>
            <div className="w-48">
              <NumberInput label="batch_size (배치당 RX)" integer value={rt.batch_size} on={(v) => setRT({ batch_size: v })} />
            </div>
            <Toggle
              label="Randomized RX batch sequence (무작위 RX 배치 — 가로줄 클러스터 완화)"
              value={rt.random_batch}
              on={(v) => setRT({ random_batch: v })}
            />
            <p className="text-xs text-slate-500">
              ON: RX 순서를 무작위로 섞어 각 배치를 [RX3, RX843, RX0 …]처럼 구성합니다.
              결과(좌표/RSRP/Ray)는 항상 원래 RX 인덱스로 정렬되어 저장됩니다.
            </p>
          </div>
        )}
      </section>

      <section className="card">
        <h2 className="text-lg font-semibold mb-3">Ray Tracing 기본</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <NumberInput label="frequency (GHz)" value={rt.frequency_ghz} on={(v) => setRT({ frequency_ghz: v })} />
          <NumberInput label="max_depth" value={rt.max_depth} on={(v) => setRT({ max_depth: v })} integer />
          <NumberInput label="seed" value={rt.seed} on={(v) => setRT({ seed: v })} integer />
          <NumberInput label="num_interesting_paths" value={rt.num_interesting_paths} on={(v) => setRT({ num_interesting_paths: v })} integer />
          <NumberInput label="max_rays_per_pair" value={rt.max_rays_per_pair} on={(v) => setRT({ max_rays_per_pair: v })} integer />
        </div>

        <button className="btn btn-secondary mt-3" onClick={() => setAdv(!adv)}>
          {adv ? '고급 옵션 숨기기' : '고급 옵션 표시'}
        </button>

        {adv && (
          <div className="mt-3 space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
              <Toggle label="PathSolver: LoS" value={rt.pathsolver_los} on={(v) => setRT({ pathsolver_los: v })} />
              <Toggle label="PathSolver: specular_reflection" value={rt.pathsolver_specular_reflection} on={(v) => setRT({ pathsolver_specular_reflection: v })} />
              <Toggle label="PathSolver: diffuse_reflection" value={rt.pathsolver_diffuse_reflection} on={(v) => setRT({ pathsolver_diffuse_reflection: v })} />
              <Toggle label="PathSolver: refraction" value={rt.pathsolver_refraction} on={(v) => setRT({ pathsolver_refraction: v })} />
              <Toggle label="PathSolver: synthetic_array" value={rt.pathsolver_synthetic_array} on={(v) => setRT({ pathsolver_synthetic_array: v })} />
              <Toggle label="PathSolver: diffraction" value={rt.pathsolver_diffraction} on={(v) => setRT({ pathsolver_diffraction: v })} />
              <Toggle label="PathSolver: edge_diffraction" value={rt.pathsolver_edge_diffraction} on={(v) => setRT({ pathsolver_edge_diffraction: v })} />
              <Toggle label="PathSolver: diffraction_lit_region" value={rt.pathsolver_diffraction_lit_region} on={(v) => setRT({ pathsolver_diffraction_lit_region: v })} />
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
              <NumberInput label="num_samples (samples/src)" integer value={rt.num_samples} on={(v) => setRT({ num_samples: v })} />
              <NumberInput label="max_num_paths" integer value={rt.max_num_paths} on={(v) => setRT({ max_num_paths: v })} />
              <NumberInput label="TX 지면 이격 [m]" value={rt.tx_ground_offset_m} on={(v) => setRT({ tx_ground_offset_m: v })} />
            </div>

            <div>
              <div className="text-xs font-semibold text-slate-500 mb-1">산란(Scattering) 설정</div>
              <p className="text-xs text-slate-400 mb-2">
                메시의 <b>기본 재질(ITU 타입)은 2.Scene 에서 지정</b>합니다. (재질 타입이 주파수에 따라
                permittivity·conductivity 를 자동 결정) 아래는 그 위에 얹는 <b>산란 거동</b> 튜닝입니다.
              </p>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
                <NumberInput label="ITU scattering coeff" value={rt.itu_scattering_coeff} on={(v) => setRT({ itu_scattering_coeff: v })} />
                <NumberInput label="ITU XPD coeff" value={rt.itu_xpd_coeff} on={(v) => setRT({ itu_xpd_coeff: v })} />
                <label className="block">
                  <span className="text-xs font-medium text-slate-600">scattering_pattern</span>
                  <select className="input" value={rt.scattering_pattern}
                    onChange={(e) => setRT({ scattering_pattern: e.target.value as any })}>
                    <option value="lambertian">lambertian</option>
                    <option value="directive">directive</option>
                    <option value="backscattering">backscattering</option>
                  </select>
                </label>
                {rt.scattering_pattern === 'directive' && (
                  <NumberInput label="directive_alpha_r" integer value={rt.directive_alpha_r} on={(v) => setRT({ directive_alpha_r: v })} />
                )}
                {rt.scattering_pattern === 'backscattering' && (
                  <>
                    <NumberInput label="backscat_alpha_r" integer value={rt.backscattering_alpha_r} on={(v) => setRT({ backscattering_alpha_r: v })} />
                    <NumberInput label="backscat_alpha_i" integer value={rt.backscattering_alpha_i} on={(v) => setRT({ backscattering_alpha_i: v })} />
                    <NumberInput label="backscat_lambda" value={rt.backscattering_lambda} on={(v) => setRT({ backscattering_lambda: v })} />
                  </>
                )}
              </div>
            </div>

            <div>
              <div className="text-xs font-semibold text-slate-500 mb-1">안테나 패턴 / 편파</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <SelectInput label="tx_pattern" value={rt.tx_pattern} on={(v) => setRT({ tx_pattern: v })} options={PATTERN_OPTIONS} />
                <SelectInput label="tx_polarization" value={rt.tx_polarization} on={(v) => setRT({ tx_polarization: v })} options={POLARIZATION_OPTIONS} />
                <SelectInput label="rx_pattern" value={rt.rx_pattern} on={(v) => setRT({ rx_pattern: v })} options={PATTERN_OPTIONS} />
                <SelectInput label="rx_polarization" value={rt.rx_polarization} on={(v) => setRT({ rx_polarization: v })} options={POLARIZATION_OPTIONS} />
              </div>
            </div>
          </div>
        )}
      </section>

      <section className="card">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Coverage Map (TX 기반 광역 heatmap)</h2>
          <Toggle label="활성화" value={rt.coverage_map.enabled}
            on={(v) => setRT({ coverage_map: { ...rt.coverage_map, enabled: v } })} />
        </div>
        {rt.coverage_map.enabled && (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
              <NumberInput label="cell size x [m]" value={rt.coverage_map.cell_size_x}
                on={(v) => setRT({ coverage_map: { ...rt.coverage_map, cell_size_x: v } })} />
              <NumberInput label="cell size y [m]" value={rt.coverage_map.cell_size_y}
                on={(v) => setRT({ coverage_map: { ...rt.coverage_map, cell_size_y: v } })} />
              <NumberInput label="height z [m]"    value={rt.coverage_map.height_m}
                on={(v) => setRT({ coverage_map: { ...rt.coverage_map, height_m: v } })} />
              <NumberInput label="samples / TX (×1e6)"
                value={Math.round(rt.coverage_map.samples_per_tx / 1e6)}
                on={(v) => setRT({ coverage_map: { ...rt.coverage_map, samples_per_tx: v * 1e6 } })}
                integer
              />
            </div>
            <button className="btn btn-secondary mt-3" onClick={() => setCovAdv(!covAdv)}>
              {covAdv ? '고급 covg 옵션 숨기기' : '고급 covg 옵션'}
            </button>
            {covAdv && (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mt-3 text-sm">
                <NumberInput label="max_depth" integer value={rt.coverage_map.max_depth}
                  on={(v) => setRT({ coverage_map: { ...rt.coverage_map, max_depth: v } })} />
                <Toggle label="specular_reflection" value={rt.coverage_map.specular_reflection}
                  on={(v) => setRT({ coverage_map: { ...rt.coverage_map, specular_reflection: v } })} />
                <Toggle label="diffuse_reflection" value={rt.coverage_map.diffuse_reflection}
                  on={(v) => setRT({ coverage_map: { ...rt.coverage_map, diffuse_reflection: v } })} />
                <Toggle label="refraction" value={rt.coverage_map.refraction}
                  on={(v) => setRT({ coverage_map: { ...rt.coverage_map, refraction: v } })} />
              </div>
            )}
          </>
        )}
      </section>

      <button className="btn btn-primary" onClick={() => navigate('/metrics')}>다음: 결과 metric 선택</button>
    </div>
  )
}

function NumberInput({ label, value, on, integer = false }: { label: string; value: number; on: (v: number) => void; integer?: boolean }) {
  return (
    <label className="block">
      <span className="text-xs font-medium text-slate-600">{label}</span>
      <input
        type="number"
        className="input"
        value={value}
        onChange={(e) => {
          const n = integer ? parseInt(e.target.value) : parseFloat(e.target.value)
          if (!isNaN(n)) on(n)
        }}
      />
    </label>
  )
}

function Toggle({ label, value, on }: { label: string; value: boolean; on: (v: boolean) => void }) {
  return (
    <label className="flex items-center gap-2 text-sm">
      <input type="checkbox" checked={value} onChange={(e) => on(e.target.checked)} />
      {label}
    </label>
  )
}

// Sionna PlanarArray 가 받는 한정 옵션 → 드롭다운으로 제공 (직접 타이핑 제거)
const PATTERN_OPTIONS = ['iso', 'dipole', 'hw_dipole', 'tr38901']
const POLARIZATION_OPTIONS = ['V', 'H', 'VH', 'cross']

function SelectInput({ label, value, on, options }: {
  label: string; value: string; on: (v: string) => void; options: string[]
}) {
  // 저장된 값이 옵션에 없으면(예전 자유입력 값) 목록에 포함시켜 표시
  const opts = options.includes(value) ? options : [value, ...options]
  return (
    <label className="block">
      <span className="text-xs font-medium text-slate-600">{label}</span>
      <select className="input" value={value} onChange={(e) => on(e.target.value)}>
        {opts.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </label>
  )
}

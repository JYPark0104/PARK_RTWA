"""raygen_tr38901.py — P1A의 TR 38.901 sub-ray 생성기(RayGen) 자기완결 벤더링.

원본: forked_25x/P1A_RT_to_Rays_2509v6_web.py 의 RayGen 클래스.
- p1a_config 의존을 제거하고 TR38.901 상수를 모듈 내부에 박음 (값 동일).
- numpy + math 만 의존 (sionna/tf 불필요) → intg_writer 가 path-level → ray-level 확장에 사용.

핵심: path(클러스터) 1개 → M(=20) 개 sub-ray 로 확장. 각 ray 에 source_path_idx,
los_nlos_flag 를 기록. 생성은 확률적(np.random)이라, 재현성을 위해 rng(Generator) 주입 가능.
"""

from __future__ import annotations

import math
import numpy as np

SPEED_OF_LIGHT = 299_792_458.0
M_RAYS_PER_PATH = 20  # 클러스터당 sub-ray 수 (P1A M_actual)

# --- TR 38.901 v16.1.0 Table 7.5-6 UMa 상수 (P1A p1a_config 값과 동일) ---
TR38901_C_ZSA = 7
TR38901_MU_LGZSD = 0.5
LOS_C_ASA, LOS_C_ASD, LOS_R_TAU, LOS_N_CLUSTERS = 11, 5, 2.5, 12
NLOS_C_ASA, NLOS_C_ASD, NLOS_R_TAU, NLOS_N_CLUSTERS = 15, 2, 2.3, 20
LOS_COMB_N, NLOS_COMB_N = 35, 25


class RayGenTR38901:
    """TR 38.901 sub-ray 생성기 (자기완결). carrier_frequency 단위: Hz."""

    def __init__(self, carrier_frequency: float, los_scenario: int, rng: np.random.Generator | None = None):
        self.carrier_frequency = float(carrier_frequency)
        self.los_scenario = int(los_scenario)
        self.wavelength = SPEED_OF_LIGHT / self.carrier_frequency
        self.rng = rng if rng is not None else np.random.default_rng()

        self.c_DS = max(0.25, 6.5622 - 3.4084 * math.log10(self.carrier_frequency)) * 1e-9
        self.c_ZSA = TR38901_C_ZSA
        self.mu_lgZSD = TR38901_MU_LGZSD
        self.c_ZSD = 3 / 8 * 10 ** (self.mu_lgZSD)

        if self.los_scenario == 1:  # LOS
            self.c_ASA, self.c_ASD = LOS_C_ASA, LOS_C_ASD
            self.r_tau, self.N = LOS_R_TAU, LOS_N_CLUSTERS
        else:                        # NLOS
            self.c_ASA, self.c_ASD = NLOS_C_ASA, NLOS_C_ASD
            self.r_tau, self.N = NLOS_R_TAU, NLOS_N_CLUSTERS

    def _subray_angle_offsets(self, spread_a, spread_b):
        # P1A generate_subray_angles: N(0, spread/7) 분포 (M개)
        phi = self.rng.normal(0, spread_a / 7, M_RAYS_PER_PATH)
        theta = self.rng.normal(0, spread_b / 7, M_RAYS_PER_PATH)
        return phi, theta

    def subray(self, tau_c, phi_r_c, phi_t_c, theta_r_c, theta_t_c, power_c):
        """클러스터(path) 1개 → M개 sub-ray (P1A subrayProposed 동치)."""
        M = M_RAYS_PER_PATH
        tau_s = tau_c + self.rng.exponential(scale=self.c_DS, size=M)
        phi_r_off, theta_r_off = self._subray_angle_offsets(self.c_ASA, self.c_ASD)
        phi_t_off, theta_t_off = self._subray_angle_offsets(self.c_ASD, self.c_ZSA)
        power_s = power_c * self.rng.exponential(scale=1.0, size=M) / M
        return (tau_s, power_s,
                phi_r_c + phi_r_off, phi_t_c + phi_t_off,
                theta_r_c + theta_r_off, theta_t_c + theta_t_off)


def determine_los_scenario(path_los_flags) -> int:
    """가장 강한(=첫) path 가 LoS 면 1, 아니면 0. (P1A determine_scenario 동치)"""
    flags = np.asarray(path_los_flags).reshape(-1)
    if flags.size == 0:
        return 0
    return 1 if int(flags[0]) == 1 else 0


def expand_paths_to_rays(
    path_tau, path_power, path_phi_r, path_phi_t, path_theta_r, path_theta_t,
    path_los_flags, carrier_frequency, rng: np.random.Generator | None = None,
) -> dict:
    """(tx,rx) 한 쌍의 path-level 배열들 → ray-level dict (가변길이, path별 M개 확장).

    입력 배열들은 모두 길이 = 유효 path 수 K (1D). 반환 dict 의 각 배열 길이 = sum(M)=K*M.
    P1A RayGen.generate_from_paths 와 동치 (단 self-contained + rng 주입).
    """
    los_scenario = determine_los_scenario(path_los_flags)
    gen = RayGenTR38901(carrier_frequency, los_scenario, rng=rng)

    tau, power, phi_r, phi_t, theta_r, theta_t, src_idx, los_flag = ([] for _ in range(8))
    K = len(np.asarray(path_tau).reshape(-1))
    for p in range(K):
        ts, ps, prr, ptt, trr, ttt = gen.subray(
            float(path_tau[p]), float(path_phi_r[p]), float(path_phi_t[p]),
            float(path_theta_r[p]), float(path_theta_t[p]), float(path_power[p]),
        )
        n = len(ts)
        tau.extend(ts); power.extend(ps)
        phi_r.extend(prr); phi_t.extend(ptt); theta_r.extend(trr); theta_t.extend(ttt)
        src_idx.extend([p] * n)
        los_flag.extend([int(path_los_flags[p])] * n)

    return {
        "tau": np.asarray(tau, np.float32),
        "power": np.asarray(power, np.float32),
        "phi_r_deg": np.asarray(phi_r, np.float32),
        "phi_t_deg": np.asarray(phi_t, np.float32),
        "theta_r_deg": np.asarray(theta_r, np.float32),
        "theta_t_deg": np.asarray(theta_t, np.float32),
        "source_path_idx": np.asarray(src_idx, np.int32),
        "los_nlos_flag": np.asarray(los_flag, np.int32),
    }


def select_top_by_power(rays: dict, max_rays: int) -> dict:
    """ray 수가 max_rays 초과면 전력 상위 max_rays개만 (P1A store_rays 선택 로직)."""
    n = len(rays["tau"])
    if n <= max_rays:
        return rays
    idx = np.argsort(-rays["power"])[:max_rays]
    return {k: v[idx] for k, v in rays.items()}

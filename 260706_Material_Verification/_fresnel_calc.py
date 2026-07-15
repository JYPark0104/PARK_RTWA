"""
반사계수의 주파수 의존성 정량 계산.
핵심 질문: ITU 재질(εr 고정, σ(f) 증가)에서 3.5 vs 7GHz 반사손실이 실제로 얼마나 다른가?
"""
import numpy as np

eps0 = 8.8541878128e-12

# (εr 고정, σ@3.5, σ@7)  ← 앞서 Sionna에서 실측한 값
mats = {
    "concrete": dict(er=5.24, s35=0.12309, s7=0.21168),
    "glass":    dict(er=6.31, s35=0.01928, s7=0.04878),
}

def eps_c(er, sigma, f):
    w = 2*np.pi*f
    return er - 1j*sigma/(w*eps0)   # 복소 상대유전율

def refl_te(eps_c, theta):  # 수직편파(TE) Fresnel
    ct = np.cos(theta); st = np.sin(theta)
    root = np.sqrt(eps_c - st**2)
    return (ct - root)/(ct + root)

def refl_tm(eps_c, theta):  # 수평편파(TM) Fresnel
    ct = np.cos(theta); st = np.sin(theta)
    root = np.sqrt(eps_c - st**2)
    return (eps_c*ct - root)/(eps_c*ct + root)

print(f"{'재질':<10}{'θ(deg)':>7}{'|Γ|@3.5':>10}{'|Γ|@7':>10}{'손실@3.5':>10}{'손실@7':>10}{'Δ(dB)':>9}")
print("-"*66)
for name, m in mats.items():
    e35 = eps_c(m["er"], m["s35"], 3.5e9)
    e7  = eps_c(m["er"], m["s7"],  7.0e9)
    for th_deg in [0, 30, 45, 60, 80]:
        th = np.radians(th_deg)
        # TE/TM 평균 (비편파 근사)
        g35 = 0.5*(abs(refl_te(e35,th))**2 + abs(refl_tm(e35,th))**2)
        g7  = 0.5*(abs(refl_te(e7 ,th))**2 + abs(refl_tm(e7 ,th))**2)
        L35 = -10*np.log10(g35); L7 = -10*np.log10(g7)
        print(f"{name:<10}{th_deg:>7}{np.sqrt(g35):>10.4f}{np.sqrt(g7):>10.4f}"
              f"{L35:>10.3f}{L7:>10.3f}{(L7-L35):>9.3f}")
    print()

print("복소유전율 비교 (εr - jσ/ωε0):")
for name, m in mats.items():
    e35 = eps_c(m["er"], m["s35"], 3.5e9)
    e7  = eps_c(m["er"], m["s7"],  7.0e9)
    print(f"  {name:<10}: 3.5GHz {e35.real:.3f}{e35.imag:+.3f}j | 7GHz {e7.real:.3f}{e7.imag:+.3f}j")

"""
P1E 수렴성 분석 그래프 생성
- 그림 1: Ground Truth 오차 (Covariance Matrix Errors)
- 그림 2: 고유값/고유벡터 오차 (Ground Truth vs Sample)
- 그림 3: 고유벡터 정렬 메트릭 (Ground Truth vs Sample)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 한글 폰트 설정 (선택적)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# CSV 파일 경로
csv_path = Path('/workspaces/Paper2/250929_251001_Ch_Separ/P1E_Validation_Results/P1E_conv_prog_251004_013529_tx1024_rx16.csv')
output_dir = csv_path.parent

# 데이터 로드
df = pd.read_csv(csv_path)
df = df.dropna(how='all')  # 빈 행 제거
print(f"Loaded {len(df)} data points from {csv_path.name}")
print(f"Sample count range: {df['sample_count'].min()} to {df['sample_count'].max()}")

# ====================
# 그림 1: Covariance Matrix Errors
# ====================
fig1, ax1 = plt.subplots(figsize=(12, 8))

ax1.loglog(df['sample_count'], df['R_AE_err_fro_rel'], 'r-o', linewidth=2, markersize=6, 
           label='AE Covariance Error', markevery=1)
ax1.loglog(df['sample_count'], df['R_BS_err_fro_rel'], 'b-s', linewidth=2, markersize=6, 
           label='BS Covariance Error', markevery=1)
ax1.loglog(df['sample_count'], df['R_UE_err_fro_rel'], 'g-^', linewidth=2, markersize=6, 
           label='UE Covariance Error', markevery=1)
ax1.loglog(df['sample_count'], df['R_KM_err_fro_rel'], 'orange', linestyle='--', linewidth=2.5, 
           marker='d', markersize=6, label='Kronecker Approximation Error', markevery=1)

ax1.set_xlabel('Sample Count', fontsize=14, fontweight='bold')
ax1.set_ylabel('Relative Frobenius Norm Error', fontsize=14, fontweight='bold')
ax1.set_title('Covariance Matrix Errors vs Sample Count', fontsize=16, fontweight='bold')
ax1.legend(fontsize=12, loc='best', framealpha=0.9)
ax1.grid(True, which='both', alpha=0.3)
ax1.grid(True, which='minor', alpha=0.15, linestyle=':')

# Y축 범위 자동 조정
y_min = min(df[['R_AE_err_fro_rel', 'R_BS_err_fro_rel', 'R_UE_err_fro_rel', 'R_KM_err_fro_rel']].min())
y_max = max(df[['R_AE_err_fro_rel', 'R_BS_err_fro_rel', 'R_UE_err_fro_rel', 'R_KM_err_fro_rel']].max())
ax1.set_ylim([y_min * 0.5, y_max * 2])

plt.tight_layout()
fig1_path = output_dir / 'P1E_fig1_covariance_errors.png'
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f"Saved: {fig1_path}")
plt.close(fig1)

# ====================
# 그림 2: Eigenvalue/Eigenvector Errors (GT vs Sample)
# ====================
fig2, ax2 = plt.subplots(figsize=(12, 8))

# Ground Truth: O 마커 (파란색), Sample: X 마커 (빨간색)
# Eigenvalue: 실선, Eigenvector: 대시
# 선은 얇게 (1.5), 마커는 크게 (12-14)
ax2.loglog(df['sample_count'], np.sort(df['eps_d'].values)[::-1], 'b-', linewidth=1.5, 
           marker='o', markersize=12, markerfacecolor='none', markeredgewidth=2,
           label='Eigenvalue Error (εd) vs True', markevery=1)
ax2.loglog(df['sample_count'], df['eps_d_sam'], 'r-', linewidth=1.5, 
           marker='+', markersize=14, markeredgewidth=2,
           label='Eigenvalue Error (εd) vs Sample', markevery=1)
ax2.loglog(df['sample_count'], df['eps_U'], 'b--', linewidth=1.5, 
           marker='o', markersize=12, markerfacecolor='none', markeredgewidth=2,
           label='Eigenvector Error (εU) vs True', markevery=1)
ax2.loglog(df['sample_count'], df['eps_U_sam'], 'r--', linewidth=1.5, 
           marker='x', markersize=14, markeredgewidth=2,
           label='Eigenvector Error (εU) vs Sample', markevery=1)

ax2.set_xlabel('Sample Count', fontsize=14, fontweight='bold')
ax2.set_ylabel('Error Value', fontsize=14, fontweight='bold')
ax2.set_title('Eigenvalue and Eigenvector Errors', fontsize=16, fontweight='bold')
# 레전드: 2행 2열, 우하단 위치
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, fontsize=11, loc='lower left', framealpha=0.9, ncol=2)
ax2.grid(True, which='both', alpha=0.3)
ax2.grid(True, which='minor', alpha=0.15, linestyle=':')

# Y축 범위 조정
y_min = min(df[['eps_d', 'eps_d_sam', 'eps_U', 'eps_U_sam']].min())
y_max = max(df[['eps_d', 'eps_d_sam', 'eps_U', 'eps_U_sam']].max())
ax2.set_ylim([max(y_min * 0.5, 1e-4), y_max * 2])

plt.tight_layout()
fig2_path = output_dir / 'P1E_fig2_eigenvalue_eigenvector_errors.png'
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f"Saved: {fig2_path}")
plt.close(fig2)

# ====================
# 그림 3: Inner Product Metrics (GT vs Sample)
# ====================
fig3, ax3 = plt.subplots(figsize=(12, 8))

# Ground Truth: O 마커, Sample: X 마커
# Max: 실선, Min Row-Max: 대시, Min Sum-Squared: 점선
# 선은 얇게 (1.5), 마커는 크게 (12-14)
ax3.loglog(df['sample_count'], df['Inner_MaxAll'], 'b-', linewidth=1.5, 
           marker='o', markersize=12, markerfacecolor='none', markeredgewidth=2,
           label='Inner Product (Max) vs True', markevery=1)
ax3.loglog(df['sample_count'], df['Inner_MaxAll_sam'], 'r-', linewidth=1.5, 
           marker='x', markersize=14, markeredgewidth=2,
           label='Inner Product (Max) vs Sample', markevery=1)
ax3.loglog(df['sample_count'], df['Inner_MinRowMax'], 'b--', linewidth=1.5, 
           marker='o', markersize=12, markerfacecolor='none', markeredgewidth=2,
           label='Inner Product (Min Row-Max) vs True', markevery=1)
ax3.loglog(df['sample_count'], df['Inner_MinRowMax_sam'], 'r--', linewidth=1.5, 
           marker='x', markersize=14, markeredgewidth=2,
           label='Inner Product (Min Row-Max) vs Sample', markevery=1)
ax3.loglog(df['sample_count'], df['Inner_MinSumSq'], 'b:', linewidth=1.5, 
           marker='o', markersize=12, markerfacecolor='none', markeredgewidth=2,
           label='Inner Product (Min Sum-Squared) vs True', markevery=1)
ax3.loglog(df['sample_count'], df['Inner_MinSumSq_sam'], 'r:', linewidth=1.5, 
           marker='x', markersize=14, markeredgewidth=2,
           label='Inner Product (Min Sum-Squared) vs Sample', markevery=1)

ax3.set_xlabel('Sample Count', fontsize=14, fontweight='bold')
ax3.set_ylabel('Inner Product Value', fontsize=14, fontweight='bold')
ax3.set_title('Eigenvector Alignment Metrics', fontsize=16, fontweight='bold')
# 레전드: 2행 3열, 우하단 위치, 위아래로 vs True와 vs Sample 비교
handles, labels = ax3.get_legend_handles_labels()
# 순서 재배치: 위 행(vs True), 아래 행(vs Sample)
# [Max-True, MinRow-True, MinSumSq-True, Max-Sample, MinRow-Sample, MinSumSq-Sample]
order = [0, 1, 2, 3, 4, 5]
ax3.legend([handles[i] for i in order], [labels[i] for i in order], 
           fontsize=10, loc='lower center', framealpha=0.9, ncol=3)
ax3.grid(True, which='both', alpha=0.3)
ax3.grid(True, which='minor', alpha=0.15, linestyle=':')

# Y축 범위 조정 (Inner_MinSumSq는 항상 1.0이므로 제외)
y_min = min(df[['Inner_MaxAll', 'Inner_MaxAll_sam', 'Inner_MinRowMax', 'Inner_MinRowMax_sam']].min())
y_max = max(df[['Inner_MaxAll', 'Inner_MaxAll_sam', 'Inner_MinRowMax', 'Inner_MinRowMax_sam']].max())
ax3.set_ylim([max(y_min * 0.2, 1e-3), y_max * 1.5])

plt.tight_layout()
fig3_path = output_dir / 'P1E_fig3_inner_product_metrics.png'
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
print(f"Saved: {fig3_path}")
plt.close(fig3)

print("\n=== 모든 그래프 생성 완료 ===")
print(f"출력 디렉토리: {output_dir}")
print(f"  - {fig1_path.name}")
print(f"  - {fig2_path.name}")
print(f"  - {fig3_path.name}")


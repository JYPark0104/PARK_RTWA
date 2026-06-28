"""RT Web Agent backend package.

import 시 :func:`backend.jobs.runtime_env.apply_webagent_gpu_env` 로 TF/Mitsuba GPU 정책을
확정한다 (기본: GPU ON + cuda RT). Blackwell(sm_120)만 자동 CPU 폴백.
"""

from backend.jobs.runtime_env import apply_webagent_gpu_env

apply_webagent_gpu_env()

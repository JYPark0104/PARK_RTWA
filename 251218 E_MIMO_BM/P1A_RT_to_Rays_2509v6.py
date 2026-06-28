#!/usr/bin/env python3
"""
RT to Paths & Rays Integrated Pipeline

Enhanced pipeline with Area-based TX/RX positioning system from P1_RT_to_Paths.ipynb.
Combines multiple Jupyter notebooks into a unified pipeline:
- Part 1: BlockRT_savePaths.ipynb (Core RT functionality) 
- Part 2: BlockInterpolator_Paths_to_Rays.ipynb (Ray conversion)
- Part 3: P1_RT_to_Paths.ipynb (Area-based TX/RX positioning)

Key Features (P2 Compatibility Enhancement):
- **MAJOR**: float32 ray data saving for P2 compatibility and 50% memory reduction  
- **MAJOR**: Massive MIMO optimized data types for large antenna arrays
- Area-based TX/RX positioning system (1-4 areas with different grid layouts)
- Grid-based RX placement with customizable coordinates per area
- Flexible RX selection: save all RX data or specific indices only
- Enhanced file naming: Area-based format (e.g., 7.5GHz_Area2_RX1_power.npy)
- Multi-TX support with configurable positions per area

Technical Features:
- Multi-frequency support with modularized architecture (≤30 lines per method)
- Professional naming conventions and cell-level traceability
- Integrated external dependencies with minimal professional comments

Original Sources:
- Part 1: backup files/old codes/BlockRT_savePaths.ipynb
- Part 2: backup files/old codes/BlockInterpolator_Paths_to_Rays.ipynb  
- Part 3: P1A_RT_to_Paths.ipynb (TX/RX positioning reference)
"""

# ========================================================================
# ENVIRONMENT SETUP
# ========================================================================

# Standard library imports
import os
import subprocess
import time
import datetime
import math
import pickle
import json
from datetime import timezone, timedelta

# Scientific computing imports
import numpy as np
import matplotlib.pyplot as plt

# GPU and TensorFlow environment configuration (Part 1 Cell 0)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TENSORBOARD_BINARY"] = "tensorboard"
os.environ["TENSORBOARD_PLUGINS"] = "scalars,images,histograms,graphs,projector,profile"

# TensorFlow import and GPU configuration (Part 1 Cell 0)
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError as e:
        print(e)

tf.get_logger().setLevel("ERROR")

tf.random.set_seed(1) # Random seed initialization (Part 1 Cell 0)
np.random.seed(1) # Random seed initialization (Part 1 Cell 0)

# Mitsuba renderer configuration (Part 1 Cell 0)
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

# Sionna RT framework imports (Part 1 Cell 0, Cell 1)
import sionna
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera, \
                     PathSolver, RadioMapSolver, subcarrier_frequencies
from sionna.phy import PI, SPEED_OF_LIGHT
from sionna.rt.radio_materials import ITURadioMaterial

# Additional utility imports
from pathlib import Path
import geopandas as gpd
import numpy.typing


# Environment validation
print("Sionna RT components imported successfully")
print("Environment setup completed")

# ========================================================================
# CENTRALIZED CONFIGURATION
# ========================================================================

class P1A_Config:
    """Centralized configuration for all pipeline parameters
    
    Organization (High to Low frequency of user changes):
    1. PIPELINE CONTROL & SYSTEM - File paths, system settings, and pipeline control flags
    2. EXPERIMENTAL PARAMETERS - Area configurations and frequency settings for experiments  
    3. ANTENNA & POSITIONING - Setup dependent antenna and device settings
    4. PROCESSING & FILE SETTINGS - RX selection and processing parameters
    5. STANDARD-BASED CONSTANTS - Rarely changed TR 38.901 standard parameters
    
    Pipeline Control Flags:
    - ENABLE_SCENE_PREVIEW: Show 3D scene visualization
    - ENABLE_PATH_SAVING: Save RT path data to disk (required for Part 2)
    - ENABLE_RAY_GENERATION: Run Part 2 ray generation from saved paths  
    - ENABLE_RAY_SAVING: Save generated ray data to disk
    - ENABLE_CHANNEL_BLOCK_SAVING: Save Sionna-compatible channel block format
    - PREVIEW_ONLY_MODE: Only show TX/RX placement (no RT computation)
    
    File Management (Enhanced):
    - UNIFIED_SAVE_DIR: Single directory for all output files
    - File naming: Area{X}_{freq}GHz_{Path|Ray|ChannelBlock}_RX{Y}_{type}.npy
    - Path files: RT simulation results from Part 1
    - Ray files: Statistical ray data from Part 2 
    - ChannelBlock files: Sionna-compatible format (optional)
    
    Multi-area & Multi-frequency Operation:
    - Part 1 processes ALL areas in AREA_INDICES × ALL frequencies in FREQUENCY_CONFIGS
    - Part 2 processes ALL saved path data from all area-frequency combinations
    - Each RX generates separate files for comprehensive coverage analysis
    
    Use Cases:
    - Preview mode: PREVIEW_ONLY_MODE=True (TX/RX placement visualization only)
    - Path-only analysis: ENABLE_RAY_GENERATION=False
    - Memory-only processing: ENABLE_PATH_SAVING=False (Part 2 will fail)
    - Ray generation without saving: ENABLE_RAY_SAVING=False
    - Single area: AREA_INDICES = [2] 
    - Multi area: AREA_INDICES = [1, 2, 3, 4]
    - Single frequency: FREQUENCY_CONFIGS = [7.5]
    - Multi frequency: FREQUENCY_CONFIGS = [3.5, 7.5, 28.0]
    """
    
    # ========== SECTION 1: PIPELINE CONTROL & SYSTEM ==========
    # Most frequently changed settings for different runs
    
    # File Management (Enhanced - Unified folder structure)
    SCENE_FILE_PATHS = ['../Jonggak/Jonggak.xml', 'Jonggak/Jonggak.xml', './Jonggak/Jonggak.xml']
    UNIFIED_SAVE_DIR = "P1A_RT_Results"                  # Single main directory for all data
    # File naming: Area{X}_{freq}GHz_{Path|Ray|ChannelBlock}_RX{Y}_{type}.npy within unified directory
    
    # Data Type Configuration (Enhancement for P2 compatibility)
    RAY_DATA_DTYPE = np.float32 # Ray data precision: float32 (P2 compatible, 50% memory saving)
    FORCE_FLOAT32_SAVING = True  # Force float32 for all ray parameters (Massive MIMO optimized)


    # System Configuration
    RANDOM_SEED = 1        # Change for different random realizations
    GPU_NUM = 0           # GPU device selection
    RT_SEED = 41          # Ray tracing random seed

    # Pipeline Control Flags (most frequently changed for different experiment types)
    ENABLE_SCENE_PREVIEW = True    # Show 3D scene visualization
    ENABLE_PATH_SAVING = True     # Path data kept in memory only (not saved to disk)
    ENABLE_RAY_GENERATION = True   # Run Part 2 ray generation from memory path data
    ENABLE_RAY_SAVING = True       # Save generated ray data to disk
    ENABLE_CHANNEL_BLOCK_SAVING = False  # Save Sionna-compatible channel block format
    ENABLE_METADATA_SAVING = True  # Save comprehensive metadata JSON with timestamp
    
    # Preview Mode Configuration (Enhanced)
    PREVIEW_ONLY_MODE = False     # True: Only show TX/RX placement visualization (no RT computation)
    
    
    # ========== SECTION 2: EXPERIMENTAL PARAMETERS ==========
    # Data I/O and processing configuration (most frequently changed)
    
    # Multi-area & Multi-frequency Configuration (Enhanced)
    AREA_INDICES = [1]     # List of areas to process: [1], [2], [1,2,3,4] for sequential processing
    FREQUENCY_CONFIGS = [7.5]  # List of frequencies in GHz: [3.5, 7.5, 28.0] for multi-frequency
    
    # Data Storage Dimensional Structure (affects file I/O and pipeline compatibility)
    BATCH_SIZE = 1             # Number of batches to process (currently formal dimension for Sionna compatibility)
    N_BS = 1                   # Number of Base Stations (transmitters)
    N_UE = 1                   # Number of User Equipment (receivers)  
    CLUSTER_DIM = 1            # Cluster dimension for channel modeling
    
    # Data Format Configuration
    USE_SIONNA_FORMAT = True   # Use Sionna compatible storage format [B, N_BS, N_UE, 1, N_rays]
    BATCH_PROCESSING = True    # Enable batch-aware processing loops (currently single batch only)
    LEGACY_FORMAT = False      # Fallback to legacy (1, 1, N_rays) format if needed

    # Area-based TX/RX Configuration (Enhanced - from P1_RT_to_Paths.ipynb)
    # Each area defines unique TX positions and RX grid layouts for comprehensive coverage analysis
    
    AREA_CONFIGS = {
        'area_1': {
            'tx_positions': [[-51.561, -21.794, 19]],
            'rx_placement': {
                'method': 'grid',
                'x_params': {'start': -136.138, 'stop': 58.862, 'num': 40},
                'y_params': {'start': -117.667, 'stop': 77.333, 'num': 40},
                'z_params': {'values': [1.5]}
            },
            'description': 'Area 1: Standard coverage zone (Grid-based)'
        },
        'area_2': {
            'tx_positions': [[-51.560643657685944, -21.793815800460543, 19]],
            'rx_placement': {
                'method': 'grid',
                'x_params': {'start': 26, 'stop': 41, 'num': 4},
                'y_params': {'start': -20, 'stop': -10, 'num': 3},
                'z_params': {'values': [1.5]}
            },
            'description': 'Area 2: Dense urban coverage (Grid-based)'
        },
        'area_3': {
            'tx_positions': [[-51.561, -21.794, 19]],
            'rx_placement': {
                'method': 'grid',
                'x_params': {'start': -103.138, 'stop': -53.138, 'num': 20},
                'y_params': {'start': -77.667, 'stop': -27.667, 'num': 20},
                'z_params': {'values': [1.5]}
            },
            'description': 'Area 1: Standard coverage zone (Grid-based)'
        },
        'area_4': {
            'tx_positions': [[-51.560643657685944, -21.793815800460543, 19]],
            'rx_placement': {
                'method': 'grid',
                'x_params': {'start': -125, 'stop': 75, 'num': 20},
                'y_params': {'start': -125, 'stop': 75, 'num': 20},
                'z_params': {'values': [1.5]}
            },
            'description': 'Area 4: Southern coverage zone (Grid-based)'
        },
        'area_5': {
            'tx_positions': [[-51.560643657685944, -21.793815800460543, 19]],
            'rx_placement': {
                'method': 'grid',
                'x_params': {'start': -125, 'stop': 75, 'num': 40},
                'y_params': {'start': -125, 'stop': 75, 'num': 40},
                'z_params': {'values': [1.5]}
            },
            'description': 'Area 4: Southern coverage zone (Grid-based)'
        },
        # --- New Area Examples for Systematic Placement ---
        'area_6_radial': {
            'tx_positions': [[-51.560643657685944, -21.793815800460543, 19]],
            'rx_placement': {
                'method': 'radial',
                'center': [-18, -21],  # Center around TX
                'radii_m': [50, 100, 150],
                'angles_deg': {'start': 0, 'stop': 360, 'num': 12}, # 30-degree steps
                'z_params': {'values': [1.5, 10, 20]} # Ground, mid, and high floors
            },
            'description': 'Area 5: Radial coverage analysis at different heights'
        },
        'area_7_street': {
            'tx_positions': [[-51.560643657685944, -21.793815800460543, 19]],
            'rx_placement': {
                'method': 'street',
                'path_points': [[10, -50], [10, -150]], # (x,y) coordinates
                'num_points': 20,
                'z_params': {'values': [1.5]}
            },
            'description': 'Area 6: Street-level scenario analysis'
        }
    }
        
    # Analysis and Processing Parameters  
    NUM_INTERESTING_PATHS = 20  # Maximum paths to analyze per TX-RX pair
    MAX_DEPTH = 5              # Maximum reflection/diffraction depth
    MAX_RAYS_PER_PAIR = 400   # Maximum rays to generate per TX-RX pair
    
    # ========== SECTION 3: ANTENNA & POSITIONING ==========
    # Setup dependent antenna and device settings
    
    # Antenna Array Configuration
    ANTENNA_NUM_ROWS = 1
    ANTENNA_NUM_COLS = 1
    ANTENNA_VERTICAL_SPACING = 0.5
    ANTENNA_HORIZONTAL_SPACING = 0.5
    ANTENNA_PATTERN = "iso"         # Isotropic antenna pattern
    ANTENNA_POLARIZATION = "V"      # Vertical polarization
    
    # Device Positioning and Visualization
    TX_ORIENTATION = [0, 0, 0]           # TX orientation [roll, pitch, yaw] in degrees
    TX_DISPLAY_RADIUS = 10               # TX visualization radius for scene preview
    RX_ORIENTATION = [0, 0, 0]           # RX orientation [roll, pitch, yaw] in degrees
    RX_DISPLAY_RADIUS = 3                # RX visualization radius for scene preview
    
    # ========== SECTION 4: PROCESSING & FILE SETTINGS ==========
    # Moderate changes for different processing requirements
    
    # RX Selection Configuration (Enhanced)
    SAVE_ALL_RX = True                # True: save all RX data, False: save only specific indices
    SELECTED_RX_INDICES = [1, 2, 3]  # Only used when SAVE_ALL_RX=False (selective saving)
        
    # ========== SECTION 5: STANDARD-BASED CONSTANTS ==========
    # Rarely changed TR 38.901 standard parameters for channel modeling
    
    # ITU Material Properties (ITU-R P.2040-1)
    ITU_TARGET_MATERIALS = ("ceiling_board", "concrete", "glass")
    ITU_SCATTERING_COEFF = 0.2
    ITU_XPD_COEFF = 0.5
    
    # TR 38.901 Channel Model Parameters (3GPP TR 38.901 v16.1.0 Table 7.5-6 UMa scenario)
    TR38901_C_ZSA = 7     # Cluster ZSA (Zenith Spread of Arrival) [deg]
    TR38901_MU_LGZSD = 0.5  # ZSD log-normal distribution mean parameter
    
    # LOS Scenario Constants (TR 38.901 Table 7.5-6 UMa LOS)
    LOS_C_ASA = 11        # Cluster ASA (Azimuth Spread of Arrival) [deg]
    LOS_C_ASD = 5         # Cluster ASD (Azimuth Spread of Departure) [deg] 
    LOS_R_TAU = 2.5       # Delay scaling parameter r_τ
    LOS_N_CLUSTERS = 12   # Number of clusters N
    
    # NLOS Scenario Constants (TR 38.901 Table 7.5-6 UMa NLOS)
    NLOS_C_ASA = 15       # Cluster ASA (Azimuth Spread of Arrival) [deg]
    NLOS_C_ASD = 2        # Cluster ASD (Azimuth Spread of Departure) [deg]
    NLOS_R_TAU = 2.3      # Delay scaling parameter r_τ 
    NLOS_N_CLUSTERS = 20  # Number of clusters N
    
    # Sub-ray Distribution Constants (TR 38.901 Table 7.5-3 and 7.5-5)
    LOS_COMB_N = 35       # Binomial distribution parameter for LOS sub-ray generation
    NLOS_COMB_N = 25      # Binomial distribution parameter for NLOS sub-ray generation
    
    # Sub-ray Probability Constants (Implementation-specific)
    SUBRAY_PROB_P1 = 0.8      # Primary probability parameter
    SUBRAY_PROB_P2 = 0.2      # Secondary probability parameter
    SUBRAY_PROB_R_BASE = 0.3  # Base probability for reflection interactions
    SUBRAY_PROB_R_MULT = 0.7  # Multiplier for reflection probability
    SUBRAY_PROB_D_BASE = 0.4  # Base probability for diffraction interactions
    SUBRAY_PROB_D_MULT = 0.6  # Multiplier for diffraction probability
    
    # Interaction Scaling Constants (Implementation-specific)
    INTERACTION_SCALE_DS = 1.0   # Full scaling for diffuse scattering contribution
    INTERACTION_SCALE_OTHER = 0.1  # Reduced scaling for other interaction types

# Create global config instance
p1a_config = P1A_Config()

# Override tf.random seed with config value if different
if p1a_config.RANDOM_SEED != 1:
    tf.random.set_seed(p1a_config.RANDOM_SEED)

# ========================================================================
# CLASS DECLARATIONS
# ========================================================================

class Utils:
    """기본 유틸리티 클래스 (Part 1 Cell 7, Part 2 Cell 2)"""
    
    @staticmethod
    def rad_to_deg(angle_rad):
        """Convert angle from radians to degrees (Part 1 Cell 7, Part 2 Cell 2)
        
        Args:
            angle_rad: Angle values in radians
        Returns:
            Angle values in degrees
        """
        return angle_rad * 180.0 / np.pi
    
    @staticmethod
    def deg_to_rad(angle_deg):
        """Convert angle from degrees to radians (Part 2 Cell 2)
        
        Args:
            angle_deg: Angle values in degrees
        Returns:
            Angle values in radians
        """
        return angle_deg * np.pi / 180.0
    
    @staticmethod
    def get_area_config(area_index):
        """Safely get area configuration with validation"""
        area_key = f'area_{area_index}'
        if area_key not in p1a_config.AREA_CONFIGS:
            raise ValueError(f"Area {area_index} not found in AREA_CONFIGS. Available areas: {list(p1a_config.AREA_CONFIGS.keys())}")
        return p1a_config.AREA_CONFIGS[area_key]
    
    @staticmethod
    def get_rx_count_from_config(area_config):
        """Calculate the number of RX positions from area configuration without generating them."""
        if 'rx_placement' not in area_config:
            # Fallback for legacy format
            return len(area_config.get('rx_x_coords', [])) * len(area_config.get('rx_y_coords', [])) * len(area_config.get('rx_z_coords', []))

        placement_config = area_config['rx_placement']
        method = placement_config['method']
        
        if method == 'grid':
            return (placement_config['x_params']['num'] * 
                    placement_config['y_params']['num'] * 
                    len(placement_config['z_params']['values']))
        elif method == 'explicit':
            return (len(placement_config['x_coords']) * 
                    len(placement_config['y_coords']) * 
                    len(placement_config['z_coords']))
        elif method == 'radial':
            return (len(placement_config['radii_m']) * 
                    placement_config['angles_deg']['num'] * 
                    len(placement_config['z_params']['values']))
        elif method == 'street':
            return (placement_config['num_points'] * 
                    len(placement_config['z_params']['values']))
        
        return 0
    
    @staticmethod
    def print_vram():
        """Print VRAM usage for monitoring (Utility function)"""
        try:
            gpu_devices = tf.config.list_physical_devices('GPU')
            if gpu_devices:
                memory_info = tf.config.experimental.get_memory_info('GPU:0')
                print(f"GPU Memory - Current: {memory_info['current']/(1024**3):.2f}GB, Peak: {memory_info['peak']/(1024**3):.2f}GB")
        except Exception as e:
            print(f"GPU memory check failed: {e}")
    
    @staticmethod
    def to_numpy(arrays_dict):
        """Convert TensorFlow tensors to numpy arrays (Part 1 Cell 7)"""
        for key in arrays_dict:
            if hasattr(arrays_dict[key], 'numpy'):
                arrays_dict[key] = arrays_dict[key].numpy()
        return arrays_dict
    
    @staticmethod
    def get_kst_timestamp():
        """Generate KST-based timestamp in YYMMDD_HHMMSS format"""
        # KST = UTC+9
        kst = timezone(timedelta(hours=9))
        now_kst = datetime.datetime.now(kst)
        return now_kst.strftime("%y%m%d_%H%M%S")
    
    @staticmethod
    def convert_for_json(obj):
        """Convert NumPy and TensorFlow types to JSON-serializable Python types"""
        if hasattr(obj, 'numpy'):  # TensorFlow tensor
            obj = obj.numpy()
        
        if hasattr(obj, 'item'):  # NumPy scalar
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: Utils.convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [Utils.convert_for_json(item) for item in obj]
        else:
            return obj
    
    @staticmethod
    def compute_ray_statistics(ray_data, param_name):
        """Compute comprehensive statistics for ray parameters (non-zero values only)"""
        if ray_data is None or len(ray_data) == 0:
            return None
        
        # Convert to numpy if needed
        if hasattr(ray_data, 'numpy'):
            ray_data = ray_data.numpy()
        
        # Flatten if multi-dimensional
        if ray_data.ndim > 1:
            ray_data = ray_data.flatten()
        
        # Filter non-zero values
        non_zero_data = ray_data[ray_data != 0]
        total_count = len(ray_data)
        non_zero_count = len(non_zero_data)
        
        if non_zero_count == 0:
            return {
                "total_count": total_count,
                "non_zero_count": 0,
                "zero_ratio": 1.0,
                "statistics": "all_zero"
            }
        
        # Compute comprehensive statistics
        stats = {
            "total_count": total_count,
            "non_zero_count": non_zero_count,
            "zero_ratio": (total_count - non_zero_count) / total_count,
            "mean": float(np.mean(non_zero_data)),
            "std": float(np.std(non_zero_data)),
            "min": float(np.min(non_zero_data)),
            "max": float(np.max(non_zero_data)),
            "median": float(np.median(non_zero_data)),
            "percentile_90": float(np.percentile(non_zero_data, 90)),
            "rms": float(np.sqrt(np.mean(non_zero_data**2)))
        }
        
        # Add parameter-specific analysis
        if param_name in ['power']:
            # Power-specific: dynamic range in dB
            if stats["min"] > 0:
                stats["dynamic_range_db"] = 10 * np.log10(stats["max"] / stats["min"])
            else:
                stats["dynamic_range_db"] = "infinite"
        
        elif param_name in ['tau']:
            # Delay-specific: delay spread
            stats["delay_spread_ns"] = (stats["max"] - stats["min"]) * 1e9
            stats["mean_ns"] = stats["mean"] * 1e9
            stats["std_ns"] = stats["std"] * 1e9
        
        elif param_name in ['theta_r_deg', 'theta_t_deg', 'phi_r_deg', 'phi_t_deg']:
            # Angle-specific: angular spread (data stored in degrees)
            stats["angular_spread_deg"] = stats["std"]
            stats["coverage_range_deg"] = stats["max"] - stats["min"]
        
        return stats
    
    @staticmethod
    def create_metadata(path_data_summary, ray_data_summary, run_config):
        """Create comprehensive metadata dictionary for JSON storage"""
        timestamp = Utils.get_kst_timestamp()
        
        metadata = {
            "session_info": {
                "timestamp_kst": timestamp,
                "pipeline_version": "P1A_RT_to_Paths_Rays",
                "generation_mode": "memory_to_ray_only"
            },
            "configuration": {
                "areas_processed": run_config["areas"],
                "frequencies_ghz": run_config["frequencies"],
                "total_combinations": len(run_config["areas"]) * len(run_config["frequencies"]),
                "enable_path_saving": p1a_config.ENABLE_PATH_SAVING,
                "enable_ray_saving": p1a_config.ENABLE_RAY_SAVING,
                "enable_channel_block_saving": p1a_config.ENABLE_CHANNEL_BLOCK_SAVING
            },
            "path_data_summary": path_data_summary,
            "ray_data_summary": ray_data_summary,
            "processing_parameters": {
                "max_interesting_paths": p1a_config.NUM_INTERESTING_PATHS,
                "max_depth": p1a_config.MAX_DEPTH,
                "max_rays_per_pair": p1a_config.MAX_RAYS_PER_PAIR,
                "random_seed": p1a_config.RANDOM_SEED,
                "rt_seed": p1a_config.RT_SEED
            },
            "area_configurations": {},
            "ray_statistics_summary": {}
        }
        
        # Add detailed area configurations
        for area_idx in run_config["areas"]:
            area_config = Utils.get_area_config(area_idx)
            rx_count = Utils.get_rx_count_from_config(area_config)
            
            metadata["area_configurations"][f"area_{area_idx}"] = {
                "description": area_config['description'],
                "tx_count": len(area_config['tx_positions']),
                "rx_count": rx_count,
                "tx_positions": area_config['tx_positions'],
                "rx_placement_config": area_config.get('rx_placement', 'Legacy format not supported in metadata')
            }
        
        # Convert all data to JSON-serializable types
        metadata = Utils.convert_for_json(metadata)
        
        return metadata, timestamp

class Scene:
    """씬 설정 및 구성 클래스 (Part 1 Cell 1-5)"""
    
    @staticmethod
    def find_file():
        """Find available scene file from configured paths (Part 1 Cell 2)"""
        for path in p1a_config.SCENE_FILE_PATHS:
            if os.path.exists(path):
                return path
        
        print(f"Current working directory: {os.getcwd()}")
        print("Available directories:", [d for d in os.listdir('.') if os.path.isdir(d)])
        raise FileNotFoundError(f"Scene file not found. Tried paths: {p1a_config.SCENE_FILE_PATHS}")
    
    @staticmethod
    def load():
        """Load and configure base scene parameters (Part 1 Cell 1)"""
        scene_path = Scene.find_file()
        print(f"Loading scene from: {scene_path}")
        
        # Cell 1: Original notebook loads scene directly after Sionna imports
        # ITU radio material plugin is automatically registered by Sionna RT
        scene = load_scene(scene_path, merge_shapes=False)
        
        print(f"Scene loaded successfully: {len(scene.objects)} objects")
        return scene
    
    @staticmethod
    def setup_antennas(scene):
        """Configure TX and RX antenna arrays (Part 1 Cell 3-4)"""
        antenna_pattern = PlanarArray(
            num_rows=p1a_config.ANTENNA_NUM_ROWS, 
            num_cols=p1a_config.ANTENNA_NUM_COLS, 
            vertical_spacing=p1a_config.ANTENNA_VERTICAL_SPACING,
            horizontal_spacing=p1a_config.ANTENNA_HORIZONTAL_SPACING, 
            pattern=p1a_config.ANTENNA_PATTERN, 
            polarization=p1a_config.ANTENNA_POLARIZATION
        )
        
        scene.tx_array = antenna_pattern
        scene.rx_array = antenna_pattern
    
    @staticmethod
    def tune_materials(scene, target=None, s_coeff=None, xpd_coeff=None):
        """Tune ITU material coefficients for all matching objects (Part 1 Cell 3)"""
        if target is None:
            target = p1a_config.ITU_TARGET_MATERIALS
        if s_coeff is None:
            s_coeff = p1a_config.ITU_SCATTERING_COEFF
        if xpd_coeff is None:
            xpd_coeff = p1a_config.ITU_XPD_COEFF
        
        n = 0
        for name in scene.objects:
            obj = scene.get(name)
            rm = getattr(obj, "radio_material", None)

            if isinstance(rm, ITURadioMaterial) and rm.itu_type in target:
                rm.scattering_coefficient = s_coeff
                rm.xpd_coefficient = xpd_coeff
                n += 1

        print(f"수정된 shape 수: {n}")
        return n
    
    @staticmethod
    def clear_devices(scene):
        """Remove all existing transmitters and receivers from scene (Part 1 Cell 5)"""
        [scene.remove(tx) for tx in scene.transmitters]
        [scene.remove(rx) for rx in scene.receivers]
        return scene
    
    @staticmethod
    def add_transmitters(scene, area_index):
        """Deploy transmitters based on selected area configuration (Part 1 Cell 5 + enhancement)"""
        area_config = Utils.get_area_config(area_index)
        tx_positions = area_config['tx_positions']
        
        tx_index = 0
        for tx_position in tx_positions:
            tx_index += 1
            scene.add(Transmitter(f"tx{tx_index}", tx_position, 
                                orientation=p1a_config.TX_ORIENTATION, 
                                display_radius=p1a_config.TX_DISPLAY_RADIUS))
    
    @staticmethod
    def add_receivers(scene, area_index):
        """Deploy receivers based on area grid configuration (Part 1 Cell 5 + Area-based enhancement)"""
        rx_positions = Scene.generate_area_rx_grid(area_index)
        
        rx_index = 0
        for rx_position in rx_positions:
            rx_index += 1
            scene.add(Receiver(f"rx{rx_index}", rx_position, 
                             orientation=p1a_config.RX_ORIENTATION, 
                             display_radius=p1a_config.RX_DISPLAY_RADIUS))
    
    @staticmethod
    def preview(scene):
        """Generate scene preview if ENABLE_SCENE_PREVIEW is True (Part 1 Cell 5)"""
        if p1a_config.ENABLE_SCENE_PREVIEW:
            print("Generating scene preview...")
            try:
                scene.preview()
            except Exception as e:
                print(f"Scene preview failed (this is normal in headless environments): {e}")
        else:
            print("Scene preview disabled (set ENABLE_SCENE_PREVIEW=True to enable)")
    
    @staticmethod
    def configure_arrays_and_materials(scene):
        """Configure TX and RX antenna arrays with ITU material tuning (Part 1 Cell 3-4)"""
        Scene.tune_materials(scene)
        Scene.setup_antennas(scene)
    
    @staticmethod
    def generate_area_rx_grid(area_index):
        """Generate RX grid positions for selected area using systematic placement methods"""
        area_config = Utils.get_area_config(area_index)
        placement_config = area_config.get('rx_placement')
        
        if not placement_config:
            # Fallback to legacy format if 'rx_placement' is not defined
            rx_x_coords = area_config['rx_x_coords']
            rx_y_coords = area_config['rx_y_coords']
            rx_z_coords = area_config['rx_z_coords']
            method_info = "legacy explicit coords"
        else:
            method = placement_config['method']
            method_info = f"'{method}' method"
            
            if method == 'grid':
                x_p = placement_config['x_params']
                rx_x_coords = np.linspace(x_p['start'], x_p['stop'], x_p['num'])
                y_p = placement_config['y_params']
                rx_y_coords = np.linspace(y_p['start'], y_p['stop'], y_p['num'])
                rx_z_coords = placement_config['z_params']['values']
            
            elif method == 'explicit':
                rx_x_coords = placement_config['x_coords']
                rx_y_coords = placement_config['y_coords']
                rx_z_coords = placement_config['z_coords']
                
            elif method == 'radial':
                center = placement_config['center']
                radii = placement_config['radii_m']
                angles_p = placement_config['angles_deg']
                angles_rad = np.linspace(np.deg2rad(angles_p['start']), np.deg2rad(angles_p['stop']), angles_p['num'])
                z_coords = placement_config['z_params']['values']
                
                rx_positions = [
                    [float(center[0] + r * np.cos(angle)), float(center[1] + r * np.sin(angle)), float(z)]
                    for z in z_coords
                    for r in radii
                    for angle in angles_rad
                ]
                print(f"Area {area_index}: Generated {len(rx_positions)} RX positions using {method_info}.")
                return rx_positions
                
            elif method == 'street':
                points = placement_config['path_points']
                num_points = placement_config['num_points']
                z_coords = placement_config['z_params']['values']
                start_point = np.array(points[0])
                end_point = np.array(points[1])
                
                rx_positions = [
                    [float(p[0]), float(p[1]), float(z)]
                    for z in z_coords
                    for p in (start_point + i * (end_point - start_point) for i in np.linspace(0, 1, num_points))
                ]
                print(f"Area {area_index}: Generated {len(rx_positions)} RX positions using {method_info}.")
                return rx_positions
                
            else:
                raise ValueError(f"Unknown RX placement method: '{method}' for Area {area_index}")

        # Common grid generation for 'grid' and 'explicit' methods
        rx_positions = [
            [float(x), float(y), float(z)] 
            for z in rx_z_coords 
            for y in rx_y_coords 
            for x in rx_x_coords
        ]
        
        print(f"Area {area_index}: Generated {len(rx_positions)} RX positions using {method_info}")
        if method in ['grid', 'explicit']:
             print(f"Grid: {len(rx_x_coords)}×{len(rx_y_coords)}×{len(rx_z_coords)} = {len(rx_positions)} positions")
        
        return rx_positions
    

class PathRT:
    """Ray Tracing 경로 생성 클래스 (Part 1 Cell 6-7)"""
    
    @staticmethod
    def setup_rf(frequency_ghz):
        """Set RF frequency for specific frequency (Part 1 Cell 2, simplified)"""
        carrier_frequency = frequency_ghz * 1e9  # Hz
        return carrier_frequency
    
    @staticmethod
    def trace_rays(scene, carrier_frequency, frequency_ghz):
        """Perform ray tracing path computation for specific frequency (Part 1 Cell 6)"""
        scene.frequency = carrier_frequency
        wavelength = tf.cast(SPEED_OF_LIGHT/scene.frequency, tf.float32)
        print(f"Ray tracing for frequency: {frequency_ghz} GHz")
        
        print("Solving ray tracing paths...")
        tic = time.time()
        Utils.print_vram()
        
        p_solver = PathSolver()
        paths = p_solver(scene=scene, max_depth=p1a_config.MAX_DEPTH, los=True, specular_reflection=True, 
                        diffuse_reflection=True, refraction=True, synthetic_array=False, seed=p1a_config.RT_SEED)
        
        toc = time.time()
        print(f"Path solving completed in {round(toc-tic, 3)} seconds")
        return paths, wavelength
    
    @staticmethod
    def extract_cir(paths):
        """Extract CIR coefficients and delays from ray tracing paths (Part 1 Cell 7)"""
        a_, tau = paths.cir(normalize_delays=True, out_type="numpy")
        return a_, tau
    
    @staticmethod
    def reshape_cir(a_, tau):
        """Reshape CIR tensors from antenna array format to TX-RX-path format (Part 1 Cell 7)"""
        # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] -> [num_tx, num_rx, num_paths]
        a_ = tf.squeeze(a_, axis=[1,3,5])
        a_ = tf.transpose(a_, perm=[1,0,2])
        tau = tf.squeeze(tau, axis=[1,3])
        tau = tf.transpose(tau, perm=[1,0,2])
        return a_, tau
    
    @staticmethod
    def reshape_angles(angle_tensor):
        """Reshape single angle tensor to TX-RX-path format (Part 1 Cell 7)"""
        angle_tensor = tf.squeeze(angle_tensor, axis=[1,3])
        angle_tensor = tf.transpose(angle_tensor, perm=[1,0,2])
        return angle_tensor
    
    @staticmethod
    def process_interactions(interactions, vertices):
        """Process path interaction types and vertex data (Part 1 Cell 7)"""
        pathTypes = tf.squeeze(interactions, axis=[2,4])
        pathTypes = tf.transpose(pathTypes, perm=[2,1,3,0])
        
        vertices_proc = tf.squeeze(vertices, axis=[2,4])
        vertices_proc = tf.transpose(vertices_proc, perm=[2,1,3,0,4])
        
        return pathTypes, vertices_proc
    
    @staticmethod
    def compute_angles(a_, tau, paths):
        """Compute path power and angle parameters (Part 1 Cell 7)
        
        Sionna RT paths object provides angles in radians.
        This function converts them to degrees for storage and processing.
        
        Returns:
            a_: Complex amplitude values
            theta_r_deg: Receiver zenith angles in degrees
            phi_r_deg: Receiver azimuth angles in degrees  
            theta_t_deg: Transmitter zenith angles in degrees
            phi_t_deg: Transmitter azimuth angles in degrees
        """
        # Cell 7: Extract and reshape angle tensors to match CIR format (radians from Sionna)
        theta_r_rad = PathRT.reshape_angles(paths.theta_r)
        phi_r_rad = PathRT.reshape_angles(paths.phi_r)
        theta_t_rad = PathRT.reshape_angles(paths.theta_t)
        phi_t_rad = PathRT.reshape_angles(paths.phi_t)
        
        # Cell 7: Convert from radians to degrees for storage
        theta_r_deg = Utils.rad_to_deg(theta_r_rad)
        phi_r_deg = Utils.rad_to_deg(phi_r_rad)
        theta_t_deg = Utils.rad_to_deg(theta_t_rad)
        phi_t_deg = Utils.rad_to_deg(phi_t_rad)
        
        # Cell 7: Original notebook works with amplitude (a_), not power
        return a_, theta_r_deg, phi_r_deg, theta_t_deg, phi_t_deg
    
    @staticmethod
    def sort_by_power(a_, tau, theta_r_deg, phi_r_deg, theta_t_deg, phi_t_deg, path_types, vertices):
        """Sort all path parameters by descending amplitude (Part 1 Cell 7)
        
        Args:
            theta_r_deg, phi_r_deg, theta_t_deg, phi_t_deg: Angle arrays in degrees
        """
        # Cell 7: Original notebook sorts by tf.abs(a_), axis=2, direction='DESCENDING'
        sorted_indices = tf.argsort(tf.abs(a_), axis=2, direction='DESCENDING')
        
        # Cell 7: Apply sorting using axis=2, batch_dims=2 (original notebook style)
        a_sorted = tf.gather(a_, sorted_indices, axis=2, batch_dims=2)
        tau_sorted = tf.gather(tau, sorted_indices, axis=2, batch_dims=2)
        theta_r_deg_sorted = tf.gather(theta_r_deg, sorted_indices, axis=2, batch_dims=2)
        phi_r_deg_sorted = tf.gather(phi_r_deg, sorted_indices, axis=2, batch_dims=2)
        theta_t_deg_sorted = tf.gather(theta_t_deg, sorted_indices, axis=2, batch_dims=2)
        phi_t_deg_sorted = tf.gather(phi_t_deg, sorted_indices, axis=2, batch_dims=2)
        
        # Cell 7: Sort path types and vertices if available
        if path_types is not None:
            path_types_sorted = tf.gather(path_types, sorted_indices, axis=2, batch_dims=2)
        else:
            path_types_sorted = None
            
        if vertices is not None:
            vertices_sorted = tf.gather(vertices, sorted_indices, axis=2, batch_dims=2)
        else:
            vertices_sorted = None
        
        return (a_sorted, tau_sorted, theta_r_deg_sorted, phi_r_deg_sorted, 
                theta_t_deg_sorted, phi_t_deg_sorted, path_types_sorted, vertices_sorted)
    
    @staticmethod
    def select_top(sorted_data, num_paths):
        """Select strongest N paths from sorted data (Part 1 Cell 7)
        
        Returns angle data in degrees as stored.
        """
        (a_sorted, tau_sorted, theta_r_deg_sorted, phi_r_deg_sorted, 
         theta_t_deg_sorted, phi_t_deg_sorted, path_types_sorted, vertices_sorted) = sorted_data
        
        # Cell 7: Select top N paths (original notebook: 0:num_InterestingPath)
        a_sorted_selected = a_sorted[:, :, 0:num_paths]
        power_sorted_selected = tf.abs(a_sorted_selected)**2  # Cell 7: Calculate power from amplitude
        
        selected_data = (
            power_sorted_selected,  # Return power for saving
            tau_sorted[:, :, 0:num_paths],
            theta_r_deg_sorted[:, :, 0:num_paths],  # Degrees
            phi_r_deg_sorted[:, :, 0:num_paths],    # Degrees
            theta_t_deg_sorted[:, :, 0:num_paths],  # Degrees
            phi_t_deg_sorted[:, :, 0:num_paths],    # Degrees
            path_types_sorted[:, :, 0:num_paths, :] if path_types_sorted is not None else None,
            vertices_sorted[:, :, 0:num_paths, :, :] if vertices_sorted is not None else None
        )
        
        return selected_data

class DataIO:
    """데이터 저장 및 로딩 클래스 (Part 1 Cell 7, Part 2 Cell 2)"""
    
    @staticmethod
    def save_paths_unified(arrays_dict, carrier_frequency, area_index, rx_index):
        """Save path data arrays with unified folder and descriptive filenames (Part 1 Cell 7 + enhancement)"""
        directory = p1a_config.UNIFIED_SAVE_DIR
        os.makedirs(directory, exist_ok=True)
        fc_ghz = carrier_frequency / 1e9
        
        for key, array in arrays_dict.items():
            filename = f"Area{area_index}_{fc_ghz}GHz_Path_RX{rx_index}_{key}.npy"
            filepath = os.path.join(directory, filename)
            np.save(filepath, array)
            print(f"Area{area_index} RX{rx_index} {key} saved: {filepath} (shape: {array.shape})")
        
        return True
    
    @staticmethod
    def save_all_rx_paths_unified(selected_data, carrier_frequency, area_index, scene):
        """Save path data for all RX or selected RX indices with unified folder structure (Part 1 Cell 7 + enhancement)"""
        if p1a_config.ENABLE_PATH_SAVING:
            (power_selected, tau_selected, theta_r_deg_selected, phi_r_deg_selected, 
             theta_t_deg_selected, phi_t_deg_selected, path_types_selected, vertices_selected) = selected_data
            
            # Determine which RX indices to save
            num_rx = len(scene.receivers)
            if p1a_config.SAVE_ALL_RX:
                rx_indices_to_save = list(range(1, num_rx + 1))
                print(f"Area {area_index}: saving {num_rx} RX data files")
            else:
                rx_indices_to_save = p1a_config.SELECTED_RX_INDICES
                print(f"Area {area_index}: saving selected {len(rx_indices_to_save)} RX data files: {rx_indices_to_save}")
            
            # Save data for each RX
            for rx_idx in rx_indices_to_save:
                if rx_idx <= num_rx:
                    # Extract data for this RX (adjust for 0-based indexing)
                    rx_array_idx = rx_idx - 1
                    arrays = {
                        'power': power_selected[:, rx_array_idx:rx_array_idx+1, :],
                        'tau': tau_selected[:, rx_array_idx:rx_array_idx+1, :],
                        'theta_r_deg': theta_r_deg_selected[:, rx_array_idx:rx_array_idx+1, :],
                        'phi_r_deg': phi_r_deg_selected[:, rx_array_idx:rx_array_idx+1, :],
                        'theta_t_deg': theta_t_deg_selected[:, rx_array_idx:rx_array_idx+1, :],
                        'phi_t_deg': phi_t_deg_selected[:, rx_array_idx:rx_array_idx+1, :],
                        'interactions': path_types_selected[:, rx_array_idx:rx_array_idx+1, :, :],
                        'vertices': vertices_selected[:, rx_array_idx:rx_array_idx+1, :, :, :]
                    }
                    
                    # Convert and save
                    arrays = Utils.to_numpy(arrays)
                    DataIO.save_paths_unified(arrays, carrier_frequency, area_index, rx_idx)
            
            print(f"Area {area_index}: {len(rx_indices_to_save)} RX data save complete")
        else:
            print("Path data saving disabled (set ENABLE_PATH_SAVING=True to enable)")
    
    @staticmethod
    def load_paths_unified(area_index, freq_ghz, rx_index):
        """Load saved path data arrays with unified folder structure (Part 2 Cell 2 + enhancement)"""
        try:
            base_dir = p1a_config.UNIFIED_SAVE_DIR
            base_filename = f"Area{area_index}_{freq_ghz}GHz_Path_RX{rx_index}"
            
            path_data = {
                'power': np.load(f"{base_dir}/{base_filename}_power.npy"),
                'tau': np.load(f"{base_dir}/{base_filename}_tau.npy"),
                'theta_r_deg': np.load(f"{base_dir}/{base_filename}_theta_r.npy"),  # Legacy file, degrees
                'theta_t_deg': np.load(f"{base_dir}/{base_filename}_theta_t.npy"),  # Legacy file, degrees
                'phi_r_deg': np.load(f"{base_dir}/{base_filename}_phi_r.npy"),      # Legacy file, degrees
                'phi_t_deg': np.load(f"{base_dir}/{base_filename}_phi_t.npy"),      # Legacy file, degrees
                'interactions': np.load(f"{base_dir}/{base_filename}_interactions.npy")
            }
            return path_data
        except FileNotFoundError:
            print(f"Skipping Area{area_index}_{freq_ghz}GHz_Path_RX{rx_index} - files not found")
            return None
    
    @staticmethod
    def save_rays_unified(ray_arrays, area_index, freq_ghz, rx_index):
        """Save ray data to unified folder with optimized naming (Part 2 Cell 2 + enhancement)"""
        if p1a_config.ENABLE_RAY_SAVING:
            save_dir = p1a_config.UNIFIED_SAVE_DIR
            os.makedirs(save_dir, exist_ok=True)
            
            # Save enhanced ray arrays with Path-Ray relationship and LoS/NLoS information
            ray_keys = ['tau', 'power', 'theta_r_deg', 'theta_t_deg', 'phi_r_deg', 'phi_t_deg', 
                       'source_path_idx', 'los_nlos_flag', 'counts']
            
            for key in ray_keys:
                filename = f"Area{area_index}_{freq_ghz}GHz_Ray_RX{rx_index}_{key}.npy"
                filepath = os.path.join(save_dir, filename)
                
                # Data type optimization for different fields
                if p1a_config.FORCE_FLOAT32_SAVING and key in ['tau', 'power', 'theta_r_deg', 'theta_t_deg', 'phi_r_deg', 'phi_t_deg']:
                    # Ray parameters: convert to float32 for P2 compatibility and memory efficiency
                    np.save(filepath, ray_arrays[key].astype(p1a_config.RAY_DATA_DTYPE))
                    print(f"Ray {key} saved (float32): {filepath}")
                else:
                    # Integer fields (counts, source_path_idx, los_nlos_flag): keep int32 dtype
                    np.save(filepath, ray_arrays[key])
                    print(f"Ray {key} saved (int32): {filepath}")
            
            # Convert to TensorFlow and save channel block format with optimized naming (optional)
            if p1a_config.ENABLE_CHANNEL_BLOCK_SAVING:
                DataIO.save_channel_blocks_unified(ray_arrays, area_index, freq_ghz, rx_index)
            else:
                print("Channel block saving disabled (set ENABLE_CHANNEL_BLOCK_SAVING=True to enable)")
        else:
            print("Ray data saving disabled (set ENABLE_RAY_SAVING=True to enable)")
    
    @staticmethod
    def compute_and_store_ray_statistics(ray_arrays, metadata, area_index, freq_ghz, rx_index):
        """Compute and store comprehensive ray statistics in metadata"""
        if ray_arrays is None:
            return
        
        key_prefix = f"Area{area_index}_{freq_ghz}GHz_RX{rx_index}"
        
        # Initialize statistics entry for this area-frequency-rx combination
        if key_prefix not in metadata["ray_statistics_summary"]:
            metadata["ray_statistics_summary"][key_prefix] = {
                "area_index": area_index,
                "frequency_ghz": freq_ghz,
                "rx_index": rx_index,
                "parameters": {}
            }
        
        # Compute statistics for each ray parameter
        ray_params = ['tau', 'power', 'theta_r_deg', 'theta_t_deg', 'phi_r_deg', 'phi_t_deg']
        
        for param in ray_params:
            if param in ray_arrays and ray_arrays[param] is not None:
                stats = Utils.compute_ray_statistics(ray_arrays[param], param)
                if stats is not None:
                    metadata["ray_statistics_summary"][key_prefix]["parameters"][param] = stats
        
        # Add ray count information
        if 'counts' in ray_arrays:
            metadata["ray_statistics_summary"][key_prefix]["ray_counts"] = {
                "total_generated": int(ray_arrays['counts'].sum()),
                "max_rays_per_pair": int(p1a_config.MAX_RAYS_PER_PAIR),
                "utilization_ratio": float(ray_arrays['counts'].sum() / p1a_config.MAX_RAYS_PER_PAIR)
            }
    
    @staticmethod
    def save_channel_blocks(ray_arrays, save_dir, freq_ghz, rx_index):
        """Convert and save ray data in channel block format (Part 2 Cell 2)"""
        # Convert to TensorFlow variables
        tau_rays_tf = tf.Variable(ray_arrays['tau'], dtype=tf.float32)
        power_rays_tf = tf.Variable(ray_arrays['power'], dtype=tf.float32)
        phi_r_rays_tf = tf.Variable(ray_arrays['phi_r_deg'], dtype=tf.float32)
        phi_t_rays_tf = tf.Variable(ray_arrays['phi_t_deg'], dtype=tf.float32)
        theta_r_rays_tf = tf.Variable(ray_arrays['theta_r_deg'], dtype=tf.float32)
        theta_t_rays_tf = tf.Variable(ray_arrays['theta_t_deg'], dtype=tf.float32)
        
        # Convert degrees to radians
        theta_r_rays_tf = Utils.deg_to_rad(theta_r_rays_tf)
        theta_t_rays_tf = Utils.deg_to_rad(theta_t_rays_tf)
        phi_r_rays_tf = Utils.deg_to_rad(phi_r_rays_tf)
        phi_t_rays_tf = Utils.deg_to_rad(phi_t_rays_tf)
        
        # Reshape for channel block format [N_batch, N_TX, N_RX, N_cluster, N_rays]
        power_for_channel_block = tf.expand_dims(tf.expand_dims(power_rays_tf, axis=0), axis=-2)
        tau_for_channel_block = tf.expand_dims(tf.expand_dims(tau_rays_tf, axis=0), axis=-2)
        phi_r_for_channel_block = tf.expand_dims(tf.expand_dims(phi_r_rays_tf, axis=0), axis=-2)
        phi_t_for_channel_block = tf.expand_dims(tf.expand_dims(phi_t_rays_tf, axis=0), axis=-2)
        theta_r_for_channel_block = tf.expand_dims(tf.expand_dims(theta_r_rays_tf, axis=0), axis=-2)
        theta_t_for_channel_block = tf.expand_dims(tf.expand_dims(theta_t_rays_tf, axis=0), axis=-2)
        
        # Save channel block format
        channel_block_files = {
            'tau_rays_for_ChannelBlock': tau_for_channel_block,
            'power_rays_for_ChannelBlock': power_for_channel_block,
            'phi_r_rays_for_ChannelBlock': phi_r_for_channel_block,
            'phi_t_rays_for_ChannelBlock': phi_t_for_channel_block,
            'theta_r_rays_for_ChannelBlock': theta_r_for_channel_block,
            'theta_t_rays_for_ChannelBlock': theta_t_for_channel_block
        }
        
        for key, tensor in channel_block_files.items():
            filename = f"{freq_ghz}GHz_RX{rx_index}_{key}.npy"
            filepath = os.path.join(save_dir, filename)
            np.save(filepath, tensor.numpy())
            print(f"Channel block {key} saved: {filepath}")
        
        print(f"Channel block format: {tau_for_channel_block.shape}")
    
    @staticmethod
    def save_channel_blocks_unified(ray_arrays, area_index, freq_ghz, rx_index):
        """Convert and save ray data in channel block format with optimized naming (Part 2 Cell 2 + enhancement)"""
        # Convert to TensorFlow variables
        tau_rays_tf = tf.Variable(ray_arrays['tau'], dtype=tf.float32)
        power_rays_tf = tf.Variable(ray_arrays['power'], dtype=tf.float32)
        phi_r_rays_tf = tf.Variable(ray_arrays['phi_r_deg'], dtype=tf.float32)
        phi_t_rays_tf = tf.Variable(ray_arrays['phi_t_deg'], dtype=tf.float32)
        theta_r_rays_tf = tf.Variable(ray_arrays['theta_r_deg'], dtype=tf.float32)
        theta_t_rays_tf = tf.Variable(ray_arrays['theta_t_deg'], dtype=tf.float32)
        
        # Convert degrees to radians
        theta_r_rays_tf = Utils.deg_to_rad(theta_r_rays_tf)
        theta_t_rays_tf = Utils.deg_to_rad(theta_t_rays_tf)
        phi_r_rays_tf = Utils.deg_to_rad(phi_r_rays_tf)
        phi_t_rays_tf = Utils.deg_to_rad(phi_t_rays_tf)
        
        # Reshape for channel block format [N_batch, N_TX, N_RX, N_cluster, N_rays]
        power_for_channel_block = tf.expand_dims(tf.expand_dims(power_rays_tf, axis=0), axis=-2)
        tau_for_channel_block = tf.expand_dims(tf.expand_dims(tau_rays_tf, axis=0), axis=-2)
        phi_r_for_channel_block = tf.expand_dims(tf.expand_dims(phi_r_rays_tf, axis=0), axis=-2)
        phi_t_for_channel_block = tf.expand_dims(tf.expand_dims(phi_t_rays_tf, axis=0), axis=-2)
        theta_r_for_channel_block = tf.expand_dims(tf.expand_dims(theta_r_rays_tf, axis=0), axis=-2)
        theta_t_for_channel_block = tf.expand_dims(tf.expand_dims(theta_t_rays_tf, axis=0), axis=-2)
        
        # Save channel block format with optimized naming
        save_dir = p1a_config.UNIFIED_SAVE_DIR
        channel_block_files = {
            'tau_rays_for_ChannelBlock': tau_for_channel_block,
            'power_rays_for_ChannelBlock': power_for_channel_block,
            'phi_r_rays_for_ChannelBlock': phi_r_for_channel_block,
            'phi_t_rays_for_ChannelBlock': phi_t_for_channel_block,
            'theta_r_rays_for_ChannelBlock': theta_r_for_channel_block,
            'theta_t_rays_for_ChannelBlock': theta_t_for_channel_block
        }
        
        for key, tensor in channel_block_files.items():
            filename = f"Area{area_index}_{freq_ghz}GHz_ChannelBlock_RX{rx_index}_{key}.npy"
            filepath = os.path.join(save_dir, filename)
            np.save(filepath, tensor.numpy())
            print(f"Channel block {key} saved: {filepath}")
        
        print(f"Channel block format: {tau_for_channel_block.shape}")
    
    @staticmethod
    def save_metadata_json(metadata, timestamp):
        """Save comprehensive metadata as JSON file with KST timestamp"""
        if p1a_config.ENABLE_METADATA_SAVING:
            save_dir = p1a_config.UNIFIED_SAVE_DIR
            os.makedirs(save_dir, exist_ok=True)
            
            filename = f"P1A_Ray_Metadata_{timestamp}.json"
            filepath = os.path.join(save_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"Metadata saved: {filepath}")
            return filepath
        else:
            print("Metadata saving disabled (set ENABLE_METADATA_SAVING=True to enable)")
            return None
    
    @staticmethod
    def save_all_rays_unified_npz(all_ray_data, area_index, freq_ghz):
        """Save all RX ray data in unified npz format for better file management
        
        Parameters
        ----------
        all_ray_data : dict
            Dictionary containing ray arrays for all RX indices
            Format: {rx_index: ray_arrays_dict, ...}
        area_index : int
            Area index
        freq_ghz : float
            Frequency in GHz
        """
        if p1a_config.ENABLE_RAY_SAVING and all_ray_data:
            save_dir = p1a_config.UNIFIED_SAVE_DIR
            os.makedirs(save_dir, exist_ok=True)
            
            # Prepare unified data dictionary
            unified_data = {}
            rx_indices = sorted(all_ray_data.keys())
            
            # Collect metadata
            unified_data['rx_indices'] = np.array(rx_indices)
            unified_data['area_index'] = area_index
            unified_data['frequency_ghz'] = freq_ghz
            unified_data['num_rx'] = len(rx_indices)
            
            # Enhanced ray parameters including Path-Ray relationship and LoS/NLoS information
            ray_params = ['tau', 'power', 'theta_r_deg', 'theta_t_deg', 'phi_r_deg', 'phi_t_deg', 
                         'source_path_idx', 'los_nlos_flag', 'counts']
            
            # Collect all RX data for each parameter
            for param in ray_params:
                param_data_list = []
                for rx_idx in rx_indices:
                    if param in all_ray_data[rx_idx]:
                        ray_data = all_ray_data[rx_idx][param]
                        
                        # Data type optimization
                        if p1a_config.FORCE_FLOAT32_SAVING and param in ['tau', 'power', 'theta_r_deg', 'theta_t_deg', 'phi_r_deg', 'phi_t_deg']:
                            # Ray parameters: convert to float32 for P2 compatibility
                            ray_data = ray_data.astype(p1a_config.RAY_DATA_DTYPE)
                        elif param in ['source_path_idx', 'los_nlos_flag', 'counts']:
                            # Integer fields: ensure int32 dtype
                            ray_data = ray_data.astype(np.int32)
                        
                        param_data_list.append(ray_data)
                
                if param_data_list:
                    # Stack all RX data along a new RX dimension
                    unified_data[param] = np.stack(param_data_list, axis=0)  # [num_rx, ...]
            
            # Save unified npz file
            filename = f"Area{area_index}_{freq_ghz}GHz_Rays_ALL_RXs.npz"
            filepath = os.path.join(save_dir, filename)
            
            np.savez_compressed(filepath, **unified_data)
            
            print(f"Unified ray data saved: {filepath}")
            print(f"  - Contains {len(rx_indices)} RX data: RX{min(rx_indices)}-RX{max(rx_indices)}")
            print(f"  - Parameters: {list(ray_params)}")
            print(f"  - Data format: float32 optimized for P2 compatibility")
            
            # Save channel blocks if enabled
            if p1a_config.ENABLE_CHANNEL_BLOCK_SAVING:
                DataIO.save_all_channel_blocks_unified_npz(all_ray_data, area_index, freq_ghz)
        else:
            print("Unified ray data saving disabled or no data provided")
    
    @staticmethod
    def save_all_channel_blocks_unified_npz(all_ray_data, area_index, freq_ghz):
        """Save all RX channel block data in unified npz format"""
        save_dir = p1a_config.UNIFIED_SAVE_DIR
        
        unified_channel_blocks = {}
        rx_indices = sorted(all_ray_data.keys())
        
        # Channel block parameters
        channel_block_params = [
            'tau_rays_for_ChannelBlock', 'power_rays_for_ChannelBlock',
            'phi_r_rays_for_ChannelBlock', 'phi_t_rays_for_ChannelBlock', 
            'theta_r_rays_for_ChannelBlock', 'theta_t_rays_for_ChannelBlock'
        ]
        
        for param in channel_block_params:
            param_data_list = []
            for rx_idx in rx_indices:
                ray_arrays = all_ray_data[rx_idx]
                
                # Convert ray data to channel block format
                if param.startswith('tau'):
                    ray_data = ray_arrays['tau']
                elif param.startswith('power'):
                    ray_data = ray_arrays['power'] 
                elif param.startswith('phi_r'):
                    ray_data = Utils.deg_to_rad(ray_arrays['phi_r_deg'])  # Convert from stored degrees
                elif param.startswith('phi_t'):
                    ray_data = Utils.deg_to_rad(ray_arrays['phi_t_deg'])  # Convert from stored degrees
                elif param.startswith('theta_r'):
                    ray_data = Utils.deg_to_rad(ray_arrays['theta_r_deg'])  # Convert from stored degrees
                elif param.startswith('theta_t'):
                    ray_data = Utils.deg_to_rad(ray_arrays['theta_t_deg'])  # Convert from stored degrees
                
                # Convert to TensorFlow and reshape for channel block format
                ray_data_tf = tf.Variable(ray_data, dtype=tf.float32)
                channel_block_data = tf.expand_dims(tf.expand_dims(ray_data_tf, axis=0), axis=-2)
                param_data_list.append(channel_block_data.numpy())
            
            if param_data_list:
                unified_channel_blocks[param] = np.stack(param_data_list, axis=0)
        
        # Save unified channel block npz file
        filename = f"Area{area_index}_{freq_ghz}GHz_ChannelBlocks_ALL_RXs.npz"
        filepath = os.path.join(save_dir, filename)
        
        np.savez_compressed(filepath, **unified_channel_blocks)
        print(f"Unified channel blocks saved: {filepath}")

class RayGen:
    """TR 38.901-based ray generator for statistical channel modeling (Part 2 Cell 2)
    
    Parameters traced to TR 38.901 v16.1.0:
    - Table 7.5-6 Part-1: Channel model parameters (c_ASA, c_ASD, r_tau, N, M, XPR)
    - Table 7.5-3: Ray offset angles within cluster (SUBRAY_PROB distribution) 
    - Table 7.5-5: Sub-cluster information (LOS_COMB_N, NLOS_COMB_N mapping)
    - Step procedures in Clause 7.5: Fast fading model (uniform/Gaussian distributions)
    """
    
    def __init__(self, carrier_frequency, los_scenario):
        """Initialize ray generator with TR 38.901 parameters (Part 2 Cell 2, simplified)"""
        self.carrier_frequency = carrier_frequency
        self.los_scenario = los_scenario
        self.wavelength = SPEED_OF_LIGHT / carrier_frequency
        
        # Calculate c_DS using TR 38.901 formula
        self.c_DS = max(0.25, 6.5622 - 3.4084 * math.log10(carrier_frequency)) * 1e-9
        self.c_ZSA = p1a_config.TR38901_C_ZSA
        self.mu_lgZSD = p1a_config.TR38901_MU_LGZSD
        self.c_ZSD = 3 / 8 * 10 ** (self.mu_lgZSD)
        
        # Set scenario-specific parameters from TR 38.901
        if los_scenario == 1:  # LOS
            self.c_ASA = p1a_config.LOS_C_ASA
            self.c_ASD = p1a_config.LOS_C_ASD
            self.r_tau = p1a_config.LOS_R_TAU
            self.N = p1a_config.LOS_N_CLUSTERS
            self.DS = -6.955 - 0.0963*math.log10(carrier_frequency)
        else:  # NLOS
            self.c_ASA = p1a_config.NLOS_C_ASA
            self.c_ASD = p1a_config.NLOS_C_ASD
            self.r_tau = p1a_config.NLOS_R_TAU
            self.N = p1a_config.NLOS_N_CLUSTERS
            self.DS = -0.24*math.log10(1+carrier_frequency) - 7.14
    
    def generate_subray_angles(self, cluster_asa, cluster_asd):
        """Generate sub-ray angles using TR 38.901 statistical parameters (Part 2 Cell 2)"""
        # Use config parameters for sub-ray generation
        M_actual = 20  # Number of rays per cluster
        
        # Generate angle offsets using implementation-specific probabilities
        phi_offsets = np.random.normal(0, cluster_asa/7, M_actual)
        theta_offsets = np.random.normal(0, cluster_asd/7, M_actual)
        
        return phi_offsets, theta_offsets
    
    def subrayProposed(self, path_interactions, los_scenario, tau_cluster, 
                      phi_r_cluster, phi_t_cluster, theta_r_cluster, theta_t_cluster, power_cluster):
        """Generate sub-rays from cluster parameters using TR 38.901 procedures (Part 2 Cell 2)"""
        # Determine sub-cluster mapping
        if los_scenario == 1:
            comb_n = p1a_config.LOS_COMB_N
        else:
            comb_n = p1a_config.NLOS_COMB_N
        
        # Generate sub-ray parameters
        M_actual = 20  # Number of rays per cluster
        
        # Sub-ray delays using c_DS parameter
        tau_subrays = tau_cluster + np.random.exponential(scale=self.c_DS, size=M_actual)
        
        # Sub-ray angles
        phi_r_offsets, theta_r_offsets = self.generate_subray_angles(self.c_ASA, self.c_ASD)
        phi_t_offsets, theta_t_offsets = self.generate_subray_angles(self.c_ASD, self.c_ZSA)
        
        phi_r_subrays = phi_r_cluster + phi_r_offsets
        phi_t_subrays = phi_t_cluster + phi_t_offsets
        theta_r_subrays = theta_r_cluster + theta_r_offsets
        theta_t_subrays = theta_t_cluster + theta_t_offsets
        
        # Sub-ray powers using binomial distribution
        power_subrays = power_cluster * np.random.exponential(scale=1.0, size=M_actual) / M_actual
        
        # Channel spread parameters
        ASA = self.c_ASA
        ASD = self.c_ASD
        ZSA = self.c_ZSA
        ZSD = self.c_ZSD
        
        return (M_actual, tau_subrays, power_subrays, phi_r_subrays, phi_t_subrays, 
                theta_r_subrays, theta_t_subrays, ASA, ASD, ZSA, ZSD)
    
    @staticmethod
    def initialize_storage():
        """Initialize arrays for storing generated rays (Part 2 Cell 2)
        Note: Creates 5D arrays [batch, N_BS, N_UE, cluster, rays] for Sionna compatibility (batch=1 is formal dimension)
        
        Enhanced P1A: TR 38.901 compatible storage format
        Shape: [batch_size, N_BS, N_UE, cluster_dim, N_rays] for standard compliance
        """
        # Use configuration parameters for dimensional structure
        batch_size = p1a_config.BATCH_SIZE
        num_tx = p1a_config.N_BS      # N_BS (Base Stations)
        num_rx = p1a_config.N_UE      # N_UE (User Equipment)
        cluster_dim = p1a_config.CLUSTER_DIM # Cluster dimension (TR 38.901 requirement)
        max_rays = p1a_config.MAX_RAYS_PER_PAIR
        
        if p1a_config.USE_SIONNA_FORMAT:
            # Sionna compatible storage format: [B, N_BS, N_UE, cluster_dim, N_rays]
            ray_shape = (batch_size, num_tx, num_rx, cluster_dim, max_rays)
            counts_shape = (batch_size, num_tx, num_rx)
            format_name = "Sionna compatible"
        else:
            # Legacy format for backward compatibility
            ray_shape = (num_tx, num_rx, max_rays)
            counts_shape = (num_tx, num_rx)
            format_name = "Legacy"
        
        ray_arrays = {
            'tau': np.zeros(ray_shape),
            'power': np.zeros(ray_shape),
            'phi_r_deg': np.zeros(ray_shape),
            'phi_t_deg': np.zeros(ray_shape),
            'theta_r_deg': np.zeros(ray_shape),
            'theta_t_deg': np.zeros(ray_shape),
            'source_path_idx': np.zeros(ray_shape, dtype=np.int32),  # Path-Ray relationship tracking
            'los_nlos_flag': np.zeros(ray_shape, dtype=np.int32),    # LoS(1)/NLoS(0) flag for each ray
            'counts': np.zeros(counts_shape, dtype=np.int32)
        }
        
        print(f"[P1A] Initialized {format_name} storage: {ray_shape}")
        print(f"[P1A] Batch processing: {p1a_config.BATCH_PROCESSING}, Batch size: {batch_size}")
        print(f"[P1A] Sionna format: {p1a_config.USE_SIONNA_FORMAT}")
        return ray_arrays, num_tx, num_rx
    
    @staticmethod
    def determine_scenario(path_types, tx, rx):
        """Determine LOS scenario from path interaction types (Part 2 Cell 2)"""
        if path_types[tx, rx, 0, 0] == 0:
            return 1  # LOS
        else:
            return 0  # NLOS
    
    @staticmethod
    def generate_for_pairs(path_data, carrier_frequency, ray_arrays, num_tx, num_rx):
        """Generate rays for all TX-RX pairs using RayGen (Part 2 Cell 2)"""
        max_rays_per_pair = p1a_config.MAX_RAYS_PER_PAIR
        
        # Extract path data arrays
        power_sorted = path_data['power']
        tau_sorted = path_data['tau']
        theta_r_deg_sorted = path_data['theta_r_deg']
        theta_t_deg_sorted = path_data['theta_t_deg']
        phi_r_deg_sorted = path_data['phi_r_deg']
        phi_t_deg_sorted = path_data['phi_t_deg']
        path_types_sorted = path_data['interactions']
        
        for tx in range(num_tx):
            for rx in range(num_rx):
                print(f"Generating rays for TX {tx}, RX {rx}")
                
                # Determine LOS scenario
                los_scenario = RayGen.determine_scenario(path_types_sorted, tx, rx)
                print(f"Scenario: {'LOS' if los_scenario == 1 else 'NLOS'}")
                
                # Create ray generator
                ray_generator = RayGen(carrier_frequency, los_scenario)
                
                # Generate rays from all paths
                rays_from_paths = RayGen.generate_from_paths(
                    ray_generator, path_types_sorted, los_scenario, 
                    tau_sorted, phi_r_deg_sorted, phi_t_deg_sorted, 
                    theta_r_deg_sorted, theta_t_deg_sorted, power_sorted, tx, rx
                )
                
                # Store generated rays
                RayGen.store_rays(rays_from_paths, ray_arrays, tx, rx, max_rays_per_pair)
    
    @staticmethod
    def generate_from_paths(ray_generator, path_types, los_scenario, tau_sorted, 
                           phi_r_deg_sorted, phi_t_deg_sorted, theta_r_deg_sorted, theta_t_deg_sorted, 
                           power_sorted, tx, rx):
        """Generate rays from all paths for specific TX-RX pair with Path-Ray relationship tracking"""
        all_tau_rays = []
        all_power_rays = []
        all_phi_r_rays = []
        all_phi_t_rays = []
        all_theta_r_rays = []
        all_theta_t_rays = []
        all_source_path_idx = []  # Track which path each ray came from
        all_los_nlos_flag = []    # Track LoS/NLoS flag for each ray
        
        num_paths = len(tau_sorted[tx, rx, :])
        
        for path_idx in range(num_paths):
            # Extract path data
            path_interactions = path_types[tx, rx, path_idx, :]
            path_tau = tau_sorted[tx, rx, path_idx]
            path_phi_r_deg = phi_r_deg_sorted[tx, rx, path_idx]
            path_phi_t_deg = phi_t_deg_sorted[tx, rx, path_idx]
            path_theta_r_deg = theta_r_deg_sorted[tx, rx, path_idx]
            path_theta_t_deg = theta_t_deg_sorted[tx, rx, path_idx]
            path_power = power_sorted[tx, rx, path_idx]
            
            # Determine LoS/NLoS for this specific path
            path_los_flag = 1 if path_interactions[0] == 0 else 0  # LoS=1, NLoS=0
            
            # Generate sub-rays from path
            M, tau_ray, power_ray, phi_r_ray, phi_t_ray, theta_r_ray, theta_t_ray, ASA, ASD, ZSA, ZSD = ray_generator.subrayProposed(
                path_interactions, 
                los_scenario, 
                path_tau, 
                path_phi_r_deg, 
                path_phi_t_deg, 
                path_theta_r_deg, 
                path_theta_t_deg, 
                path_power
            )
            
            # Store rays with Path-Ray relationship information
            if isinstance(tau_ray, np.ndarray) and len(tau_ray) > 0:
                all_tau_rays.extend(tau_ray)
                all_power_rays.extend(power_ray)
                all_phi_r_rays.extend(phi_r_ray)
                all_phi_t_rays.extend(phi_t_ray)
                all_theta_r_rays.extend(theta_r_ray)
                all_theta_t_rays.extend(theta_t_ray)
                
                # Add Path-Ray relationship information for each generated ray
                all_source_path_idx.extend([path_idx] * len(tau_ray))    # Track source path index
                all_los_nlos_flag.extend([path_los_flag] * len(tau_ray)) # Track LoS/NLoS flag
        
        return {
            'tau': np.array(all_tau_rays),
            'power': np.array(all_power_rays),
            'phi_r_deg': np.array(all_phi_r_rays),
            'phi_t_deg': np.array(all_phi_t_rays),
            'theta_r_deg': np.array(all_theta_r_rays),
            'theta_t_deg': np.array(all_theta_t_rays),
            'source_path_idx': np.array(all_source_path_idx, dtype=np.int32),
            'los_nlos_flag': np.array(all_los_nlos_flag, dtype=np.int32)
        }
    
    @staticmethod
    def store_rays(rays_from_paths, ray_arrays, tx, rx, max_rays_per_pair):
        """Store generated rays with Path-Ray relationship and LoS/NLoS information"""
        total_rays = len(rays_from_paths['tau'])
        
        # Include new Path-Ray relationship fields
        ray_fields = ['tau', 'power', 'phi_r_deg', 'phi_t_deg', 'theta_r_deg', 'theta_t_deg', 
                      'source_path_idx', 'los_nlos_flag']
        
        if total_rays > max_rays_per_pair:
            print(f"TX {tx}, RX {rx}: {total_rays} rays generated, selecting top {max_rays_per_pair}")
            # Select top rays by power
            sort_indices = np.argsort(-rays_from_paths['power'])[:max_rays_per_pair]
            
            for key in ray_fields:
                if p1a_config.USE_SIONNA_FORMAT:
                    ray_arrays[key][0, tx, rx, 0, :max_rays_per_pair] = rays_from_paths[key][sort_indices]
                else:
                    ray_arrays[key][tx, rx, :max_rays_per_pair] = rays_from_paths[key][sort_indices]
            
            if p1a_config.USE_SIONNA_FORMAT:
                ray_arrays['counts'][0, tx, rx] = max_rays_per_pair
            else:
                ray_arrays['counts'][tx, rx] = max_rays_per_pair
        else:
            print(f"TX {tx}, RX {rx}: {total_rays} rays generated")
            
            for key in ray_fields:
                if p1a_config.USE_SIONNA_FORMAT:
                    ray_arrays[key][0, tx, rx, 0, :total_rays] = rays_from_paths[key]
                else:
                    ray_arrays[key][tx, rx, :total_rays] = rays_from_paths[key]
            
            if p1a_config.USE_SIONNA_FORMAT:
                ray_arrays['counts'][0, tx, rx] = total_rays
            else:
                ray_arrays['counts'][tx, rx] = total_rays

class Pipeline:
    """파이프라인 실행 및 관리 클래스 (Part 1 + Part 2)"""
    
    @staticmethod
    def execute_preview_only():
        """AREA_CONFIGS 기반 TX/RX 배치 3D 시각화 프리뷰 (enhancement)"""
        print("=" * 80)
        print("AREA_CONFIGS Visual Preview Mode")
        print("3D Scene Visualization with TX/RX Placement - No RT Computation")
        print("=" * 80)
        
        try:
            # Load scene for visualization
            print("Loading scene for preview...")
            scene = Scene.load()
            
            # Configure scene once for all areas
            print("Configuring antennas and materials...")
            Scene.configure_arrays_and_materials(scene)
            
            # Process all configured areas for visual preview
            areas = p1a_config.AREA_INDICES
            frequencies = p1a_config.FREQUENCY_CONFIGS
            print(f"Preview Mode: Processing {len(areas)} areas: {areas}")
            print(f"Frequencies: {frequencies} GHz")
            
            for area_index in areas:
                area_config = Utils.get_area_config(area_index)
                print(f"\n=== AREA {area_index} VISUAL PREVIEW ===")
                print(f"Area {area_index}: {area_config['description']}")
                
                # Clear existing devices for this area preview
                scene = Scene.clear_devices(scene)
                
                # Deploy TX and RX for preview
                Scene.add_transmitters(scene, area_index)
                Scene.add_receivers(scene, area_index)
                
                # Show configuration summary
                tx_positions = area_config['tx_positions']
                total_rx = len(scene.receivers)
                placement_config = area_config.get('rx_placement', {})
                method = placement_config.get('method', 'explicit')

                print(f"TX Positions: {len(tx_positions)} transmitters")
                for i, pos in enumerate(tx_positions):
                    print(f"  - TX{i+1}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
                
                print(f"RX Placement Method: '{method}' ({total_rx} receivers)")
                
                if method == 'grid':
                    x_p = placement_config['x_params']
                    y_p = placement_config['y_params']
                    z_p = placement_config['z_params']
                    print(f"  - X params: start={x_p['start']}, stop={x_p['stop']}, num={x_p['num']}")
                    print(f"  - Y params: start={y_p['start']}, stop={y_p['stop']}, num={y_p['num']}")
                    print(f"  - Z values: {z_p['values']}")
                elif method == 'radial':
                    c = placement_config['center']
                    r = placement_config['radii_m']
                    a = placement_config['angles_deg']
                    z = placement_config['z_params']['values']
                    print(f"  - Center: {c}")
                    print(f"  - Radii (m): {r}")
                    print(f"  - Angles (deg): start={a['start']}, stop={a['stop']}, num={a['num']}")
                    print(f"  - Z values: {z}")
                elif method == 'street':
                    p = placement_config['path_points']
                    n = placement_config['num_points']
                    z = placement_config['z_params']['values']
                    print(f"  - Path: from {p[0]} to {p[1]}")
                    print(f"  - Number of points: {n}")
                    print(f"  - Z values: {z}")
                elif method == 'explicit':
                    # Fallback for old format or explicit definition
                    rx_x = placement_config.get('x_coords') or area_config.get('rx_x_coords')
                    rx_y = placement_config.get('y_coords') or area_config.get('rx_y_coords')
                    rx_z = placement_config.get('z_coords') or area_config.get('rx_z_coords')
                    print(f"  - X: {rx_x}")
                    print(f"  - Y: {rx_y}")
                    print(f"  - Z: {rx_z}")

                # Display 3D scene preview
                print(f"\nDisplaying 3D scene preview for Area {area_index}...")
                Scene.preview(scene)
                
                # Show expected output files summary
                print(f"\nExpected output files: {total_rx * len(frequencies) * 7} files")
                print(f"  - Path data: Area{area_index}_{{freq}}GHz_RX{{n}}_{{type}}.npy")
                print(f"  - Ray data: Area{area_index}_{{freq}}GHz_RX{{n}}_rays_{{type}}.npy")
            
            print(f"\n=== Visual Preview Completed for {len(areas)} areas ===")
            print("Set PREVIEW_ONLY_MODE=False to run full RT pipeline with computation")
            
        except Exception as e:
            print(f"\nPreview failed: {e}")
            print("This may happen in headless environments or without display support")
            print("Falling back to configuration summary...")
            
            # Fallback: show configuration info only
            areas = p1a_config.AREA_INDICES
            for area_index in areas:
                area_config = Utils.get_area_config(area_index)
                print(f"\nArea {area_index}: {area_config['description']}")
                print(f"TX: {len(area_config['tx_positions'])} transmitters")
                rx_count = Utils.get_rx_count_from_config(area_config)
                print(f"RX: {rx_count} receivers")
    
    @staticmethod
    def execute_main():
        """Main execution pipeline with memory-based path-to-ray processing (Enhanced)"""
        
        # Check for preview-only mode
        if p1a_config.PREVIEW_ONLY_MODE:
            Pipeline.execute_preview_only()
            return
        
        print("=" * 80)
        print("P1A RT to Paths & Rays Integrated Pipeline")
        print("Multi-area & Multi-frequency with TR 38.901 Batch Processing")
        print("=" * 80)
        
        # TR 38.901 Batch Processing Configuration
        batch_size = p1a_config.BATCH_SIZE
        batch_processing = p1a_config.BATCH_PROCESSING
        
        if batch_processing and batch_size > 1:
            print(f"[P1A] Batch processing enabled: {batch_size} batches")
            for batch_idx in range(batch_size):
                print(f"\n=== BATCH {batch_idx + 1}/{batch_size} ===")
                Pipeline.execute_single_batch(batch_idx)
        else:
            print(f"[P1A] Single batch processing (batch_size={batch_size}) - formal batch dimension only")
            Pipeline.execute_single_batch(0)
    
    @staticmethod
    def execute_single_batch(batch_idx):
        """Execute single batch of the pipeline
        Note: batch_idx=0 for current single-batch implementation (formal batch dimension)"""
        print(f"[P1A] Processing batch {batch_idx} with Sionna compatibility")
        
        # Initialize scene and memory storage
        scene = Scene.load()
        memory_path_data = {}  # Store all path data in memory
        
        # Prepare run configuration for metadata
        areas = p1a_config.AREA_INDICES
        frequencies = p1a_config.FREQUENCY_CONFIGS
        run_config = {"areas": areas, "frequencies": frequencies, "batch_idx": batch_idx}
        
        # Execute integrated workflow (memory-based)
        path_summary, ray_summary, metadata = Pipeline.execute_integrated_workflow(scene, memory_path_data, run_config)
        
        # Save metadata with ray statistics (include batch info)
        _, timestamp = Utils.create_metadata(path_summary, ray_summary, run_config)
        
        # Add batch information to metadata
        if p1a_config.BATCH_PROCESSING:
            metadata["batch_info"] = {
                "batch_index": batch_idx,
                "total_batches": p1a_config.BATCH_SIZE,
                "sionna_format": p1a_config.USE_SIONNA_FORMAT
            }
        
        DataIO.save_metadata_json(metadata, timestamp)
        
        # Print completion summary
        print(f"\n=== Batch {batch_idx} Completed Successfully ===")
        completed_tasks = [f"path data generation for {len(frequencies)} frequencies (memory-only)"]
        
        if p1a_config.ENABLE_RAY_GENERATION:
            if p1a_config.ENABLE_RAY_SAVING:
                completed_tasks.append("ray data generation and saving")
            else:
                completed_tasks.append("ray data generation (saving disabled)")
        
        if p1a_config.ENABLE_METADATA_SAVING:
            completed_tasks.append(f"metadata saved with timestamp {timestamp}")
        
        print(f"Completed: {', '.join(completed_tasks)}")
        print(f"Processed: {len(areas)} areas × {len(frequencies)} frequencies = {len(areas) * len(frequencies)} combinations")
        Utils.print_vram()
    
    @staticmethod
    def execute_integrated_workflow(scene, memory_path_data, run_config):
        """Execute integrated Path-to-Ray workflow with memory-based processing"""
        areas = run_config["areas"]
        frequencies = run_config["frequencies"]
        path_summary = {"total_areas": len(areas), "total_frequencies": len(frequencies), "path_data": {}}
        ray_summary = {"total_ray_files": 0, "ray_data": {}}
        
        # Initialize metadata with ray statistics
        metadata, _ = Utils.create_metadata(path_summary, ray_summary, run_config)
        
        print("\n--- INTEGRATED WORKFLOW: Memory-based Path-to-Ray Processing ---")
        print(f"Processing {len(areas)} areas × {len(frequencies)} frequencies = {len(areas) * len(frequencies)} combinations")
        
        for area_idx, area_index in enumerate(areas):
            print(f"\n=== AREA {area_index} ({area_idx + 1}/{len(areas)}) ===")
            area_config = Utils.get_area_config(area_index)
            print(f"Area {area_index}: {area_config['description']}")
            
            # Configure scene for this area
            Scene.configure_arrays_and_materials(scene)
            Scene.clear_devices(scene)
            Scene.add_transmitters(scene, area_index)
            Scene.add_receivers(scene, area_index)
            
            print(f"Area {area_index}: Scene configured with {len(scene.transmitters)} TX and {len(scene.receivers)} RX")
            Scene.preview(scene)
            
            # Process all frequencies for this area
            for freq_idx, frequency_ghz in enumerate(frequencies):
                print(f"\n--- Area {area_index} | Frequency {freq_idx + 1}/{len(frequencies)}: {frequency_ghz} GHz ---")
                
                # Generate path data (memory only)
                path_data_dict = Pipeline.process_frequency_memory_only(scene, frequency_ghz, area_index)
                
                # Store path data summary
                key = f"Area{area_index}_{frequency_ghz}GHz"
                path_summary["path_data"][key] = {
                    "frequency_ghz": frequency_ghz,
                    "area_index": area_index,
                    "num_tx": len(scene.transmitters),
                    "num_rx": len(scene.receivers),
                    "num_paths": path_data_dict[0].shape[2] if path_data_dict else 0  # power is first element
                }
                
                # Generate rays directly from memory path data
                if p1a_config.ENABLE_RAY_GENERATION and path_data_dict:
                    rx_count = len(scene.receivers)
                    
                    # Collect all RX ray data for unified saving
                    all_ray_data = {}
                    
                    print(f"Generating rays for Area{area_index}_{frequency_ghz}GHz (RX1-RX{rx_count})")
                    
                    for rx_index in range(1, rx_count + 1):
                        # Extract path data for this RX
                        rx_path_data = Pipeline.extract_rx_path_data(path_data_dict, rx_index)
                        
                        if rx_path_data:
                            # Generate rays
                            carrier_frequency = PathRT.setup_rf(frequency_ghz)
                            ray_arrays, _, _ = RayGen.initialize_storage()
                            RayGen.generate_for_pairs(rx_path_data, carrier_frequency, ray_arrays, 1, 1)
                            
                            # Compute and store ray statistics in metadata
                            DataIO.compute_and_store_ray_statistics(ray_arrays, metadata, area_index, frequency_ghz, rx_index)
                            
                            # Collect ray data for unified saving
                            all_ray_data[rx_index] = ray_arrays
                            
                            # Store ray data summary
                            ray_key = f"Area{area_index}_{frequency_ghz}GHz_RX{rx_index}"
                            ray_summary["ray_data"][ray_key] = {
                                "frequency_ghz": frequency_ghz,
                                "area_index": area_index,
                                "rx_index": rx_index,
                                "num_rays": ray_arrays["counts"][0, 0]
                            }
                    
                    # Save unified ray data for this area-frequency combination
                    if all_ray_data:
                        DataIO.save_all_rays_unified_npz(all_ray_data, area_index, frequency_ghz)
                        ray_summary["total_ray_files"] += 1  # One unified file per area-frequency
        
        return path_summary, ray_summary, metadata
    
    @staticmethod
    def extract_rx_path_data(path_data_dict, rx_index):
        """Extract path data for specific RX from memory storage"""
        try:
            selected_data = path_data_dict
            if not selected_data:
                return None
            
            (power_selected, tau_selected, theta_r_deg_selected, phi_r_deg_selected, 
             theta_t_deg_selected, phi_t_deg_selected, path_types_selected, vertices_selected) = selected_data
            
            # Extract data for this RX (adjust for 0-based indexing)
            rx_array_idx = rx_index - 1
            
            rx_path_data = {
                'power': power_selected[:, rx_array_idx:rx_array_idx+1, :],
                'tau': tau_selected[:, rx_array_idx:rx_array_idx+1, :],
                'theta_r_deg': theta_r_deg_selected[:, rx_array_idx:rx_array_idx+1, :],
                'phi_r_deg': phi_r_deg_selected[:, rx_array_idx:rx_array_idx+1, :],
                'theta_t_deg': theta_t_deg_selected[:, rx_array_idx:rx_array_idx+1, :],
                'phi_t_deg': phi_t_deg_selected[:, rx_array_idx:rx_array_idx+1, :],
                'interactions': path_types_selected[:, rx_array_idx:rx_array_idx+1, :, :],
                'vertices': vertices_selected[:, rx_array_idx:rx_array_idx+1, :, :, :]
            }
            
            # Convert to numpy
            return Utils.to_numpy(rx_path_data)
            
        except Exception as e:
            print(f"Error extracting RX {rx_index} path data: {e}")
            return None
    
    @staticmethod
    def process_frequency_memory_only(scene, frequency_ghz, area_index):
        """Process ray tracing for a single frequency (memory only, no file saving)"""
        print(f"Processing Area {area_index}, frequency {frequency_ghz} GHz (memory only)")
        
        # Configure RF parameters for this frequency
        carrier_frequency = PathRT.setup_rf(frequency_ghz)
        
        # Execute ray tracing
        paths, wavelength = PathRT.trace_rays(scene, carrier_frequency, frequency_ghz)
        
        # Process and sort paths following original notebook logic
        a_, tau = PathRT.extract_cir(paths)
        a_, tau = PathRT.reshape_cir(a_, tau)
        a_, theta_r_deg, phi_r_deg, theta_t_deg, phi_t_deg = PathRT.compute_angles(a_, tau, paths)
        
        # Reshape path types and vertices before sorting
        path_types_reshaped, vertices_reshaped = PathRT.process_interactions(paths.interactions, paths.vertices)
        
        # Sort and select strongest paths
        sorted_data = PathRT.sort_by_power(a_, tau, theta_r_deg, phi_r_deg, theta_t_deg, phi_t_deg, 
                                         path_types_reshaped, vertices_reshaped)
        selected_data = PathRT.select_top(sorted_data, p1a_config.NUM_INTERESTING_PATHS)
        
        print(f"Area {area_index}, frequency {frequency_ghz} GHz processing completed (memory only)")
        return selected_data
    
    @staticmethod
    def execute_part1(scene):
        """Execute complete RT path generation workflow with multi-area and multi-frequency support (Part 1 Cells 3-7)"""
        print("\n--- PART 1: RT Path Generation (Multi-area & Multi-frequency) ---")
        
        # Process all areas and frequencies
        areas = p1a_config.AREA_INDICES
        frequencies = p1a_config.FREQUENCY_CONFIGS
        print(f"Part 1: Processing {len(areas)} areas: {areas} with {len(frequencies)} frequencies: {frequencies} GHz")
        print(f"Total combinations: {len(areas)} × {len(frequencies)} = {len(areas) * len(frequencies)}")
        
        for area_idx, area_index in enumerate(areas):
            print(f"\n=== AREA {area_index} ({area_idx + 1}/{len(areas)}) ===")
            area_config = Utils.get_area_config(area_index)
            print(f"Area {area_index}: {area_config['description']}")
            
            # Configure scene for this area
            Scene.configure_arrays_and_materials(scene)
            Scene.clear_devices(scene)
            Scene.add_transmitters(scene, area_index)
            Scene.add_receivers(scene, area_index)
            
            print(f"Area {area_index}: Scene configured with {len(scene.transmitters)} TX and {len(scene.receivers)} RX")
            Scene.preview(scene)
            
            # Process all frequencies for this area
            for freq_idx, frequency_ghz in enumerate(frequencies):
                print(f"\n--- Area {area_index} | Frequency {freq_idx + 1}/{len(frequencies)}: {frequency_ghz} GHz ---")
                Pipeline.process_frequency(scene, frequency_ghz, area_index)
        
        print(f"\nPart 1 completed for {len(areas)} areas and {len(frequencies)} frequencies")
        print(f"Total processed: {len(areas) * len(frequencies)} area-frequency combinations")
    
    @staticmethod
    def execute_part2():
        """Execute complete path to ray conversion workflow with multi-area and multi-frequency support (Part 2 Cell 2 + enhancement)"""
        
        if p1a_config.ENABLE_RAY_GENERATION:
            print("\n--- PART 2: Path to Ray Conversion (Multi-area & Multi-frequency) ---")
            
            # Get processing parameters from configurations
            areas = p1a_config.AREA_INDICES
            frequencies = p1a_config.FREQUENCY_CONFIGS
            
            print(f"Part 2: Processing {len(areas)} areas: {areas} with {len(frequencies)} frequencies: {frequencies} GHz")
            
            for area_index in areas:
                area_config = Utils.get_area_config(area_index)
                print(f"\n=== PART 2: AREA {area_index} ===")
                print(f"Area {area_index}: {area_config['description']}")
                
                # Calculate total RX positions for this area
                rx_coords = area_config['rx_x_coords']
                ry_coords = area_config['rx_y_coords'] 
                rz_coords = area_config['rx_z_coords']
                total_rx = len(rx_coords) * len(ry_coords) * len(rz_coords)
                
                for freq_ghz in frequencies:
                    for rx_index in range(1, total_rx + 1):
                        print(f"\nProcessing Area{area_index}_{freq_ghz}GHz_RX{rx_index}")
                        
                        # Load path data using unified format
                        path_data = DataIO.load_paths_unified(area_index, freq_ghz, rx_index)
                        if path_data is None:
                            continue
                        
                        # Configure RF system parameters
                        carrier_frequency = PathRT.setup_rf(freq_ghz)
                        
                        # Initialize ray storage
                        ray_arrays, num_tx, num_rx = RayGen.initialize_storage()
                        
                        # Generate rays for each TX-RX pair
                        RayGen.generate_for_pairs(path_data, carrier_frequency, ray_arrays, num_tx, num_rx)
                        
                        # Save ray data if enabled (unified folder)
                        DataIO.save_rays_unified(ray_arrays, area_index, freq_ghz, rx_index)
            
            print(f"Part 2 completed for {len(areas)} areas and {len(frequencies)} frequencies")
        else:
            print("\n--- PART 2: Path to Ray Conversion (DISABLED) ---")
            print("Ray generation disabled (set ENABLE_RAY_GENERATION=True to enable)")
    
    @staticmethod
    def process_frequency(scene, frequency_ghz, area_index):
        """Process ray tracing for a single frequency and area using modular approach (Part 1 Cells 2,6,7)"""
        print(f"Processing Area {area_index}, frequency {frequency_ghz} GHz")
        
        # Configure RF parameters for this frequency
        carrier_frequency = PathRT.setup_rf(frequency_ghz)
        
        # Execute ray tracing
        paths, wavelength = PathRT.trace_rays(scene, carrier_frequency, frequency_ghz)
        
        # Cell 7: Process and sort paths following original notebook logic
        a_, tau = PathRT.extract_cir(paths)
        a_, tau = PathRT.reshape_cir(a_, tau)
        a_, theta_r_deg, phi_r_deg, theta_t_deg, phi_t_deg = PathRT.compute_angles(a_, tau, paths)
        
        # Cell 7: Reshape path types and vertices before sorting (original notebook order)
        path_types_reshaped, vertices_reshaped = PathRT.process_interactions(paths.interactions, paths.vertices)
        
        # Cell 7: Sort and select strongest paths
        sorted_data = PathRT.sort_by_power(a_, tau, theta_r_deg, phi_r_deg, theta_t_deg, phi_t_deg, 
                                         path_types_reshaped, vertices_reshaped)
        selected_data = PathRT.select_top(sorted_data, p1a_config.NUM_INTERESTING_PATHS)
        
        # Save results if enabled (Unified folder structure for all/selected RX)
        DataIO.save_all_rx_paths_unified(selected_data, carrier_frequency, area_index, scene)
        
        print(f"Area {area_index}, frequency {frequency_ghz} GHz processing completed")


# ========================================================================
# EXECUTION PIPELINE
# ========================================================================

if __name__ == "__main__":
    # Execute the main pipeline
    Pipeline.execute_main()
import argparse
import json
import logging
import os
import os.path as osp
import sys
import threading
import time
from typing import List, Optional, Tuple

try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None

import numpy as np
import torch
import torch.utils.data
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from config import make_cfg
from model import create_model
from geotransformer.modules.registration.metrics import isotropic_transform_error
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
)
from geotransformer.utils.pointcloud import (
    apply_transform,
    get_transform_from_rotation_translation,
)
from geotransformer.utils.torch import to_cuda


# -----------------------------------------------------------------------------
# 路径配置
# -----------------------------------------------------------------------------
WORKING_DIR = osp.dirname(osp.realpath(__file__))
ROOT_DIR = osp.dirname(osp.dirname(WORKING_DIR))

# 固定路径配置：将数据放在同一文件夹内，仅需修改这里
# DATA_DIR = '/home/xchu/data/ltloc_result/parkinglot_raw_geo'  # TODO: 修改为包含先验地图、帧点云和 TUM 位姿的目录
# PRIOR_MAP_FILENAME = 'parkinglot_raw.pcd'

DATA_DIR='/home/xchu/data/ltloc_result/geo_transformer/stairs_bob_geo'
PRIOR_MAP_FILENAME = 'stairs_bob.pcd'

# DATA_DIR='/home/xchu/data/ltloc_result/geo_transformer/20220216_corridor_day_ref_geo'
# PRIOR_MAP_FILENAME = '20220216_corridor_day_ref.pcd'

# DATA_DIR = '/home/xchu/data/ltloc_result/geo_transformer/20220225_building_day_ref_geo'
# PRIOR_MAP_FILENAME = '20220225_building_day_ref.pcd'

# DATA_DIR = '/home/xchu/data/ltloc_result/cave02_geo'
# PRIOR_MAP_FILENAME = 'lauren_cavern01.pcd'

TUM_FILENAME = 'optimized_poses_tum.txt'
OUTPUT_TUM_FILENAME = 'geo_tum.txt'
FRAME_METRICS_FILENAME = 'frame_metrics.txt'
SUMMARY_METRICS_FILENAME = 'metrics_summary.txt'
RUNTIME_PROFILE_FILENAME = 'geo_runtime.txt'
RUNTIME_SUMMARY_FILENAME = 'runtime_profile_summary.txt'
MERGED_MAP_FILENAME = 'geo_merged_map.pcd'
RESULT_SUBDIR = 'geo_results'
KEYFRAME_SUBDIR = 'key_point_frame'

# 可选：将 LIO 里程计坐标系统一变换到地图坐标系的初始位姿
# 优先使用 4x4 齐次矩阵 INITIAL_TRANSFORM；若为 None 则退回到 RPY 形式的 INITIAL_POSE。
# 1) 使用矩阵形式（推荐）：把 T_map_lio0 写成 4x4 列表，例如：
#    INITIAL_TRANSFORM = [
#        [1.0, 0.0, 0.0, 10.0],
#        [0.0, 1.0, 0.0,  5.0],
#        [0.0, 0.0, 1.0,  0.0],
#        [0.0, 0.0, 0.0,  1.0],
#    ]
# 2) 使用 RPY 形式：将 INITIAL_TRANSFORM 设为 None，
#    再把 INITIAL_POSE 填成 (tx, ty, tz, roll, pitch, yaw)，角度单位由 POSE_IN_RADIANS 决定。
# INITIAL_TRANSFORM =  [ 0.556912342, -0.830538784, -0.007341485,-223.941216745,  #PK01
#                   0.830297811 ,0.556480475 , 0.030576983 ,-534.004804865,
#                   -0.021309977,-0.023124317 , 0.999505452, -1.854225961,
#                   0.000000000 , 0.000000000, 0.000000000 , 1.000000000 ]

INITIAL_TRANSFORM =  [ -0.519301, 0.850557, 0.082936 ,-11.347226, #stairs
                  -0.852164, -0.522691, 0.024698, 3.002144,
                  0.064357, -0.057849, 0.996249, -0.715776,
                  0.000000, 0.000000, 0.000000, 1.000000 ]

# INITIAL_TRANSFORM = [0.962393,  -0.269109 , 0.0371485 ,   6.26396, #CORRIDOR
#                   0.267793,   0.962772 , 0.0368319  ,0.0850816,
#                   -0.0456773, -0.0254987  , 0.998631  , 0.792745,
#                   0,        0,       0,        1 ]

# building day
# INITIAL_TRANSFORM = [ 0.448165, -0.893951 ,-0.000477, -41.163830,
#                   0.891230, 0.446760 ,0.078197, -46.008873,
#                   -0.069691, -0.035470, 0.996938, 0.261990,
#                   0.000000, 0.000000, 0.000000, 1.000000 ]

# cave02
# INITIAL_TRANSFORM = [ 0.274473488331, 0.961305737495, 0.023577133194,33.809207916260,
#                   -0.961209475994, 0.274974942207, -0.021568179131,-72.459877014160,
#                   -0.027216726914, -0.016742672771, 0.999489367008,21.207557678223,
#                   0,      0,     0,      1 ]

INITIAL_POSE = None
POSE_IN_RADIANS = False


class SinglePairDataset(torch.utils.data.Dataset):
    def __init__(self, ref_points: np.ndarray, src_points: np.ndarray, transform: np.ndarray):
        self.sample = {
            'seq_id': np.int64(0),
            'ref_frame': np.int64(0),
            'src_frame': np.int64(1),
            'ref_points': ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'ref_feats': np.ones((ref_points.shape[0], 1), dtype=np.float32),
            'src_feats': np.ones((src_points.shape[0], 1), dtype=np.float32),
            'transform': transform.astype(np.float32),
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        return self.sample


def load_point_cloud(path: str) -> np.ndarray:
    ext = osp.splitext(path)[1].lower()
    if ext == '.npy':
        points = np.load(path)
    elif ext in ['.pcd', '.ply', '.xyz']:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(path)
        if len(pcd.points) == 0:
            raise ValueError(f'Point cloud {path} is empty.')
        points = np.asarray(pcd.points)
    elif ext == '.bin':
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    else:
        raise ValueError(f'Unsupported point cloud format: {ext}')
    return points.astype(np.float32)


def pose_to_transform(pose, radians: bool = False) -> np.ndarray:
    """将 (tx, ty, tz, r, p, y) 转为 4x4 齐次变换矩阵。

    若 radians=False，则 r/p/y 按角度（度）解释，与 custom_infer 中保持一致。
    """
    if pose is None:
        raise ValueError('INITIAL_POSE 不能为空。')
    translation = np.asarray(pose[:3], dtype=np.float64)
    angles = np.asarray(pose[3:], dtype=np.float64)
    rotation = Rotation.from_euler('xyz', angles, degrees=not radians).as_matrix()
    transform = get_transform_from_rotation_translation(rotation, translation)
    return transform.astype(np.float64)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.shape[0] == 0 or voxel_size <= 0:
        return points
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsampled = pcd.voxel_down_sample(voxel_size)
    return np.asarray(downsampled.points, dtype=np.float32)


def parse_tum_file(path: str) -> List[dict]:
    poses = []
    with open(path, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                logging.warning('Line %d in %s is invalid: %s', line_idx, path, line)
                continue
            timestamp = float(parts[0])
            translation = np.asarray(parts[1:4], dtype=np.float64)
            quat = np.asarray(parts[4:8], dtype=np.float64)
            norm = np.linalg.norm(quat)
            if norm == 0:
                logging.warning('Quaternion on line %d is zero norm, skip.', line_idx)
                continue
            quat = quat / norm
            rotation = Rotation.from_quat(quat).as_matrix()
            transform = get_transform_from_rotation_translation(rotation, translation)
            poses.append({
                'timestamp': timestamp,
                'transform': transform.astype(np.float32),
            })
    return poses


def crop_prior_map(map_points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """在 XY 平面上按半径裁剪局部地图，仿照 PCL CropBox 的思路实现。

    使用初始位姿的平移 (cx, cy) 作为中心，只保留满足
        (x - cx)^2 + (y - cy)^2 <= radius^2
    的点，Z 方向不做限制（等价于 C++ 示例中 Z 维度为 ±inf）。
    全部在 numpy 上完成，不依赖 KDTree。
    """
    if map_points.shape[0] == 0:
        logging.info('crop_prior_map: 输入点云为空')
        return map_points

    cx, cy = float(center[0]), float(center[1])
    r2 = float(radius) * float(radius)
    xy = map_points[:, :2]
    dx = xy[:, 0] - cx
    dy = xy[:, 1] - cy
    dist2 = dx * dx + dy * dy
    mask = dist2 <= r2
    cropped = map_points[mask]
    logging.info('crop_prior_map: 半径=%.2f, 输入点数=%d, 输出点数=%d', radius, map_points.shape[0], cropped.shape[0])
    return cropped


def prepare_batch(ref_points: np.ndarray, src_points: np.ndarray, transform: np.ndarray, cfg, neighbor_limits: List[int]):
    dataset = SinglePairDataset(ref_points, src_points, transform)
    collated = registration_collate_fn_stack_mode(
        [dataset[0]],
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        precompute_data=True,
    )
    return collated


def log_batch_details(batch: dict):
    """记录各层点数与邻接表大小，便于排查内存或精度问题。"""
    stage_msgs = []
    points_list = batch.get('points', [])
    neighbors_list = batch.get('neighbors', [])
    for stage_idx, points in enumerate(points_list):
        num_points = int(points.shape[0])
        neighbor_shape = tuple(neighbors_list[stage_idx].shape) if stage_idx < len(neighbors_list) else ()
        stage_msgs.append(f'S{stage_idx}: points={num_points}, neighbors_shape={neighbor_shape}')
    features_shape = tuple(batch['features'].shape) if 'features' in batch else None
    if stage_msgs:
        logging.info('batch 各阶段统计：features=%s, %s', features_shape, '; '.join(stage_msgs))
    else:
        logging.info('batch 暂无分层统计信息，可检查预处理是否正常生成。')


def has_valid_ground_truth_transform(transform_tensor: torch.Tensor, atol: float = 1e-4) -> bool:
    """判断数据字典里是否真正提供了配准的 GT 变换矩阵。"""
    if transform_tensor is None:
        return False
    if not torch.is_tensor(transform_tensor):
        transform_tensor = torch.as_tensor(transform_tensor)
    candidate = transform_tensor
    if candidate.dim() == 2:
        candidate = candidate.unsqueeze(0)
    if candidate.dim() != 3 or candidate.shape[-2:] != (4, 4):
        # 形状不是标准 4x4，按“提供了其它合法 GT”处理，避免错误判定
        return True
    eye = torch.eye(4, dtype=candidate.dtype, device=candidate.device).unsqueeze(0)
    max_diff = torch.max(torch.abs(candidate - eye))
    return max_diff > atol


def run_single_registration(
    model,
    cfg,
    device,
    ref_points: np.ndarray,
    src_points: np.ndarray,
    neighbor_limits: Optional[List[int]],
    keep_ratio: float,
    sample_threshold: int,
):
    logging.info('进入 run_single_registration: ref_points=%d, src_points=%d, neighbor_limits=%s',
                 ref_points.shape[0], src_points.shape[0],
                 'None' if neighbor_limits is None else str(neighbor_limits))
    if ref_points.shape[0] == 0 or src_points.shape[0] == 0:
        raise ValueError('Empty point cloud encountered during registration.')

    transform_guess = np.eye(4, dtype=np.float32)
    dataset = SinglePairDataset(ref_points, src_points, transform_guess)

    if neighbor_limits is None:
        logging.info('首次帧将按 custom_infer 风格自动校准邻域上限...')
        neighbor_limits = calibrate_neighbors_stack_mode(
            dataset,
            registration_collate_fn_stack_mode,
            cfg.backbone.num_stages,
            cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius,
            keep_ratio=keep_ratio,
            sample_threshold=sample_threshold,
        )
        logging.info('Calibrated neighbor limits: %s', neighbor_limits)

    logging.info('开始构建 batch (prepare_batch)，ref_points=%d, src_points=%d',
                 ref_points.shape[0], src_points.shape[0])
    batch_start = time.perf_counter()
    batch = prepare_batch(ref_points, src_points, transform_guess, cfg, neighbor_limits)
    batch_time_ms = (time.perf_counter() - batch_start) * 1000.0
    logging.info('batch 构建完成，用时 %.2f ms', batch_time_ms)
    log_batch_details(batch)
    if device.type == 'cuda':
        batch = to_cuda(batch)
        logging.info('batch 已转移至 GPU 执行推理')

    start_time = time.perf_counter()
    logging.info('开始执行模型前向推理')
    with torch.no_grad():
        output = model(batch)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    runtime_ms = (time.perf_counter() - start_time) * 1000.0
    logging.info('模型前向推理完成，耗时 %.2f ms', runtime_ms)

    est_transform = output['estimated_transform']
    if est_transform.dim() == 2:
        est_transform = est_transform.unsqueeze(0)
    est_transform = est_transform.squeeze(0).detach().cpu().numpy()

    if has_valid_ground_truth_transform(batch.get('transform')):
        rre, rte = isotropic_transform_error(batch['transform'], output['estimated_transform'])
        rre_value = rre.item()
        rte_value = rte.item()
    else:
        logging.info('当前数据缺少有效 GT 变换，跳过 RRE/RTE 计算。')
        rre_value = float('nan')
        rte_value = float('nan')
    return neighbor_limits, est_transform.astype(np.float64), rre_value, rte_value, runtime_ms, batch_time_ms


def write_tum_file(path: str, results: List[Tuple[float, np.ndarray]], append: bool = False):
    mode = 'a' if append else 'w'
    with open(path, mode) as f:
        for timestamp, transform in results:
            rotation = transform[:3, :3]
            translation = transform[:3, 3]
            quat = Rotation.from_matrix(rotation).as_quat()
            line = [
                f'{timestamp:.9f}',
                f'{translation[0]:.8f}',
                f'{translation[1]:.8f}',
                f'{translation[2]:.8f}',
                f'{quat[0]:.8f}',
                f'{quat[1]:.8f}',
                f'{quat[2]:.8f}',
                f'{quat[3]:.8f}',
            ]
            f.write(' '.join(line) + '\n')


def write_frame_metrics_txt(path: str, logs: List[dict]):
    header = '# index timestamp scan_path runtime_ms rre_deg rte_m correction_matrix(16 values row-major)\n'
    with open(path, 'w') as f:
        f.write(header)
        for log in logs:
            matrix_str = ' '.join(f'{v:.8f}' for v in log['matrix'])
            line = (
                f"{log['index']} "
                f"{log['timestamp']:.9f} "
                f"{log['scan_path']} "
                f"{log['runtime_ms']:.3f} "
                f"{log['rre_deg']:.6f} "
                f"{log['rte_m']:.6f} "
                f"{matrix_str}"
            )
            f.write(line + '\n')


def write_summary_metrics_txt(path: str, logs: List[dict]):
    if not logs:
        return

    def avg(key):
        values = [log[key] for log in logs if np.isfinite(log[key])]
        if not values:
            return float('nan')
        return float(np.mean(values))

    metrics = {
        'frame_count': len(logs),
        'avg_runtime_ms': avg('runtime_ms'),
        'avg_rre_deg': avg('rre_deg'),
        'avg_rte_m': avg('rte_m'),
    }

    with open(path, 'w') as f:
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f'{key}: {value:.6f}\n')
            else:
                f.write(f'{key}: {value}\n')


def init_runtime_profile_txt(path: str):
    header = '# index timestamp ref_points src_points crop_time_ms batch_time_ms registration_time_ms total_time_ms\n'
    with open(path, 'w') as f:
        f.write(header)


def append_runtime_profile_txt(path: str, log: dict):
    line = (
        f"{log['index']} "
        f"{log['timestamp']:.9f} "
        f"{log['ref_points']} "
        f"{log['src_points']} "
        f"{log['crop_time_ms']:.3f} "
        f"{log['batch_time_ms']:.3f} "
        f"{log['registration_time_ms']:.3f} "
        f"{log['total_time_ms']:.3f}\n"
    )
    with open(path, 'a') as f:
        f.write(line)


def write_runtime_summary_txt(path: str, logs: List[dict]):
    keys = ['crop_time_ms', 'batch_time_ms', 'registration_time_ms', 'total_time_ms']
    with open(path, 'w') as f:
        if not logs:
            f.write('# 没有可用的帧耗时统计\n')
            return
        for key in keys:
            values = [log[key] for log in logs]
            avg_value = float(np.mean(values))
            f.write(f'avg_{key}: {avg_value:.6f}\n')


def flush_merged_map(merged_points_list: List[np.ndarray], output_path: str) -> bool:
    if not output_path:
        return False
    if not merged_points_list:
        logging.warning('当前没有可用于融合的点云，无法写出融合地图。')
        return False
    merged_points = np.concatenate(merged_points_list, axis=0)
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_points)
        o3d.io.write_point_cloud(output_path, pcd)
        logging.info('融合地图已保存到 %s，总点数 %d', output_path, merged_points.shape[0])
        return True
    except Exception as err:
        logging.warning('写入融合地图时出错：%s', err)
        return False


def start_control_listener(stop_event: threading.Event, merge_event: threading.Event):
    if not sys.stdin.isatty():
        logging.info('标准输入非交互，按键控制不可用。')
        return None

    def _listener():
        logging.info('控制提示：按一次空格即可请求停止，按 m 可立即输出融合地图。')
        fd = sys.stdin.fileno()
        use_raw = termios is not None and tty is not None
        old_settings = None
        if use_raw:
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        try:
            while not stop_event.is_set():
                if use_raw:
                    ch = sys.stdin.read(1)
                    if not ch:
                        break
                    if ch == ' ':
                        logging.info('收到停止指令，将在当前帧结束后停止处理。')
                        stop_event.set()
                        break
                    lowered = ch.lower()
                    if lowered == 'm':
                        logging.info('收到立即融合指令，将尽快写出融合地图。')
                        merge_event.set()
                else:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    stripped = line.strip().lower()
                    if line.strip() == '' and ' ' in line:
                        logging.info('收到停止指令，将在当前帧结束后停止处理。')
                        stop_event.set()
                        break
                    if stripped in {'space', 'stop'}:
                        logging.info('收到停止指令，将在当前帧结束后停止处理。')
                        stop_event.set()
                        break
                    if stripped == 'm':
                        logging.info('收到立即融合指令，将尽快写出融合地图。')
                        merge_event.set()
        finally:
            if use_raw and old_settings is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    thread = threading.Thread(target=_listener, daemon=True)
    thread.start()
    return thread


def parse_args():
    parser = argparse.ArgumentParser(description='Register sequential scans against a prior map using GeoTransformer.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='包含先验地图、点云帧和 TUM 初值的目录 (默认使用上方常量)')
    parser.add_argument('--scan_dir', type=str, default=None, help='若不设置则默认为 data_dir/key_point_frame')
    parser.add_argument('--map_path', type=str, default=None, help='若不设置则默认为 data_dir + PRIOR_MAP_FILENAME')
    parser.add_argument('--tum_path', type=str, default=None, help='若不设置则默认为 data_dir + TUM_FILENAME')
    parser.add_argument('--output_tum', type=str, default=None, help='若不设置则默认为 data_dir + OUTPUT_TUM_FILENAME')
    parser.add_argument('--snapshot', type=str, default=osp.join(ROOT_DIR, 'weights', 'geotransformer-kitti.pth.tar'), help='GeoTransformer 预训练权重路径')
    parser.add_argument('--scan_extension', type=str, default='.pcd', help='单帧点云的扩展名 (默认 .pcd)')
    parser.add_argument('--scan_voxel_size', type=float, default=0.2, help='单帧体素下采样分辨率 (米)')
    parser.add_argument('--map_voxel_size', type=float, default=0.5, help='先验地图体素下采样分辨率 (米)')
    parser.add_argument('--crop_radius', type=float, default=100.0, help='裁剪先验地图的半径 (米)')
    parser.add_argument('--keep_ratio', type=float, default=0.8, help='邻域校准的保留比例')
    parser.add_argument('--sample_threshold', type=int, default=1, help='邻域校准的最小样本数')
    parser.add_argument('--use_gpu', action='store_true', help='兼容参数：当前脚本在检测到 CUDA 时会默认启用 GPU')
    parser.add_argument('--log_path', type=str, default=None, help='若提供则保存 JSON 日志')
    parser.add_argument('--frame_metrics', type=str, default=None, help='逐帧指标 txt（默认 data_dir/frame_metrics.txt）')
    parser.add_argument('--summary_metrics', type=str, default=None, help='平均指标 txt（默认 data_dir/metrics_summary.txt）')
    parser.add_argument('--runtime_profile', type=str, default=None, help='逐帧耗时统计 txt（默认 data_dir/runtime_profile.txt）')
    parser.add_argument('--runtime_summary', type=str, default=None, help='耗时汇总 txt（默认 data_dir/runtime_profile_summary.txt）')
    parser.add_argument('--merged_map', type=str, default=None, help='融合后的全局地图 pcd 路径（默认 data_dir/geo_merged_map.pcd）')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
    result_dir = osp.join(args.data_dir, RESULT_SUBDIR)

    if args.scan_dir is None:
        args.scan_dir = osp.join(args.data_dir, KEYFRAME_SUBDIR)
    if args.map_path is None:
        args.map_path = osp.join(args.data_dir, PRIOR_MAP_FILENAME)
    if args.tum_path is None:
        args.tum_path = osp.join(args.data_dir, TUM_FILENAME)
    if args.output_tum is None:
        args.output_tum = osp.join(result_dir, OUTPUT_TUM_FILENAME)
    if args.frame_metrics is None:
        args.frame_metrics = osp.join(result_dir, FRAME_METRICS_FILENAME)
    if args.summary_metrics is None:
        args.summary_metrics = osp.join(result_dir, SUMMARY_METRICS_FILENAME)
    if args.runtime_profile is None:
        args.runtime_profile = osp.join(result_dir, RUNTIME_PROFILE_FILENAME)
    if args.runtime_summary is None:
        args.runtime_summary = osp.join(result_dir, RUNTIME_SUMMARY_FILENAME)
    if args.merged_map is None:
        args.merged_map = osp.join(result_dir, MERGED_MAP_FILENAME)

    if not osp.isdir(args.scan_dir):
        raise FileNotFoundError(f'Scan directory not found: {args.scan_dir}')
    if not osp.isfile(args.map_path):
        raise FileNotFoundError(f'Prior map not found: {args.map_path}')
    if not osp.isfile(args.tum_path):
        raise FileNotFoundError(f'TUM pose file not found: {args.tum_path}')
    if not osp.isfile(args.snapshot):
        raise FileNotFoundError(f'Snapshot not found: {args.snapshot}')

    os.makedirs(result_dir, exist_ok=True)
    logging.info('结果文件将输出到子目录：%s', result_dir)

    cfg = make_cfg()

    # 若设置了初始矩阵 / 位姿，则将整体 LIO 轨迹左乘这一初始变换，统一到地图坐标系
    global_init_transform = None
    if INITIAL_TRANSFORM is not None:
        global_init_transform = np.asarray(INITIAL_TRANSFORM, dtype=np.float64).reshape(4, 4)
        logging.info('使用全局初始变换矩阵 (LIO->地图)，来自 INITIAL_TRANSFORM：\n%s', global_init_transform)
    elif INITIAL_POSE is not None:
        global_init_transform = pose_to_transform(INITIAL_POSE, radians=POSE_IN_RADIANS)
        logging.info('使用全局初始位姿矩阵 (LIO->地图)，来自 INITIAL_POSE=%s', str(INITIAL_POSE))

    if torch.cuda.is_available():
        if not args.use_gpu:
            logging.info('检测到 CUDA，默认启用 GPU。如需仅使用 CPU，请手动修改脚本。')
        device = torch.device('cuda')
    else:
        raise RuntimeError('GeoTransformer 当前实现依赖 CUDA，但未检测到可用 GPU。请在支持 CUDA 的环境运行。')

    model = create_model(cfg).to(device)
    state_dict = torch.load(args.snapshot, map_location='cpu')
    if 'model' not in state_dict:
        raise RuntimeError('Snapshot does not contain a "model" key.')
    model.load_state_dict(state_dict['model'], strict=True)
    model.eval()

    logging.info('加载先验地图：%s', args.map_path)
    prior_map = load_point_cloud(args.map_path)
    logging.info('原始先验地图点数：%d', prior_map.shape[0])
    prior_map = voxel_downsample(prior_map, args.map_voxel_size)
    logging.info('体素下采样 (%.3fm) 后先验地图点数：%d', args.map_voxel_size, prior_map.shape[0])
    if prior_map.shape[0] == 0:
        raise RuntimeError('先验地图在体素下采样后为空，请检查数据。')

    poses = parse_tum_file(args.tum_path)
    if not poses:
        raise RuntimeError('未能从 TUM 文件解析到有效轨迹。')

    stop_event = threading.Event()
    merge_event = threading.Event()
    start_control_listener(stop_event, merge_event)

    neighbor_limits = None
    refined_results = []
    per_frame_logs = []
    runtime_profile_logs = []
    merged_points_list = []
    # 清空已有 TUM 结果，以便逐帧追加
    write_tum_file(args.output_tum, [], append=False)
    logging.info('清空/创建输出 TUM 轨迹文件：%s', args.output_tum)
    init_runtime_profile_txt(args.runtime_profile)
    logging.info('清空/创建逐帧耗时文件：%s', args.runtime_profile)

    for idx, pose in enumerate(poses):
        if stop_event.is_set():
            logging.info('检测到停止指令，提前结束后续帧处理。')
            break
        scan_path = osp.join(args.scan_dir, f'{idx}{args.scan_extension}')
        if not osp.isfile(scan_path):
            logging.warning('点云 %s 不存在，跳过。', scan_path)
            continue

        scan_points = load_point_cloud(scan_path)
        logging.info('帧 %d 原始点数：%d', idx, scan_points.shape[0])
        scan_points = voxel_downsample(scan_points, args.scan_voxel_size)
        logging.info('帧 %d 体素下采样 (%.3fm) 后点数：%d', idx, args.scan_voxel_size, scan_points.shape[0])
        if scan_points.shape[0] == 0:
            logging.warning('点云 %s 在下采样后为空，跳过。', scan_path)
            continue

        pose_transform = pose['transform'].astype(np.float64)
        if global_init_transform is not None:
            init_transform = global_init_transform @ pose_transform
        else:
            init_transform = pose_transform
        if idx == 0:
            logging.info('首帧 init_transform:\n%s', init_transform)
        scan_in_map = apply_transform(scan_points.copy(), init_transform)

        center = init_transform[:3, 3]
        logging.info('帧 %d 裁剪先验地图，中心=(%.3f, %.3f, %.3f), 半径=%.2fm', idx, center[0], center[1], center[2], args.crop_radius)
        crop_start = time.perf_counter()
        cropped_map = crop_prior_map(prior_map, center, args.crop_radius)
        crop_time_ms = (time.perf_counter() - crop_start) * 1000.0
        logging.info('帧 %d 裁剪耗时：%.2f ms', idx, crop_time_ms)
        if cropped_map.shape[0] == 0:
            logging.warning('基于初始位姿裁剪先验地图为空，使用完整先验地图。')
            cropped_map = prior_map
        ref_points_count = int(cropped_map.shape[0])
        src_points_count = int(scan_points.shape[0])

        try:
            neighbor_limits, correction, rre, rte, registration_time_ms, batch_time_ms = run_single_registration(
                model,
                cfg,
                device,
                cropped_map,
                scan_in_map,
                neighbor_limits,
                args.keep_ratio,
                args.sample_threshold,
            )
        except ValueError as err:
            logging.warning('注册帧 %d 失败：%s', idx, err)
            continue

        refined_transform = (correction @ init_transform).astype(np.float64)
        refined_results.append((pose['timestamp'], refined_transform))
        write_tum_file(args.output_tum, [(pose['timestamp'], refined_transform)], append=True)
        logging.info('帧 %d 的优化姿态已写入 TUM 文件。', idx)
        if args.merged_map:
            registered_points = apply_transform(scan_points.copy(), refined_transform)
            merged_points_list.append(registered_points.astype(np.float32))
            logging.info('帧 %d 点云已加入融合地图，点数 %d', idx, registered_points.shape[0])
        total_time_ms = crop_time_ms + batch_time_ms + registration_time_ms
        runtime_profile_logs.append({
            'index': idx,
            'timestamp': pose['timestamp'],
            'ref_points': ref_points_count,
            'src_points': src_points_count,
            'crop_time_ms': crop_time_ms,
            'batch_time_ms': batch_time_ms,
            'registration_time_ms': registration_time_ms,
            'total_time_ms': total_time_ms,
        })
        append_runtime_profile_txt(args.runtime_profile, runtime_profile_logs[-1])

        per_frame_logs.append({
            'index': idx,
            'scan_path': scan_path,
            'timestamp': pose['timestamp'],
            'runtime_ms': registration_time_ms,
            'rre_deg': rre,
            'rte_m': rte,
            'matrix': correction.astype(np.float64).reshape(-1).tolist(),
        })

        if np.isfinite(rre) and np.isfinite(rte):
            metric_msg = f'RRE {rre:.4f}°，RTE {rte:.4f} m'
        else:
            metric_msg = 'RRE/RTE 无法计算（缺少 GT）'
        logging.info('帧 %d 完成：%s，Registration %.1f ms，点云 %s', idx, metric_msg, registration_time_ms, scan_path)
        logging.info(
            '帧 %d 耗时统计：裁剪 %.2f ms + batch 构建 %.2f ms + 注册 %.2f ms (不含 tensor 转换) = %.2f ms',
            idx,
            crop_time_ms,
            batch_time_ms,
            registration_time_ms,
            total_time_ms,
        )
        if args.merged_map and merge_event.is_set():
            flush_merged_map(merged_points_list, args.merged_map)
            merge_event.clear()
        if stop_event.is_set():
            logging.info('收到停止指令，在处理完帧 %d 后退出。', idx)
            break

    if not refined_results:
        if stop_event.is_set():
            logging.warning('由于用户请求停止，未成功注册任何帧。')
            return
        raise RuntimeError('未成功注册任何帧。')

    logging.info('已实时写入全部优化轨迹到 %s', args.output_tum)

    write_frame_metrics_txt(args.frame_metrics, per_frame_logs)
    logging.info('逐帧指标已保存到 %s', args.frame_metrics)

    write_summary_metrics_txt(args.summary_metrics, per_frame_logs)
    logging.info('平均指标已保存到 %s', args.summary_metrics)

    write_runtime_summary_txt(args.runtime_summary, runtime_profile_logs)
    logging.info('逐帧耗时统计汇总已保存到 %s', args.runtime_summary)

    if args.merged_map:
        flush_merged_map(merged_points_list, args.merged_map)

    if args.log_path is not None:
        with open(args.log_path, 'w') as f:
            json.dump(per_frame_logs, f, indent=2)
        logging.info('帧级日志已写入 %s', args.log_path)


if __name__ == '__main__':
    main()

import numpy as np
from typing import List, Tuple, Dict

# -------------------- 간단 DBSCAN (Naive, O(N^2)) --------------------
def _dbscan_naive(X: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """
    X: (N,3) 점들, eps: 군집 반경, min_pts: 최소 이웃 수
    return labels: (N,)  -1은 노이즈, 0..K-1은 클러스터 ID
    """
    N = X.shape[0]
    labels = np.full(N, -1, dtype=int)
    if N == 0:
        return labels
    # 거리 행렬 (Naive)
    D2 = np.sum((X[:, None, :] - X[None, :, :])**2, axis=2)
    eps2 = eps * eps
    neighbors = [np.where(D2[i] <= eps2)[0] for i in range(N)]

    visited = np.zeros(N, dtype=bool)
    cid = 0
    for i in range(N):
        if visited[i]:
            continue
        visited[i] = True
        Ni = neighbors[i]
        if Ni.size < min_pts:
            labels[i] = -1  # noise (임시)
            continue
        # 새로운 클러스터 확장
        labels[i] = cid
        seeds = list(Ni.tolist())
        j = 0
        while j < len(seeds):
            p = seeds[j]
            if not visited[p]:
                visited[p] = True
                Np = neighbors[p]
                if Np.size >= min_pts:
                    # core point면 이웃을 병합
                    for q in Np:
                        if q not in seeds:
                            seeds.append(int(q))
            if labels[p] == -1:
                labels[p] = cid
            elif labels[p] < 0:
                labels[p] = cid
            j += 1
        cid += 1
    return labels

# -------------------- 평면 추정/회전 구성 유틸 --------------------
def _fit_plane_svd(pts: np.ndarray):
    p0 = pts.mean(axis=0)
    U, S, Vt = np.linalg.svd(pts - p0, full_matrices=False)
    n = Vt[-1]
    n = n / (np.linalg.norm(n) + 1e-12)
    d = -np.dot(n, p0)
    return n, d, p0

def _point_plane_dist(pts: np.ndarray, n: np.ndarray, d: float):
    return np.abs(pts @ n + d)

def _ransac_planes(points: np.ndarray,
                   max_planes: int,
                   dist_thresh: float,
                   min_points_per_plane: int,
                   iters: int = 400) -> List[Dict]:
    remaining = points.copy()
    planes = []
    if remaining.shape[0] < 3:
        return planes
    for _ in range(max_planes):
        if remaining.shape[0] < max(3, min_points_per_plane):
            break
        best_idx = None
        best_n = None
        best_d = None
        idx_all = np.arange(remaining.shape[0])
        for _ in range(iters):
            idx = np.random.choice(remaining.shape[0], 3, replace=False)
            p = remaining[idx]
            v1, v2 = p[1] - p[0], p[2] - p[0]
            n = np.cross(v1, v2)
            if np.linalg.norm(n) < 1e-9:
                continue
            n = n / np.linalg.norm(n)
            d = -np.dot(n, p[0])
            dists = _point_plane_dist(remaining, n, d)
            inliers = idx_all[dists <= dist_thresh]
            if best_idx is None or inliers.size > best_idx.size:
                best_idx = inliers
                best_n, best_d = n, d
        if best_idx is None or best_idx.size < min_points_per_plane:
            break
        inliers = remaining[best_idx]
        n, d, p0 = _fit_plane_svd(inliers)
        planes.append({"n": n, "d": d, "inliers": inliers, "p0": p0})
        mask = np.ones(remaining.shape[0], dtype=bool)
        mask[best_idx] = False
        remaining = remaining[mask]
    return planes

def _orient_normals_outward(planes: List[Dict], all_mean: np.ndarray):
    for pl in planes:
        n, d = pl["n"], pl["d"]
        side = np.dot(n, all_mean) + d  # >0면 평균점이 n쪽
        if side > 0:
            pl["n"], pl["d"] = -n, -d

def _build_rotation_from_planes(planes: List[Dict]) -> np.ndarray:
    def _norm(v): 
        n = np.linalg.norm(v); 
        return v if n < 1e-12 else v/n

    axes = []
    if len(planes) >= 1:
        axes.append(_norm(planes[0]["n"]))
    if len(planes) >= 2:
        n1 = axes[0]
        # n1과 가장 직교(내적 절댓값 최소)
        cands = [pl["n"] for pl in planes[1:]]
        dots = [abs(np.dot(n1, c)) for c in cands]
        n2 = _norm(cands[int(np.argmin(dots))] - np.dot(cands[int(np.argmin(dots))], n1)*n1)
        axes.append(n2)
    if len(axes) == 1:
        # 평면 내 PCA로 보조축 생성
        pts = planes[0]["inliers"]
        pts_c = pts - pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
        t1, t2 = _norm(Vt[0]), _norm(Vt[1])
        t2 = _norm(t2 - np.dot(t2, t1)*t1)
        n = axes[0]
        return np.stack([t1, t2, n], axis=1)
    # 세 번째 축은 외적
    a1 = _norm(axes[0])
    a2 = _norm(axes[1] - np.dot(axes[1], a1)*a1)
    a3 = _norm(np.cross(a1, a2))
    return np.stack([a1, a2, a3], axis=1)

def _estimate_center_from_planes(planes: List[Dict], cube_size: float, mean_all: np.ndarray) -> np.ndarray:
    s = cube_size * 0.5
    if len(planes) == 0:
        return mean_all.copy()
    if len(planes) == 1:
        face_center = planes[0]["inliers"].mean(axis=0)
        return face_center - s * planes[0]["n"]
    A, b = [], []
    for pl in planes:
        n, d = pl["n"], pl["d"]
        A.append(n); b.append(-d - s)
    A = np.vstack(A)                 # (k,3)
    b = np.array(b).reshape(-1, 1)   # (k,1)
    C, *_ = np.linalg.lstsq(A, b, rcond=None)
    return C.reshape(3)

def _estimate_cube_pose_from_points(points: np.ndarray,
                                    cube_size: float,
                                    ransac_dist_thresh: float,
                                    min_points_per_plane: int,
                                    max_planes: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    mean_all = points.mean(axis=0)
    planes = _ransac_planes(points, max_planes, ransac_dist_thresh, min_points_per_plane)
    _orient_normals_outward(planes, mean_all)
    center = _estimate_center_from_planes(planes, cube_size, mean_all)
    if len(planes) == 0:
        # 아무 면도 못 찾으면 PCA 기반 근사
        pts_c = points - mean_all
        _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
        a1, a2 = Vt[0], Vt[1]
        a1 /= (np.linalg.norm(a1) + 1e-12)
        a2 -= np.dot(a2, a1)*a1
        a2 /= (np.linalg.norm(a2) + 1e-12)
        a3 = np.cross(a1, a2); a3 /= (np.linalg.norm(a3) + 1e-12)
        R = np.stack([a1, a2, a3], axis=1)
    else:
        R = _build_rotation_from_planes(planes)
    info = {"planes": planes, "used_plane_count": len(planes)}
    return center, R, info

# -------------------- 공개 함수: 섞인 점군 → 여러 큐브 포즈 --------------------
def estimate_cubes_from_points(points: np.ndarray,
                               cube_size: float = 1.0,
                               # DBSCAN 파라미터 (스케일 1=10cm 기준)
                               cluster_eps: float = 0.35,      # ~3.5cm
                               cluster_min_pts: int = 30,
                               # 평면 RANSAC 파라미터
                               ransac_dist_thresh: float = 0.02,  # ~2% edge = 2mm(스케일 1=10cm면 0.02=2mm)
                               min_points_per_plane: int = 12,
                               max_planes_per_cluster: int = 3
                               ) -> List[Dict]:
    """
    points: (N,3) — 서로 다른 큐브/배경이 섞여 있을 수 있는 점들
    return: 리스트[{ 'center':(3,), 'R':(3,3), 'indices': np.ndarray, 'info': {...} }, ...]
    """
    assert points.ndim == 2 and points.shape[1] == 3, "points는 (N,3) 이어야 합니다."
    if points.shape[0] == 0:
        return []

    # 1) 공간 군집화 (섞인 점군 → 여러 클러스터)
    labels = _dbscan_naive(points, eps=cluster_eps, min_pts=cluster_min_pts)
    results = []
    uniq = sorted([l for l in np.unique(labels) if l >= 0])
    for lab in uniq:
        idx = np.where(labels == lab)[0]
        if idx.size < max(min_points_per_plane, 6):
            continue
        pts = points[idx]

        # 2) 클러스터별 큐브 포즈 추정
        center, R, info = _estimate_cube_pose_from_points(
            pts,
            cube_size=cube_size,
            ransac_dist_thresh=ransac_dist_thresh,
            min_points_per_plane=min_points_per_plane,
            max_planes=max_planes_per_cluster
        )

        # 간단 품질지표: 사용한 면 수 + 평면 잔차 평균
        residual = None
        if info["used_plane_count"] > 0:
            res = []
            s = cube_size * 0.5
            for pl in info["planes"]:
                # 이상적인 중심이라면 n·C + d ≈ -s
                res.append(abs(np.dot(pl["n"], center) + pl["d"] + s))
            residual = float(np.mean(res))
        results.append({
            "center": center,
            "R": R,
            "indices": idx,   # 이 결과에 쓰인 원본 점들의 인덱스
            "quality": {
                "plane_count": info["used_plane_count"],
                "mean_plane_residual": residual
            },
            "info": info
        })
    return results

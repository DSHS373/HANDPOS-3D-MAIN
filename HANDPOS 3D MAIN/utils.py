import os
import numpy as np

def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P

# ======================
# 기본 DLT (두 카메라 P 행렬 + 대응 픽셀 → 3D)
# ======================
def DLT(P1, P2, point1, point2):
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    B = A.transpose() @ A
    # scipy.linalg 사용
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)
    return Vh[3,0:3]/Vh[3,3]

# ======================
# 파일 I/O
# ======================
def read_camera_parameters(camera_id):
    inf = open('camera_parameters/camera'+ str(camera_id) + '_intrinsics.dat', 'r')

    cmtx = []
    dist = []

    line = inf.readline()  # header
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)

    line = inf.readline()  # "distortion:" header
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)

    return np.array(cmtx), np.array(dist)

def read_rotation_translation(camera_id, savefolder = 'camera_parameters/'):
    inf = open(savefolder + 'camera'+ str(camera_id) + '_rot_trans.dat', 'r')

    inf.readline()
    rot = []
    trans = []
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        rot.append(line)

    inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        trans.append(line)

    inf.close()
    return np.array(rot), np.array(trans)

def _convert_to_homogeneous(pts):
    pts = np.array(pts)
    if len(pts.shape) > 1:
        w = np.ones((pts.shape[0], 1))
        return np.concatenate([pts, w], axis = 1)
    else:
        return np.concatenate([pts, [1]], axis = 0)

def get_projection_matrix(camera_id):
    # read camera parameters
    cmtx, dist = read_camera_parameters(camera_id)
    rvec, tvec = read_rotation_translation(camera_id)
    # calculate projection matrix
    P = cmtx @ _make_homogeneous_rep_matrix(rvec, tvec)[:3,:]
    return P

# ======================
# 새로 추가/보강된 함수들
# ======================
def read_world_to_camera_rot_trans(camera_id, savefolder='camera_parameters/', prefix='world_to_'):
    """
    파일 형태:
    R:
    r11 r12 r13
    r21 r22 r23
    r31 r32 r33
    T:
    tx
    ty
    tz
    """
    path = os.path.join(savefolder, f'{prefix}camera{camera_id}_rot_trans.dat')
    with open(path, 'r') as f:
        # 'R:' 라인까지 이동
        line = f.readline()
        while line and not line.strip().startswith('R'):
            line = f.readline()
        rot = []
        for _ in range(3):
            vals = [float(x) for x in f.readline().split()]
            rot.append(vals)

        # 'T:' 라인까지 이동
        line = f.readline()
        while line and not line.strip().startswith('T'):
            line = f.readline()

        trans_vals = []
        # 일반적으로 한 줄에 한 값 씩 3줄
        for _ in range(3):
            vals = [float(x) for x in f.readline().split()]
            if len(vals) == 1:
                trans_vals.append(vals[0])
            else:
                # 혹시 "tx ty tz" 한 줄일 수 있어 방어
                trans_vals.extend(vals)
                break

    R = np.array(rot, dtype=float)
    T = np.array(trans_vals[:3], dtype=float).reshape(3)
    return R, T

def shift_world_origin(R_wc, T_wc, new_origin_in_old_world, new_world_rotation=None):
    """
    기존 world->camera: x_c = R_wc x_w + T_wc
    새 월드 w'를 기존 월드에서 p 만큼 평행이동(및 선택 회전 S)하면
    x_w = S x_{w'} + p (S=None이면 단위행렬)
    => x_c = R_wc S x_{w'} + (R_wc p + T_wc)
    평행이동만: S = I
    """
    p = np.asarray(new_origin_in_old_world, dtype=float).reshape(3)
    S = np.eye(3) if new_world_rotation is None else np.asarray(new_world_rotation, dtype=float).reshape(3, 3)

    R_new = R_wc @ S
    T_new = (R_wc @ p) + T_wc
    return R_new, T_new

def get_projection_matrix_with_shift(camera_id, new_origin_in_old_world, 
                                     use_world_to_files=True,
                                     savefolder='camera_parameters/', prefix='world_to_'):
    """
    - camera_id: 0, 1, ...
    - new_origin_in_old_world: 기존 월드 좌표계에서의 새 원점 [x, y, z]
    - use_world_to_files=True 이면 world_to_cameraX_rot_trans 파일을 읽음
      False 이면 기존 cameraX_rot_trans.dat 사용
    반환: 3x4 P = K [R|T]
    """
    cmtx, dist = read_camera_parameters(camera_id)

    if use_world_to_files:
        R_wc, T_wc = read_world_to_camera_rot_trans(camera_id, savefolder=savefolder, prefix=prefix)
    else:
        R_wc, T_wc = read_rotation_translation(camera_id, savefolder=savefolder)
        T_wc = T_wc.reshape(3)

    R_new, T_new = shift_world_origin(R_wc, T_wc, new_origin_in_old_world)
    P = cmtx @ _make_homogeneous_rep_matrix(R_new, T_new)[:3, :]
    return P

def save_extrinsic_rot_trans(R, T, camera_id, savefolder='camera_parameters/', prefix='worldshift_to_'):
    """
    계산된 (world'->camera) R, T를 파일로 저장 (예: worldshift_to_camera0_rot_trans.dat)
    """
    os.makedirs(savefolder, exist_ok=True)
    path = os.path.join(savefolder, f'{prefix}camera{camera_id}_rot_trans.dat')  # ← 확장자 보강
    with open(path, 'w') as f:
        f.write("R:\n")
        for i in range(3):
            f.write(f"{R[i,0]} {R[i,1]} {R[i,2]}\n")
        f.write("T:\n")
        f.write(f"{T[0]}\n{T[1]}\n{T[2]}\n")

def write_keypoints_to_disk(filename, kpts):
    with open(filename, 'w') as fout:
        for frame_kpts in kpts:
            for kpt in frame_kpts:
                if len(kpt) == 2:
                    fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ')
                else:
                    fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ' + str(kpt[2]) + ' ')
            fout.write('\n')

if __name__ == '__main__':
    # 간단 테스트(파일 존재 가정)
    P0 = get_projection_matrix_with_shift(0, [0.0, 0.0, 0.0], use_world_to_files=True)
    P1 = get_projection_matrix_with_shift(1, [0.0, 0.0, 0.0], use_world_to_files=True)
    print("P0:\n", P0)
    print("P1:\n", P1)

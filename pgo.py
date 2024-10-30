import symforce

symforce.set_epsilon_to_number()

import sym
import symforce.symbolic as sf
from symforce import codegen
from symforce.codegen import codegen_util

import scipy.sparse as sp
import sksparse.cholmod as cholmod

import numpy as np
from scipy.spatial.transform import Rotation

np.set_printoptions(linewidth=np.inf, suppress=True, precision=4)


def quat_to_rotvec(qx, qy, qz, qw):
    rotation = Rotation.from_quat([qx, qy, qz, qw])
    return rotation.as_rotvec()


def rotvec_to_quat(rotvec):
    rotation = Rotation.from_rotvec(rotvec)
    q = rotation.as_quat()
    return q


def rotmat_to_rotvec(R):
    rotation = Rotation.from_matrix(R)
    rotvec = rotation.as_rotvec()
    return rotvec


def rotvec_to_rotmat(rotvec):
    rotation = Rotation.from_rotvec(rotvec)
    R = rotation.as_matrix()
    return R


def read_g2o_files(file_path):
    points = {}
    edges = []

    for line in open(file_path, "r"):
        data = line.strip().split()
        if not data:
            continue
        tag = data[0]
        if tag == "VERTEX_SE3:QUAT":
            node_id = int(data[1])
            x, y, z = map(float, data[2:5])
            t = np.array([x, y, z])
            qx, qy, qz, qw = map(float, data[5:9])
            r = quat_to_rotvec(qx, qy, qz, qw)
            points[node_id] = {"t": t, "r": r}

        if tag == "EDGE_SE3:QUAT":
            id_from = int(data[1])
            id_to = int(data[2])
            x, y, z = map(float, data[3:6])
            t = np.array([x, y, z])
            qx, qy, qz, qw = map(float, data[6:10])
            r = quat_to_rotvec(qx, qy, qz, qw)
            info = np.diag([1.0, 1.0, 1.0, 100.0, 100.0, 100.0])
            edges.append({"from": id_from, "to": id_to, "r": r, "t": t, "info": info})

    return points, edges


def sf_between_error(Ti: sf.Pose3, Tj: sf.Pose3, Tij: sf.Pose3):
    return Tij.inverse() * (Ti.inverse() * Tj)


def sf_codegen():
    between_error_codegen = codegen.Codegen.function(
        func=sf_between_error,
        config=codegen.PythonConfig(),
    )

    between_error_codegen_with_jacobians = between_error_codegen.with_jacobians(
        which_args=["Ti", "Tj"],
        include_results=True,
    )

    namespace = "nano_pgo"
    between_error_codegen_with_jacobians_data = (
        between_error_codegen_with_jacobians.generate_function(namespace)
    )

    return codegen_util.load_generated_package(
        namespace, between_error_codegen_with_jacobians_data.function_dir
    )


gen_module = sf_codegen()


def between_factor_jacobian_by_symforce(ti, ri, tj, rj, tij, rij):
    residual, res_D_Ti, res_D_Tj = gen_module.sf_between_error_with_jacobians01(
        Ti=sym.Pose3(R=sym.rot3.Rot3(rotvec_to_quat(ri)), t=ti),
        Tj=sym.Pose3(R=sym.rot3.Rot3(rotvec_to_quat(rj)), t=tj),
        Tij=sym.Pose3(R=sym.rot3.Rot3(rotvec_to_quat(rij)), t=tij),
    )

    # because symforce uses the order of [r, t]
    sf_Ji = np.zeros((6, 6))
    sf_Ji[:3, :3] = res_D_Ti[3:, 3:]
    sf_Ji[3:, :3] = res_D_Ti[:3, 3:]
    sf_Ji[:3, 3:] = res_D_Ti[3:, :3]
    sf_Ji[3:, 3:] = res_D_Ti[:3, :3]

    sf_Jj = np.zeros((6, 6))
    sf_Jj[:3, :3] = res_D_Tj[3:, 3:]
    sf_Jj[3:, :3] = res_D_Tj[:3, 3:]
    sf_Jj[:3, 3:] = res_D_Tj[3:, :3]
    sf_Jj[3:, 3:] = res_D_Tj[:3, :3]

    residual = residual.to_tangent()
    r_err, t_err = residual[:3].copy(), residual[3:].copy()
    residual[:3], residual[3:] = t_err, r_err

    return residual, sf_Ji, sf_Jj


class Optimizer:

    def __init__(self):
        self.cauchy_c = 10
        self.lambda_ = 0.001
        self.lambda_allowed_range = [1e-7, 1e5]
        self.max_iters = 15

    def add_initial(self, points, edges):
        self.poses_initial = points
        self.num_poses = len(points)
        self.pose_indices = list(points.keys())
        self.index_map = {pose_id: idx for idx, pose_id in enumerate(points)}
        self.edges = edges

    def optimize(self):
        self.relax_rotation()

        self.x = np.zeros(6 * self.num_poses)
        for pose_id, pose in self.poses_initial.items():
            idx = self.index_map[pose_id]
            self.x[6 * idx : 6 * idx + 3] = pose["t"]
            self.x[6 * idx + 3 : 6 * idx + 6] = pose["r"]

        for _ in range(self.max_iters):
            delta_x, error_before_opt = self.step()

            x_new = self.x + delta_x

            error_after_opt = self.eval(x_new)

            self.x = x_new if error_after_opt < error_before_opt else self.x

            # update parameters
            if error_after_opt < error_before_opt:
                if self.lambda_allowed_range[0] < self.lambda_:
                    self.lambda_ /= 10.0
            else:
                if self.lambda_ < self.lambda_allowed_range[1]:
                    self.lambda_ *= 10.0
                min_cauchy_c = 1.0
                if self.cauchy_c / 2.0 > min_cauchy_c:
                    self.cauchy_c /= 2.0

            print("err", error_before_opt, "->", error_after_opt)

        self.vis()

    def get_state_block(self, states_vector, block_idx):
        return states_vector[6 * block_idx : 6 * (block_idx + 1)]

    def cauchy_weight(self, s):
        epsilon = 1e-5
        return self.cauchy_c / (np.sqrt(self.cauchy_c**2 + s) + epsilon)

    def process_edge(self, edge, x):
        idx_i = self.index_map[edge["from"]]
        idx_j = self.index_map[edge["to"]]

        xi = self.get_state_block(x, idx_i)
        xj = self.get_state_block(x, idx_j)

        ti, ri = xi[:3], xi[3:]
        tj, rj = xj[:3], xj[3:]
        tij, rij, info = edge["t"], edge["r"], edge["info"]

        residual, Ji, Jj = between_factor_jacobian_by_symforce(ti, ri, tj, rj, tij, rij)

        if not (abs(edge["from"] - edge["to"]) == 1):
            weight = self.cauchy_weight(residual.T @ info @ residual)
        else:
            weight = 1.0

        residual *= weight
        Ji *= weight
        Jj *= weight

        return residual, Ji, Jj

    def eval(self, x_new):
        total_error = 0.0
        for edge in self.edges:
            info = edge["info"]
            residual, _, _ = self.process_edge(edge, x_new)
            total_error += residual.T @ info @ residual
        return total_error

    def step(self):
        # build sprase system
        H_row = []
        H_col = []
        H_data = []
        b = np.zeros(6 * len(self.index_map))
        total_error = 0.0

        for _, edge in enumerate(self.edges):
            idx_i = self.index_map[edge["from"]]
            idx_j = self.index_map[edge["to"]]
            info = edge["info"]

            residual, Ji, Jj = self.process_edge(edge, self.x)

            total_error += residual.T @ info @ residual

            Hii = Ji.T @ info @ Ji
            Hjj = Jj.T @ info @ Jj
            Hij = Ji.T @ info @ Jj
            bi = Ji.T @ info @ residual
            bj = Jj.T @ info @ residual

            for i in range(6):
                for j in range(6):
                    # Hii
                    H_row.append((6 * idx_i) + i)
                    H_col.append((6 * idx_i) + j)
                    H_data.append(Hii[i, j])

                    # Hjj
                    H_row.append(6 * idx_j + i)
                    H_col.append(6 * idx_j + j)
                    H_data.append(Hjj[i, j])

                    # Hij
                    H_row.append(6 * idx_i + i)
                    H_col.append(6 * idx_j + j)
                    H_data.append(Hij[i, j])

                    # Hji
                    H_row.append(6 * idx_j + i)
                    H_col.append(6 * idx_i + j)
                    H_data.append(Hij[j, i])

            # bi and bj
            b[(6 * idx_i) : (6 * idx_i) + 6] -= bi
            b[(6 * idx_j) : (6 * idx_j) + 6] -= bj

        # add prior to prevent gauge freedom
        if True:
            residual_prior = np.zeros(6)
            J_prior = -np.identity(6)
            info_prior = 1e-2 * np.identity(6)

            H_prior = J_prior.T @ info_prior @ J_prior
            b_prior = J_prior.T @ info_prior @ residual_prior

            prior_pose_id = self.pose_indices[0]
            idx_prior = self.index_map[prior_pose_id]

            for i in range(6):
                for j in range(6):
                    H_row.append(6 * idx_prior + i)
                    H_col.append(6 * idx_prior + j)
                    H_data.append(H_prior[i, j])

            b[(6 * idx_prior) : (6 * idx_prior) + 6] -= b_prior

        H = sp.csc_matrix(
            (H_data, (H_row, H_col)),
            shape=(6 * self.num_poses, 6 * self.num_poses),
        )

        # Solves H * delta_x = b using the Cholesky factorization with damping (Levenberg-Marquardt).
        H += sp.diags(self.lambda_ * H.diagonal(), format="csc")
        delta_x = cholmod.cholesky(H).solve_A(b)

        return delta_x, total_error

    def relax_rotation(self):
        # Chordal relaxation, Section III.B of https://dellaert.github.io/files/Carlone15icra1.pdf

        prev_dx = None
        num_epochs = 3
        for epoch in range(num_epochs):
            H_row = []
            H_col = []
            H_data = []
            b = np.zeros(9 * self.num_poses)
            info = 1.0 * np.identity(3)

            for edge in self.edges:
                from_id, to_id = edge["from"], edge["to"]

                Ri = rotvec_to_rotmat(self.poses_initial[from_id]["r"])
                Rj = rotvec_to_rotmat(self.poses_initial[to_id]["r"])
                Rij = rotvec_to_rotmat(edge["r"])

                for row_i in range(3):
                    residual = Rij.T @ Ri[row_i, :] - Rj[row_i, :]  # eq 21

                    if not (abs(from_id - to_id) == 1):
                        weight = self.cauchy_weight(residual.T @ info @ residual)
                    else:
                        weight = 1.0

                    residual *= weight
                    Ji = Rij.T * weight
                    Jj = -np.identity(3) * weight

                    H_ii = Ji.T @ info @ Ji
                    H_jj = Jj.T @ info @ Jj
                    H_ij = Ji.T @ info @ Jj
                    bi = Ji.T @ info @ residual
                    bj = Jj.T @ info @ residual

                    from_var_idx = 9 * self.index_map[from_id] + 3 * row_i
                    to_var_idx = 9 * self.index_map[to_id] + 3 * row_i

                    for i in range(3):
                        for j in range(3):
                            H_row.append(from_var_idx + i)
                            H_col.append(from_var_idx + j)
                            H_data.append(H_ii[i, j])

                            H_row.append(to_var_idx + i)
                            H_col.append(to_var_idx + j)
                            H_data.append(H_jj[i, j])

                            H_row.append(from_var_idx + i)
                            H_col.append(to_var_idx + j)
                            H_data.append(H_ij[i, j])

                            H_row.append(to_var_idx + i)
                            H_col.append(from_var_idx + j)
                            H_data.append(H_ij[j, i])

                    b[from_var_idx : from_var_idx + 3] -= bi
                    b[to_var_idx : to_var_idx + 3] -= bj

            # add prior
            prior_pose_id = self.pose_indices[0]
            idx_prior = self.index_map[prior_pose_id]

            R_est = np.identity(3)
            R_meas = rotvec_to_rotmat(self.poses_initial[idx_prior]["r"])
            residual_prior = (R_est - R_meas).flatten()
            J_prior = np.identity(9)
            info_prior = 1e-2 * np.identity(9)

            H_prior = J_prior.T @ info_prior @ J_prior
            b_prior = J_prior.T @ info_prior @ residual_prior

            for i in range(9):
                for j in range(9):
                    H_row.append(9 * idx_prior + i)
                    H_col.append(9 * idx_prior + j)
                    H_data.append(H_prior[i, j])

            b[(9 * idx_prior) : (9 * idx_prior) + 9] -= b_prior

            # solve
            H = sp.csc_matrix(
                (H_data, (H_row, H_col)),
                shape=(9 * self.num_poses, 9 * self.num_poses),
            )
            delta_x = cholmod.cholesky(H).solve_A(b)

            delta_x_norm = np.linalg.norm(delta_x)
            if prev_dx is None:
                dx_gain = delta_x_norm
            else:
                dx_gain = np.linalg.norm(delta_x - prev_dx)
            prev_dx = delta_x
            print(f"[relax] epoch {epoch} |delta_x| {delta_x_norm} dx_gain {dx_gain}")

            # update rot in pose
            for pose_id, pose in self.poses_initial.items():
                # eq 22
                M = rotvec_to_rotmat(pose["r"])
                var_start_idx = 9 * self.index_map[pose_id]
                M += delta_x[var_start_idx : var_start_idx + 9].reshape(3, 3)

                # eq 23
                U, _, Vt = np.linalg.svd(M)
                det_sign = np.sign(np.linalg.det(U @ Vt))
                S = np.diag([1, 1, det_sign])
                R_star = U @ S @ Vt

                self.poses_initial[pose_id]["r"] = rotmat_to_rotvec(R_star)

    def vis(self):
        import open3d as o3d

        points = [self.x[6 * i : 6 * i + 3] for i in range(0, self.num_poses)]
        point_set = o3d.geometry.PointCloud()
        point_set.points = o3d.utility.Vector3dVector(points)
        colors = np.tile([0, 1, 0], (len(points), 1))
        point_set.colors = o3d.utility.Vector3dVector(colors)

        lines = [
            [self.index_map[edge["from"]], self.index_map[edge["to"]]]
            for edge in self.edges
        ]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_colors = np.tile([0, 0, 1], (len(lines), 1))
        line_set.colors = o3d.utility.Vector3dVector(line_colors)

        o3d.visualization.draw_geometries(
            [point_set, line_set],
            zoom=0.8,
            front=[0, 0, 1],
            lookat=points[len(points) // 2],
            up=[0, 1, 0],
        )


if __name__ == "__main__":

    dataset_dir = "data"
    dataset_name = f"{dataset_dir}/cubicle.g2o"

    opt = Optimizer()
    points, edges = read_g2o_files(dataset_name)
    opt.add_initial(points, edges)
    opt.optimize()

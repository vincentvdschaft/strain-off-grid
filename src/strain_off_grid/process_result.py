import h5py
import numpy as np
from imagelib import Image
from zea.tracking import BlockMatchingTracker

from strain_off_grid import DynamicPhantom, load_dataclass


def compute_results(path):
    im_das = Image.load(path).abs().normalize().log_compress().clip(-60, 0).to_pixels()

    index0 = 1

    phantom: DynamicPhantom = load_dataclass(path, group="/custom/phantom")

    def discard_diverged_mask(velocities, threshold=1e-2):
        norms = np.linalg.norm(velocities, axis=-1)
        return norms < threshold

    def discard_large_internal_dot_product(positions: np.ndarray, threshold=1e-2):
        """
        Args:
            positions: (n_points, n_frames, 2) array of positions
            threshold: threshold for dot product
        Returns:
            mask: (n_points,) boolean array indicating which points to keep
        """
        velocities = np.diff(positions, axis=1)
        norms = np.linalg.norm(velocities, axis=-1, keepdims=True)
        normalized_velocities = velocities / (norms + 1e-8)
        dot_products = np.min(
            np.einsum(
                "ijk,ijk->ij",
                normalized_velocities[:, :-1],
                normalized_velocities[:, 1:],
            ),
            axis=1,
        )
        return dot_products < threshold

    def get_baseline_tracking_velocities(
        images: Image, center_positions: np.ndarray, dt: float
    ):
        center_index = images.shape[-1] // 2

        tracker = BlockMatchingTracker(extent=images[..., 0].T.extent)

        current_positions = center_positions.copy()[:, np.array([0, 2], dtype=int)]
        velocities = []
        for n in range(center_index + 1, images.shape[-1] - 1):
            block_matching_points = tracker.track(
                prev_frame=images.array[..., n],
                next_frame=images.array[..., n - 1],
                points=current_positions,
            )
            velocities.append(-(block_matching_points - current_positions) / dt)
            current_positions = block_matching_points

        current_positions = center_positions.copy()[:, np.array([0, 2], dtype=int)]
        for n in range(center_index - 1, -1, -1):
            block_matching_points = tracker.track(
                prev_frame=images.array[..., n + 1],
                next_frame=images.array[..., n],
                points=current_positions,
            )
            velocities.append(-(block_matching_points - current_positions) / dt)
            current_positions = block_matching_points

        return np.mean(velocities, axis=0)

    def discard_outliers_mask(velocities, threshold=1e-2):
        mean_velocity = np.mean(velocities, axis=1, keepdims=True)
        diffs = velocities - mean_velocity
        distances = np.linalg.norm(diffs, axis=-1)
        mean_distance = np.mean(distances, axis=1)
        return mean_distance < threshold

    def get_average_velocity_vectors(positions: np.ndarray, dt: float):
        velocities = []
        for i in range(positions.shape[1] - 1):
            delta = positions[:, i + 1] - positions[:, i]
            velocity = delta / dt
            velocities.append(velocity)
        return np.mean(velocities, axis=0)

    def load_positions_and_velocities(path):
        with h5py.File(path, "r") as f:
            keys = list(f["positions"].keys())
            keys.sort()
            positions = np.stack([f["positions"][key][:] for key in keys], axis=1)

            intensities = np.stack(
                [f["intensities"][f"{i:03d}"][:] for i in range(len(keys))]
            )
            intensities = np.mean(intensities, axis=0)
            mask = intensities / np.max(intensities) > 0.1

            timestamps = f["timestamps"][:].ravel()
            dt = timestamps[1] - timestamps[0]
            velocities = get_average_velocity_vectors(positions, dt)
            deltas = velocities * dt

            mask_diverged = discard_diverged_mask(velocities, threshold=10.0e-2)
            print(f"Discarded {np.sum(~mask_diverged)} diverged points")
            mask = np.logical_and(mask, mask_diverged)

            try:
                mask_large_dot = discard_large_internal_dot_product(
                    positions, threshold=0.1
                )
                print(
                    f"Discarded {np.sum(~mask_large_dot)} points with large internal dot product"
                )
                # mask = np.logical_and(mask, mask_large_dot)
            except ValueError as e:
                print(f"Could not compute internal dot product: {e}")

        sort_indices = np.argsort(intensities)[::-1]
        positions = positions[sort_indices]
        deltas = deltas[sort_indices]
        n_points = positions.shape[0]
        middle_position_index = positions.shape[1] // 2

        return (
            positions[mask][:, middle_position_index],
            deltas[mask] / dt,
            timestamps,
            positions[mask],
        )

    def get_ground_truth_velocities(
        positions, index0: int, timestamps: np.ndarray
    ) -> np.ndarray:
        velocities = phantom.get_velocities(
            positions,
            timestamps[index0],
            dt=dt,
        )
        return velocities[:, np.array([0, 2], dtype=int)]

    def compute_errors(estimated_velocities, ground_truth_velocities):
        if estimated_velocities.shape[-1] == 3:
            estimated_velocities = estimated_velocities[:, [0, 2]]
        if ground_truth_velocities.shape[-1] == 3:
            ground_truth_velocities = ground_truth_velocities[:, [0, 2]]
        diffs = estimated_velocities - ground_truth_velocities
        distances = np.linalg.norm(diffs, axis=-1)
        return distances

    def normalize_velocities(velocities):
        norms = np.linalg.norm(velocities, axis=-1, keepdims=True)
        return velocities / (norms + 1e-8)

    positions, velocities_solver, timestamps, track_positions = (
        load_positions_and_velocities(path)
    )
    dt = timestamps[1] - timestamps[0]
    velocities_ground_truth = get_ground_truth_velocities(positions, index0, timestamps)
    velocities_baseline = get_baseline_tracking_velocities(im_das, positions, dt)

    return {
        "positions": positions,
        "velocities_solver": velocities_solver,
        "velocities_ground_truth": velocities_ground_truth,
        "velocities_baseline": velocities_baseline,
        "timestamps": timestamps,
        "im_das": im_das,
        "solver_errors": compute_errors(velocities_solver, velocities_ground_truth),
        "baseline_errors": compute_errors(velocities_baseline, velocities_ground_truth),
        "track_positions": track_positions,
    }


def compute_results_multiple(paths) -> dict:
    results_list = []
    for path in paths:
        results = compute_results(path)
        results_list.append(results)
    return _combine_results_dicts(results_list)


def _combine_results_dicts(results_list):
    keys_to_concatenate = [
        "positions",
        "velocities_solver",
        "velocities_ground_truth",
        "velocities_baseline",
        "timestamps",
        "solver_errors",
        "baseline_errors",
        "track_positions",
    ]
    combined_results = {}
    for key in keys_to_concatenate:
        combined_results[key] = np.concatenate(
            [results[key] for results in results_list], axis=0
        )
    combined_results["im_das"] = results_list[0]["im_das"]
    return combined_results

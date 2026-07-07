import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from strain_off_grid.phantoms import ChainedBoxesPhantom


def main():
    bpm = 80.0
    bps = bpm / 60.0
    period = 1 / bps
    box_endpoints_t0 = np.array(
        [
            [[0.10598309, -0.01506975], [0.11597899, -0.0338389]],
            [[0.09368607, -0.02118231], [0.10030204, -0.04110207]],
            [[0.08016653, -0.02161378], [0.08038227, -0.04246832]],
            [[0.06786951, -0.02053509], [0.06679082, -0.04268415]],
            [[0.05636134, -0.01959596], [0.05240616, -0.04145733]],
            [[0.04370476, -0.01894875], [0.03378084, -0.03440992]],
            [[0.03579439, -0.01240472], [0.02565473, -0.02024318]],
            [[0.03442805, -0.00571687], [0.02428836, -0.00564495]],
            [[0.03687308, -0.00025152], [0.02644575, 0.00967239]],
            [[0.04276989, 0.00715546], [0.03464377, 0.02125027]],
            [[0.05571413, 0.01528156], [0.05096788, 0.03023939]],
            [[0.0671482, 0.01693554], [0.0654223, 0.03311582]],
            [[0.08009244, 0.01751084], [0.07800698, 0.03441025]],
            [[0.0920299, 0.01765467], [0.0931805, 0.03512943]],
            [[0.10161788, 0.01610364], [0.10347573, 0.03358971]],
        ],
        dtype=np.float32,
    )[:, :, [1, 0]]

    # box_endpoints are of shape (n_points, 2, 2), which means (n_segments, 2 endpoints, 2 coordinates). The first endpoint of each segment is the inner endpoint, and the second endpoint is the outer endpoint.
    # The boxes share one edge with the next box, so the second inner endpoint of box zero is the first inner endpoint of box one, and so on.
    n_points = box_endpoints_t0.shape[0]
    index_anchor = n_points // 2

    def compute_box_lengths(box_endpoints: np.ndarray) -> np.ndarray:
        """Computes the lengths of the boxes along the segments.

        Returns:
            np.ndarray: An array of shape (n_segments-1, inner_outer(2))"""
        return np.linalg.norm(np.diff(box_endpoints, axis=0), axis=2)

    def compute_box_widths(box_endpoints: np.ndarray) -> np.ndarray:
        """Computes the widths of the boxes along the segments.

        Returns:
            np.ndarray: An array of shape (n_segments, inner_outer(2))"""
        return np.linalg.norm(np.diff(box_endpoints, axis=1), axis=2).squeeze(-1)

    inner, outer = 0, 1

    def unit_vector(vector: np.ndarray) -> np.ndarray:
        """Returns the vector scaled to unit length."""
        return vector / np.linalg.norm(vector, axis=-1, keepdims=True)

    def vertex_index(segment: int, side: int) -> int:
        """Flattens a (segment, inner/outer) pair into a single vertex index."""
        return segment * 2 + side

    def width_edges(lengths: np.ndarray, widths: np.ndarray) -> list:
        """Returns the inner-to-outer edge of each segment as (start, end, vector)."""
        edges = []
        for segment in range(n_points):
            direction = unit_vector(
                box_endpoints_t0[segment, outer] - box_endpoints_t0[segment, inner]
            )
            edges.append(
                (
                    vertex_index(segment, inner),
                    vertex_index(segment, outer),
                    direction * widths[segment],
                )
            )
        return edges

    def length_edges(lengths: np.ndarray, widths: np.ndarray) -> list:
        """Returns the segment-to-segment edge along the inner side as (start, end, vector).

        The outer side is left unconstrained; its positions follow from the width edges."""
        edges = []
        for box in range(n_points - 1):
            direction = unit_vector(
                box_endpoints_t0[box + 1, inner] - box_endpoints_t0[box, inner]
            )
            edges.append(
                (
                    vertex_index(box, inner),
                    vertex_index(box + 1, inner),
                    direction * lengths[box],
                )
            )
        return edges

    def incidence_matrix(edges: list) -> np.ndarray:
        """Builds the edge-vertex incidence matrix (one row per edge)."""
        matrix = np.zeros((len(edges), n_points * 2))
        for row, (start, end, _) in enumerate(edges):
            matrix[row, start] = -1
            matrix[row, end] = 1
        return matrix

    def fit_vertices(
        edges: list, anchor_index: int, anchor_position: np.ndarray
    ) -> np.ndarray:
        """Least-squares fits vertex positions to the desired edge vectors, fixing the anchor."""
        incidence = incidence_matrix(edges)
        anchor_row = np.zeros((1, incidence.shape[1]))
        anchor_row[0, anchor_index] = 1
        matrix = np.vstack([incidence, anchor_row])
        targets = np.array([vector for _, _, vector in edges])
        coordinates = []
        for axis in range(2):
            right_hand_side = np.concatenate(
                [targets[:, axis], [anchor_position[axis]]]
            )
            solution, *_ = np.linalg.lstsq(matrix, right_hand_side, rcond=None)
            coordinates.append(solution)
        return np.stack(coordinates, axis=1)

    def solve_for_lengths_and_widths(
        lengths: np.ndarray, widths: np.ndarray
    ) -> np.ndarray:
        """Computes the new box endpoints for the given lengths and widths along the segments."""
        edges = width_edges(lengths, widths) + length_edges(lengths, widths)
        anchor_index = vertex_index(index_anchor, outer)
        anchor_position = box_endpoints_t0[index_anchor, outer]
        vertices = fit_vertices(edges, anchor_index, anchor_position)
        return vertices.reshape(n_points, 2, 2)

    relative_length_amplitudes = np.ones((n_points - 1,)) * 0.2
    relative_width_amplitudes = np.ones((n_points,)) * 0.1

    relative_length_amplitudes[2] = 0.0
    relative_length_amplitudes[3] = 0.0
    relative_width_amplitudes[2] *= 0.2
    relative_width_amplitudes[3] *= 0.2
    # relative_width_amplitudes[0] = 0.0
    # relative_width_amplitudes[-1] = 0.0

    def length_and_widths_at_time(t: float) -> tuple[np.ndarray, np.ndarray]:
        """Computes the inner lengths and widths of the boxes at time t.

        Returns:
            lengths: np.ndarray of shape (n_segments-1,)
            widths: np.ndarray of shape (n_segments,)
        """
        lengths = compute_box_lengths(box_endpoints_t0)[:, inner]
        widths = compute_box_widths(box_endpoints_t0)

        lengths = lengths * (
            1 + relative_length_amplitudes * np.sin(2 * np.pi / period * t)
        )
        widths = widths * (
            1 + relative_width_amplitudes * -np.sin(2 * np.pi / period * t)
        )
        return lengths, widths

    def box_endpoints_at_time(t: float) -> np.ndarray:
        """Computes the box endpoints at time t from the time-varying lengths and widths."""
        lengths, widths = length_and_widths_at_time(t)
        return solve_for_lengths_and_widths(lengths, widths)

    def plot_boxes(box_endpoints: np.ndarray):
        def _plot_box(coords: np.ndarray):
            """Coords of shape (2, 2, 2)"""
            plt.plot(
                [
                    coords[0, 0, 0],
                    coords[0, 1, 0],
                    coords[1, 1, 0],
                    coords[1, 0, 0],
                    coords[0, 0, 0],
                ],
                [
                    coords[0, 0, 1],
                    coords[0, 1, 1],
                    coords[1, 1, 1],
                    coords[1, 0, 1],
                    coords[0, 0, 1],
                ],
                "r-",
            )

        plt.figure(figsize=(6, 6))
        for i in range(box_endpoints.shape[0] - 1):
            _plot_box(box_endpoints[i : i + 2, :, :])

        plt.plot(
            box_endpoints[index_anchor, 1, 0], box_endpoints[index_anchor, 1, 1], "k+"
        )
        plt.show()

    def box_outline(box_endpoints: np.ndarray, box_index: int) -> tuple[list, list]:
        """Returns the closed x and y outlines of a single box."""
        corners = box_endpoints[box_index : box_index + 2, :, :]
        order = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]
        xs = [corners[segment, side, 0] for segment, side in order]
        ys = [corners[segment, side, 1] for segment, side in order]
        return xs, ys

    def animate_boxes(n_frames: int = 60):
        """Animates the boxes deforming over one cardiac cycle."""
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.set_aspect("equal")
        lines = [axis.plot([], [], "r-")[0] for _ in range(n_points - 1)]

        def update(frame: int):
            box_endpoints = box_endpoints_at_time(frame / n_frames)
            for box_index, line in enumerate(lines):
                line.set_data(*box_outline(box_endpoints, box_index))
            return lines

        set_animation_limits(axis)
        animation = FuncAnimation(
            figure, update, frames=n_frames, interval=50, blit=True
        )
        plt.show()
        return animation

    def set_animation_limits(axis):
        """Sets axis limits to fit the boxes across the whole cycle."""
        all_endpoints = np.array(
            [box_endpoints_at_time(t) for t in np.linspace(0, 1, 20)]
        )
        axis.set_xlim(all_endpoints[..., 0].min(), all_endpoints[..., 0].max())
        axis.set_ylim(all_endpoints[..., 1].min(), all_endpoints[..., 1].max())

    def box_endpoints_to_polygon(box_endpoints: np.ndarray):
        """Converts the box endpoints to a polygon representation.

        Args:
            box_endpoints: An array of shape (n_segments, 2, 2) representing the endpoints of each box.

        Returns:
            polygon: An array of shape (n_vertices, 2) representing the vertices of the polygon.
        """
        n_segments = box_endpoints.shape[0]
        polygon = []
        for i in range(n_segments):
            inner_point = box_endpoints[i, 0]
            polygon.append(inner_point)
        for i in reversed(range(n_segments)):
            outer_point = box_endpoints[i, 1]
            polygon.append(outer_point)
        return np.array(polygon)

    bpm = 60.0
    bps = bpm / 60.0
    period = 1 / bps
    times = np.linspace(0, period, 50)

    points = np.stack([box_endpoints_at_time(t) for t in times], axis=0)
    print(points.shape)
    ChainedBoxesPhantom(
        box_endpoints=np.stack([box_endpoints_at_time(t) for t in times], axis=0),
        keyframe_times=times,
        thickness=0.0,
        period=period,
    ).to_hdf5("out/miccai_cardiac_phantom.hdf5").sample_points(1000)

    # animation = animate_boxes()

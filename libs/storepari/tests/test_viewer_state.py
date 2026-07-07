"""
Manual smoke test: build a ViewerState with one of every layer type and
non-default settings, save it to an HDF5 file, reload it from disk, and
run it in a live napari viewer.

Run directly with:
    python tests/test_viewer_state.py
"""

from pathlib import Path

import numpy as np

from storepari import (
    AxesSettings,
    CameraSettings,
    DimsSettings,
    GridSettings,
    NapariImage,
    NapariPoints,
    NapariTracks,
    NapariVectors,
    ViewerSettings,
    ViewerState,
)

OUTPUT_PATH = Path(__file__).parent / "viewer_state.hdf5"

N_T, N_Z, N_Y, N_X = 5, 8, 64, 64


def build_viewer_state() -> ViewerState:
    rng = np.random.default_rng(0)

    image = NapariImage(
        name="raw",
        data=rng.random((N_T, N_Z, N_Y, N_X), dtype=np.float32),
        colormap="magma",
        gamma=1.2,
        opacity=0.9,
        blending="additive",
        contrast_limits=(0.0, 1.0),
    )

    n_points = 50
    points = NapariPoints(
        name="detections",
        data=rng.random((n_points, 4)) * [N_T, N_Z, N_Y, N_X],
        size=6,
        face_color="cyan",
        border_color="black",
        symbol="disc",
        opacity=0.8,
    )

    n_tracks, track_len = 3, 5
    track_rows = [
        [track_id, t, *(rng.random(3) * [N_Z, N_Y, N_X])]
        for track_id in range(n_tracks)
        for t in range(track_len)
    ]
    tracks = NapariTracks(
        name="trajectories",
        data=np.array(track_rows),
        tail_length=10,
        tail_width=3,
        colormap="viridis",
    )

    n_vectors = 20
    starts = rng.random((n_vectors, 4)) * [N_T, N_Z, N_Y, N_X]
    projections = rng.normal(size=(n_vectors, 4))
    vectors = NapariVectors(
        name="flow",
        data=np.stack([starts, projections], axis=1),
        edge_color="yellow",
        edge_width=0.5,
        length=3,
    )

    settings = ViewerSettings(
        dims=DimsSettings(ndisplay=2),
        axes=AxesSettings(visible=True, colored=True, labels=True),
        grid=GridSettings(enabled=True, stride=1),
        camera=CameraSettings(zoom=2.0, mouse_pan=True),
    )

    return ViewerState(
        layers=[image, points, tracks, vectors],
        settings=settings,
    )


def main() -> None:
    state = build_viewer_state()

    OUTPUT_PATH.unlink(missing_ok=True)
    state.to_hdf5(str(OUTPUT_PATH))
    print(f"Saved viewer state to {OUTPUT_PATH}")

    loaded_state = ViewerState.load(str(OUTPUT_PATH))
    print("Reloaded viewer state from disk, launching viewer...")
    loaded_state.run()


if __name__ == "__main__":
    main()

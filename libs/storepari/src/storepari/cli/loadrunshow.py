import argparse
import subprocess

from storepari import ViewerState, console


def main():
    parser = argparse.ArgumentParser(
        description="Run a napari viewer with a saved state."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to the saved viewer state file in hdf5 format. The file may be on a remote system.",
    )
    args = parser.parse_args()

    path = args.path

    def _is_remote_path(path: str) -> bool:
        """Check if the given path is a remote path (with format host:/path/to/file)."""
        return ":" in path and not path.startswith("/")

    def _download(remote_path, local_path="viewer_state.hdf5"):
        with console.status("Downloading file... (you may need to authenticate)"):
            subprocess.run(["scp", remote_path, local_path], check=True)

    if _is_remote_path(path):
        _download(path)
        path = "viewer_state.hdf5"

    viewer_state = ViewerState.load(path).run()

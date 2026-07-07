from pathlib import Path

from trspecfit.utils import discover_xy_files, group_xy_files, load_xy_project


def write_xy(path: Path, values: list[tuple[float, float]]) -> None:
    lines = [f"{x:.1f} {y:.1f}" for x, y in values]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_xy_project_groups_by_core_level_and_depth(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    write_xy(data_dir / "O1s_depth_1.xy", [(0.0, 1.0), (1.0, 2.0)])
    write_xy(data_dir / "O1s_depth_2.xy", [(0.0, 3.0), (1.0, 4.0)])
    write_xy(data_dir / "C1s_depth_1.xy", [(0.0, 5.0), (1.0, 6.0)])
    write_xy(data_dir / "C1s_depth_2.xy", [(0.0, 7.0), (1.0, 8.0)])

    pattern = r"^(?P<core_level>.+?)_(?P<depth>.+)\.xy$"
    paths = discover_xy_files(data_dir, metadata_pattern=pattern)
    assert len(paths) == 4

    grouped = group_xy_files(paths, metadata_pattern=pattern)
    assert set(grouped) == {
        ("O1s", "depth_1"),
        ("O1s", "depth_2"),
        ("C1s", "depth_1"),
        ("C1s", "depth_2"),
    }

    project, files = load_xy_project(
        data_dir,
        project_path=tmp_path / "project",
        project_name="test",
        metadata_pattern=pattern,
    )

    assert project.name == "test"
    assert set(files) == set(grouped)
    assert files[("O1s", "depth_1")].data.shape == (2,)

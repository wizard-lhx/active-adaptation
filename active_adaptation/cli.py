import argparse
import subprocess
import warnings
from pathlib import Path

from .project_loading.discovery import discover_projects
from .project_loading.manifest import PROJECTS_FILE, load_projects, save_projects


def aa_pull():
    """
    Runs `git pull` for active-adaptation and all projects discovered and listed in `projects.json`.

    If `--all` is passed, it will pull all projects, including inactive ones.
    """
    parser = argparse.ArgumentParser(description="Update all projects")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Pull all projects, including inactive ones",
    )
    args = parser.parse_args()

    if args.all:
        print("Pulling all projects, including inactive ones")
    else:
        print("Pulling active projects only")

    projects = load_projects()
    project_paths = {Path(__file__).resolve().parents[1]}

    for category in ("environment", "learning"):
        for project_info in projects.get(category, {}).values():
            if args.all or project_info["enabled"]:
                project_paths.add(Path(project_info["path"]))

    for i, project_path in enumerate(project_paths):
        print(f"[{i + 1}/{len(project_paths)}] Pulling {project_path}")
        subprocess.run(["git", "branch"], cwd=project_path)
        result = subprocess.run(["git", "pull"], cwd=project_path)
        if result.returncode != 0:
            warnings.warn(
                f"Failed to pull {project_path} with result: {result.returncode}"
            )
            print(result.stderr)


def _task_dir_for_path(project_path: Path) -> Path | None:
    """Return cfg/task directory for a project path, or None if not found."""
    for candidate in (project_path, project_path.parent, project_path.parent.parent):
        task_dir = candidate / "cfg" / "task"
        if task_dir.is_dir():
            return task_dir
    return None


def aa_discover_projects(enabled: bool = False):
    projects = discover_projects(enabled=enabled)

    for project_info in projects.get("environment", {}).values():
        task_dir = _task_dir_for_path(Path(project_info["path"]))
        project_info["task_dir"] = str(task_dir) if task_dir is not None else None

    save_projects(projects)
    print(f"Modify {PROJECTS_FILE} to enable/disable projects.")
    return projects


def aa_list_tasks():
    """
    List task names from YAML files under cfg/task in active-adaptation and in
    all projects from projects.json. Task names preserve the directory prefix
    (e.g. "G1/G1LocoFlat" instead of "G1LocoFlat").
    """
    repo_root = Path(__file__).resolve().parents[1]
    task_dirs: list[tuple[str, Path]] = []
    main_task_dir = repo_root / "cfg" / "task"
    if main_task_dir.is_dir():
        task_dirs.append(("active-adaptation", main_task_dir))

    projects = load_projects()
    for project_name, project_info in projects.get("environment", {}).items():
        task_dir_str = project_info.get("task_dir")
        task_dir = Path(task_dir_str) if task_dir_str else _task_dir_for_path(
            Path(project_info["path"])
        )
        if task_dir is None or not task_dir.is_dir():
            continue
        if any(existing_task_dir == task_dir for _, existing_task_dir in task_dirs):
            continue
        task_dirs.append((project_name, task_dir))

    for source_name, task_dir in task_dirs:
        for yaml_path in sorted(task_dir.rglob("*.yaml")):
            rel = yaml_path.relative_to(task_dir)
            task_id = str(rel.with_suffix("")).replace("\\", "/")
            print(f"  {task_id}  (from {source_name})")

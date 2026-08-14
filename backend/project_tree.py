import pathlib
import sys
import argparse

# ── Danh sách mặc định các thư mục bị bỏ qua ──
DEFAULT_IGNORE = {
    ".git", "__pycache__", "node_modules",
    ".mypy_cache", ".pytest_cache", ".idea", ".vscode",
    ".ipynb_checkpoints", "dist", "build", "egg-info", ".tox",
    "htmlcov", ".coverage"
}

def generate_tree(directory: pathlib.Path, prefix: str = "",
                  ignore_dirs: set = None, max_depth: int = None,
                  current_depth: int = 0) -> str:
    """Tạo chuỗi hiển thị cây thư mục với khả năng bỏ qua thư mục."""
    if max_depth is not None and current_depth >= max_depth:
        return ""

    if ignore_dirs is None:
        ignore_dirs = set()

    lines = []
    entries = sorted(
        [e for e in directory.iterdir()
         if not e.name.startswith(".") and e.name != "__pycache__"
         and e.name not in ignore_dirs],
        key=lambda x: (not x.is_dir(), x.name.lower())
    )

    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        lines.append(prefix + connector + entry.name)
        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            subtree = generate_tree(entry, prefix + extension,
                                    ignore_dirs, max_depth, current_depth + 1)
            if subtree:
                lines.append(subtree)
    return "\n".join(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hiển thị cây thư mục")
    parser.add_argument("directory", nargs="?", default=".",
                        help="Thư mục gốc (mặc định: hiện tại)")
    parser.add_argument("--ignore", nargs="*", default=[],
                        help="Thư mục cần bỏ qua thêm (ngoài danh sách mặc định)")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Độ sâu tối đa (bỏ qua nếu không chỉ định)")
    args = parser.parse_args()

    root = pathlib.Path(args.directory).resolve()
    ignore_set = DEFAULT_IGNORE.union(set(args.ignore))

    print(root.name)
    print(generate_tree(root, ignore_dirs=ignore_set, max_depth=args.max_depth))
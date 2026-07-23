# project_tree.py
import pathlib
import sys

def generate_tree(directory: pathlib.Path, prefix: str = "") -> str:
    """Đệ quy tạo chuỗi hiển thị cây thư mục."""
    lines = []
    entries = sorted(
        [e for e in directory.iterdir() if not e.name.startswith(".") and e.name != "__pycache__"],
        key=lambda x: (not x.is_dir(), x.name.lower())
    )
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        lines.append(prefix + connector + entry.name)
        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            lines.append(generate_tree(entry, prefix + extension))
    return "\n".join(lines)

if __name__ == "__main__":
    root = pathlib.Path.cwd()
    print(root.name)
    print(generate_tree(root))
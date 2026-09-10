"""一次性运行全部单元测试。

用法：
    python run_tests.py            # 全部
    python run_tests.py -v         # 详细输出

说明：tests/ 未加 __init__.py（保持命名空间包），因此 `unittest discover`
在部分 Python 版本下会报 "Start directory is not importable"，
这里显式枚举测试模块以规避该问题。
"""

import os
import sys
import unittest

TESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")


def discover_modules() -> list[str]:
    modules = []
    for entry in sorted(os.listdir(TESTS_DIR)):
        if entry.startswith("test_") and entry.endswith(".py"):
            modules.append(f"tests.{entry[:-3]}")
    return modules


def main() -> int:
    modules = discover_modules()
    if not modules:
        print("未找到任何测试模块", file=sys.stderr)
        return 1

    print(f"发现 {len(modules)} 个测试模块：")
    for name in modules:
        print(f"  - {name}")
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite(loader.loadTestsFromName(name) for name in modules)
    verbosity = 2 if "-v" in sys.argv else 1
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())

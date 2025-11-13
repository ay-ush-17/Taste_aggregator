import sys
import platform
import importlib
import importlib.metadata
import subprocess
import traceback
import os

def print_header(title):
    print('\n' + '='*10 + f' {title} ' + '='*10)

def get_version_by_metadata(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None

def try_import(name, attr_version_name=None):
    try:
        mod = importlib.import_module(name)
        ver = None
        if attr_version_name:
            ver = getattr(mod, attr_version_name, None)
        if not ver:
            ver = getattr(mod, '__version__', None)
        if not ver:
            ver = get_version_by_metadata(name)
        return True, ver or 'unknown'
    except Exception as e:
        return False, str(e)

if __name__ == '__main__':
    print_header('Python & Platform')
    print('python:', sys.version.replace('\n',' '))
    print('executable:', sys.executable)
    print('platform:', platform.platform())
    print('cwd:', os.getcwd())

    print_header('Pip freeze (top 100 chars)')
    try:
        out = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=False)
        lines = out.stdout.splitlines()
        for line in lines[:200]:
            print(line)
    except Exception:
        print('Failed to run pip freeze:')
        traceback.print_exc()

    print_header('Module import checks')
    packages = [
        ('feedparser', None),
        ('requests', None),
        ('bs4', None),
        ('transformers', None),
        ('sentence_transformers', None),
        ('sklearn', '__version__'),
        ('pandas', None),
        ('numpy', None),
        ('flask', None),
        ('torch', None),
    ]

    for pkg, ver_attr in packages:
        ok, info = try_import(pkg, ver_attr)
        status = 'OK' if ok else 'MISSING/ERROR'
        print(f"{pkg:20} {status:15} {info}")

    print_header('Quick environment hints')
    # Check for common issues
    # 1) Python 3.13 known compatibility check (some libraries may lag)
    py_version = sys.version_info
    if py_version.major == 3 and py_version.minor >= 13:
        print('Note: Python 3.13 detected. Some libraries may not yet publish wheels for 3.13; expect possible compatibility issues.')

    # 2) Virtualenv path check
    venv_cfg = os.path.join(os.path.dirname(sys.executable), '..', 'pyvenv.cfg')
    venv_cfg = os.path.abspath(venv_cfg)
    print('Expected pyvenv.cfg location (derived):', venv_cfg)

    print('\nDone.')

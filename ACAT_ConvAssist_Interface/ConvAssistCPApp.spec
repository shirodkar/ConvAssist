# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(5000)
from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_data_files

datas = [('Assets', '.')]
datas += [('Assets/icon_tray.png', 'Assets')]
datas += [('Assets/button_back.png', 'Assets')]
datas += [('Assets/button_clear.png', 'Assets')]
datas += [('Assets/button_exit.png', 'Assets')]
datas += [('Assets/button_license.png', 'Assets')]
datas += [('Assets/frame.png', 'Assets')]
datas += [('scipy.libs', '.')]
datas += [('scipy.libs/libopenblas-802f9ed1179cb9c9b03d67ff79f48187.dll', 'scipy.libs')]
datas += copy_metadata('tqdm')
datas += copy_metadata('torch')
datas += copy_metadata('regex')
datas += copy_metadata('filelock')
datas += copy_metadata('packaging')
datas += copy_metadata('requests')
datas += copy_metadata('numpy')
datas += copy_metadata('tokenizers')
datas += copy_metadata('transformers')
datas += copy_metadata('huggingface-hub')
datas += copy_metadata('pyyaml')
datas += collect_data_files("en_core_web_sm")


block_cipher = None


a = Analysis(
    ['ConvAssistCPApp.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['en_core_web_sm', 'huggingface_hub.hf_api', 'huggingface_hub.repository', 'torch', 'tqdm', 'scipy.datasets', 'scipy.fftpack', 'scipy.misc', 'scipy.odr', 'scipy.signal', 'sklearn.utils._typedefs', 'sklearn.metrics._pairwise_distances_reduction._datasets_pair','sklearn.metrics._pairwise_distances_reduction._middle_term_computer', 'sklearn.utils._heap', 'sklearn.utils._sorting','sklearn.utils._vector_sentinel'],
    
    #hiddenimports=['en_core_web_sm','huggingface_hub.hf_api', 'huggingface_hub.repository', 'torch', 'tqdm', 'scipy.datasets', 'scipy.fftpack', 'scipy.misc', 'scipy.odr', 'scipy.signal', 'sklearn.utils._typedefs', 'sklearn.metrics._pairwise_distances_reduction._datasets_pair','sklearn.metrics._pairwise_distances_reduction._middle_term_computer', 'sklearn.utils._heap', 'sklearn.utils._sorting','sklearn.utils._vector_sentinel'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
for d in a.datas:
    if '_C.cp310-win_amd64.pyd' in d[0]:
        a.datas.remove(d)
        break
for d in a.datas:
    if '_C_flatbuffer.cp310-win_amd64.pyd' in d[0]:
        a.datas.remove(d)
        break

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
	[('W ignore', None, 'OPTION')],
    name='ConvAssistApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon_tray.ico',
)

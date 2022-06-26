# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['hpe.py'],
    pathex=[],
    binaries=[],
    datas=[
        ( 'yolov4_cfg', 'yolov4_cfg' ),
        ( 'simple_baseline_cfg', 'simple_baseline_cfg' ),
        ( 'data', 'data' ),
        ( 'models', 'models' ),
        ( 'output', 'output' ),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Avoid warning
for d in a.datas:
    if '_C.cp37-win_amd64.pyd' in d[0]:
        a.datas.remove(d)
        break

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

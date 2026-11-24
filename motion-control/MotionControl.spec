# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec: explicitly bundle MediaPipe's graph/model resources.

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

mp_datas, mp_binaries, mp_hiddenimports = collect_all("mediapipe")
# Explicitly collect MediaPipe package resources (binary graphs, models, configs).
mp_datas += collect_data_files("mediapipe", include_py_files=True)
mp_binaries += collect_dynamic_libs("mediapipe")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=mp_binaries,
    datas=mp_datas,
    hiddenimports=[
        "customtkinter",
        "PIL._tkinter_finder",
        *mp_hiddenimports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="MotionControl",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

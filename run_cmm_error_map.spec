# -*- mode: python ; coding: utf-8 -*-


config_data = [('src\cmm_error_map\config\*.toml', 'src\cmm_error_map\config'),
               ('src\cmm_error_map\config\gui_configs\*.pkl', 'src\cmm_error_map\config\gui_configs'),
               ('static','static'), ]

a = Analysis(
    ['run_cmm_error_map.py'],
    pathex=[],
    binaries=[],
    datas=config_data,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

splash = Splash(
    'static\splash.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=None,
    text_size=12,
    minify_script=True,
    always_on_top=True,
)

exe = EXE(
    pyz,
    a.scripts,
    splash,
    splash.binaries,
    [],
    exclude_binaries=True,
    name='cmm_error_map',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon = "static\icon.ico",
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='run_cmm_error_map',
)
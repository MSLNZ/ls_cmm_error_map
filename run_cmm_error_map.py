import os
import sys


from cmm_error_map.ls_cmm_error_map import main
from cmm_error_map import __version__

import tkinter

print("Loading CMM Error Map GUI...")
print(f"version {__version__}")
# try:
#     import pyi_splash

#     pyi_splash.update_text("UI Loaded ...")
#     pyi_splash.close()
# except:
#     pass

print("Loading complete")
main()

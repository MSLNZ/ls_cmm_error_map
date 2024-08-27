# cmm-error-map

An app to allow the user to adjust the 21 components of a CMM error map and visualise the effect each component or combination of components has on the overall measurement volume and on  measurements of specific artefacts (ball-plate and gauge block set).

This will help with using the ball-plate and gauge blocks to assess the CMM uncertainty.It could also  be used as a teaching tool.


## Progress

- 2024-08-05 - `modelled_mmts_XYZ_with_sliders.py` old matplotlib version running on up to date versions of python (3.11) and dependencies
- 2024-08-07 - `plot_3d.py` pyqtgraph 3d plot of CMM deformation
- 2024-08-08 - `ls_cmm_error_map.py` beginnings of pyqt gui
- 2024-08-12 - basic 3d deformation plot with sliders working
- 2024-08-13 - further gui work, move plot parameters to each plot dock etc.
- 2024-08-14 - basic pyqtgraph plot of ballplate error working `plot_ballplate.py` - needs styling
- 2024-08-19 - moving ballplate  plot to main app
- 2024-08-20 - ballplate plot responding to model sliders
- 2024-08-21 - class `Plot2Dock`
- 2024-08-22 - choice of matrix transform library - choose mathutils from Blender as well maintained
- 2024-08-26 - working on multiple aretefact plots per axis
- 2024-08-27 - position and orientaion of artefact connected

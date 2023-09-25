


# BUG 1
### Symptom: cv2.imshow(...) freeze
### Cause: python packages av and opencv-python clash (imshow freezes if av is installed)
### Solution: pip uninstall av


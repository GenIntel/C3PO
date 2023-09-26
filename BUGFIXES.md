


# BUG 1
### Symptom
    cv2.imshow(...) freeze
### Cause
    python packages av and opencv-python clash (imshow freezes if av is installed)
### Solution
    pip uninstall av


# BUG 2
### Symptom
    RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the ‘spawn’ start method   
### Cause
    Sharing GPU memory across multiprocesses   
### Solution
    Don't use GPU with dataloader
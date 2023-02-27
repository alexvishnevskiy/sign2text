# sign2text
**All of the following steps were done on M1 silicon**

Assuming that you have activated virtual environment in python3.7: 


`source env/bin/activate`


## Local Host installation

### Mediapipe

- Python 3.7-3.10 (`brew install python@3.7`)
- PIP 19.0 or higher (>20.3 for macOS)

### Mediapipe
- `pip install mediapipe-silicon`

### Jupyter Notebook
- `pip3 install jupyter`

## Anaconda Installation
- `CONDA_SUBDIR=osx-64 conda create -n Env37 python=3.7`
- `conda activate Env37`
- `pip install mediapipe`

----------------
### todo:
- [ ] Resolve `_INFO: Created TensorFlow Lite XNNPACK delegate for CPU.` in Jupyter notebook


- [ ] Resolve `Process finished with exit code 132 (interrupted by signal 4: SIGILL)` on the local host


- [x] Try to start mediapipe in Anaconda
- - [ ] `Error: Process finished with exit code 132 (interrupted by signal 4: SIGILL)`

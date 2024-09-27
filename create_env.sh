#!/usr/bin/env python
# coding: utf-8

# to get the right reqs file: pipreqs ./spo

# global vars
VENVS_DIR="PATH_TO_VENV"
VENV_NAME="neuromancer"
PYTHON_VER="3.9"
GRB_VER="10.0.3"
LOGDIR="./logs"

# load module
echo "Load module..."
module purge
module load cuda
module load python/$PYTHON_VER
module load gurobi/$GRB_VER
module load scip
module load ipopt
module load rust
# check if the license is set
gurobi_cl 1> /dev/null && echo Success || echo Fail
echo ""

# create virtual env
if [ ! -d "./$VENVS_DIR/$VENV_NAME" ]; then
  echo "Create venv..."
  # create source
  virtualenv --no-download $VENVS_DIR/$VENV_NAME
  source $VENVS_DIR/$VENV_NAME/bin/activate
  echo ""

  echo "Install requirements..."

  echo "  Install NeuroMANCER..."
  pip install neuromancer

  # install gurobipy
  echo "  Install GurobiPy..."
  cp -r $GUROBI_HOME/ .
  cd $GRB_VER
  python setup.py install
  cd ..
  rm -r $GRB_VER

  # pip install
  echo "  Install requirements..."
  pip install --no-index --upgrade pip
  pip install tqdm
  pip install numpy
  pip install pandas
  pip install Pyomo
  pip install --no-index torch torchvision torchtext torchaudio

  echo "  Install submitit..."
  pip install submitit

# activate virtual env
else
  echo "Activate venv..."
  source $VENVS_DIR/$VENV_NAME/bin/activate

fi
echo ""

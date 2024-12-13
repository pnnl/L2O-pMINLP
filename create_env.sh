#!/usr/bin/env python
# coding: utf-8

# to get the right reqs file: pipreqs ./spo

# global vars
VENVS_DIR="PATH_TO_VENV"
VENV_NAME="solver"
PYTHON_VER="3.10.13"
CUDA_VER="12.2"
SCIP_VER="9.0.0"
IPOPT_VER="3.14.14"
GRB_VER="11.0.1"
TORCH_VER="2.5.0"
NM_VER="1.5.2"
HSL_ARCHIVE_PATH="./coinhsl.tar.gz"
LOGDIR="./logs"

# load module
echo "Load module..."
module purge
module load StdEnv/2023
module load gcc/12.3
module load python/$PYTHON_VER
module load ipopt/$IPOPT_VER
module load scipoptsuite/$SCIP_VER
module load gurobi/$GRB_VER
modude load cuda/$CUDA_VER
gurobi_cl 1> /dev/null && echo Success || echo Fail

# create virtual env
if [ ! -d "./$VENVS_DIR/$VENV_NAME" ]; then
  echo "Create venv..."
  # create source
  virtualenv --no-download $VENVS_DIR/$VENV_NAME
  source $VENVS_DIR/$VENV_NAME/bin/activate
  echo ""

  echo "Install requirements..."
  pip install --no-index --upgrade pip
  pip install tqdm
  pip install numpy==1.22.4+computecanada
  pip install oldest-supported-numpy
  pip install pandas
  pip install Pyomo
  pip install --no-index torch==$TORCH_VER torchvision torchtext torchaudio
  pip install scipy matplotlib scikit-learn networkx==3.0
  pip install dill pydot==1.4.2 pyts numba plum-dispatch==1.7.3
  pip install graphviz torchsde
  pip install mlflow==2.5.0 --ignore-installed
  pip install opentelemetry-api opentelemetry-sdk sqlparse cachetools
  pip install cvxpy cvxopt casadi cvxpylayers
  pip install torchdiffeq toml
  pip install lightning wandb
  pip install neuromancer==$NM_VER --no-deps
  pip install submitit

  # Setup Coin-HSL
  echo "Setting up Coin-HSL..."
  git clone https://github.com/coin-or-tools/ThirdParty-HSL.git

  echo "Unpacking Coin-HSL archive..."
  tar -xzf "$HSL_ARCHIVE_PATH"
  mv coinhsl-archive-2023.11.17 ThirdParty-HSL/coinhsl

  echo "Configuring and compiling Coin-HSL..."
  cd ThirdParty-HSL || exit
  ./configure --prefix=$HOME/local
  make
  make install

  # Add the lib path to LD_LIBRARY_PATH
  echo "Adding Coin-HSL to LD_LIBRARY_PATH..."
  ln -s $HOME/local/lib/libcoinhsl.so $HOME/local/lib/libhsl.so
  export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
  echo 'export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
  source ~/.bashrc
  cd ..

# activate virtual env
else
  echo "Activate venv..."
  source $VENVS_DIR/$VENV_NAME/bin/activate

fi
echo ""

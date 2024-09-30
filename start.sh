# Run from filesys

# If we're running from a new location
if [ ! -f "plasma-converter" ]; then
    git clone https://github.com/ohm-tree/plasma-converter.git --recurse-submodules
fi
cd plasma-converter
git checkout pretrain

# Install micromamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

./bin/micromamba shell init -s bash -r ~/micromamba
source ~/.bashrc


micromamba env create -n ohm-tree
micromamba activate ohm-tree
pip install -r requirements.txt
pip install -U flash-attn --no-build-isolation
pip install -e .

curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | \
    sh -s -- --default-toolchain leanprover/lean4:stable --no-modify-path -y

# Update bashrc with micromamba and elan paths
cp ./.bashrc ~/.bashrc

cd mathlib4
lake build

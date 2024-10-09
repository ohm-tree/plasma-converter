SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# print where this script is. This should be the location of the
# external file system, which will not always have a consistent name.
echo $SCRIPT_DIR

# download lean 4
bash -c 'curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain leanprover/lean4:stable'

# add /home/ubuntu/.elan/bin to PATH
export PATH="$PATH:/home/ubuntu/.elan/bin"

cd $SCRIPT_DIR

# access private repos
export GITHUB_TOKEN="<ADD YOUR TOKEN HERE>"

# check if plasma-converter directory exists.
# if not, git clone it.
if [ ! -d "plasma-converter" ]; then
  git clone --recurse-submodules https://$GITHUB_TOKEN:x-oauth-basic@github.com/ohm-tree/plasma-converter.git
fi

cd $SCRIPT_DIR
cd plasma-converter
pip install -r plasma-converter/requirements.txt
pip install -U flash-attn --no-build-isolation
pip install -e .

cd mathlib4
lake build


# export PATH="$PATH:/home/ubuntu/.elan/bin"
# Add this line to ~/.bashrc
echo "export PATH=\"\$PATH:/home/ubuntu/.elan/bin\"" >> ~/.bashrc

# add this line to ~/.bashrc
echo $'alias nvkill=\"lsof /dev/nvidia* | awk \'{print \$2}\' | xargs -I {} kill {}\"' >> ~/.bashrc

source ~/.bashrc
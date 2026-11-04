# Source before every command. Keeps tools, caches and temp files inside this folder,
# so deleting capstone_pr removes everything. Nothing here touches the real home directory.
export CAP=/Users/anil.kodithyala/Downloads/Capstone/capstone_pr
export L=$CAP/.localenv
export HOME=$L/home
export TMPDIR=$L/tmp
export XDG_CACHE_HOME=$L/cache XDG_CONFIG_HOME=$L/config XDG_DATA_HOME=$L/share
export PIP_CACHE_DIR=$L/cache/pip MPLCONFIGDIR=$L/cache/mpl TORCH_HOME=$L/cache/torch
export CONDA_PKGS_DIRS=$L/miniforge/pkgs PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export PATH=$L/miniforge/bin:/usr/bin:/bin:/usr/sbin:/sbin
mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$MPLCONFIGDIR"

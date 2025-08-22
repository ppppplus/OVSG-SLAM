# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# extract semantic feature
python -m preprocess.dataset.preprocess -s data/Omni/ob_1
# train
python scripts/slam.py configs/omni/slam.py
# postprocess
python scripts/post_slam_opt.py configs/omni/post_slam_opt.py
# visualize
python viz_scripts/online_recon.py configs/omni/post_slam_opt.py
python viz_scripts/final_recon.py configs/omni/post_slam_opt.py

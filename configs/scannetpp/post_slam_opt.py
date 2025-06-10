from os.path import join as p_join

scenes = ["b20a261fdf"]

primary_device = "cuda:0"
seed = 0
group_name = "ScanNet++_postopt"
scene_name = "b20a261fdf"
param_name = f"{scene_name}_{seed}"
run_name = f"postopt_{param_name}"
param_ckpt_path = f"./experiments/ScanNet++/{param_name}/params.npz"
use_train_split = True

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=0,
    primary_device=primary_device,
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    report_iter_progress=False,
    use_wandb=False,
    wandb=dict(
        entity="my-project", # Please change the entity name
        project="SGS-SLAM",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        dataset_name="scannetpp",
        basedir="./data/scannetpp/data",
        sequence=scene_name,
        ignore_bad=False,
        use_train_split=use_train_split,
        desired_image_height=584,
        desired_image_width=876,
        start=0,
        end=-1,
        stride=1,
        eval_stride=5,
        eval_num_frames=55,
        num_frames=-1,
        param_ckpt_path=param_ckpt_path,
        load_semantics=True,
        num_semantic_classes=101
    ),
    train=dict(
        num_iters_mapping=15000,
        sil_thres=0.5, # For Addition of new Gaussians & Visualization
        use_sil_for_loss=True, # Use Silhouette for Loss during Tracking
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            seg=0.1, #0.10,
        ),
        lrs_mapping=dict(
            means3D=0.00032,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.005,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
            semantic_colors=0.0025,
        ),
        lrs_mapping_means3D_final=0.0000032,
        lr_delay_mult=0.01,
        use_gaussian_splatting_densification=True, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=500,
            remove_big_after=3000,
            stop_after=15000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=True,
            reset_opacities_every=3000, # Doesn't consider iter 0
        ),
    ),
    viz=dict(
        render_mode='color', # ['color', 'depth', 'centers', 'semantic_color]
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        visualize_cams=True, # Visualize Camera Frustums and Trajectory
        viz_w=1200, viz_h=680,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5, # FPS for Online Recon Viz
        enter_interactive_post_online=True, # Enter Interactive Mode after Online Recon Viz
        scene_name = scene_name,
        # color_dict_path="./data/Replica/color_dict.json",
        load_semantics=True, # Whether load semantic information
    ),
)

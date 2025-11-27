#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from random import randint
from utils.loss_utils import l1_loss, l1_loss_mask, l2_loss, ssim, l1_loss_confidence, ssim_confidence, load_confidence_map, uncertainty_to_confidence
from utils.depth_utils import estimate_depth, compute_confidence_map_runtime

import sys

from gaussian_renderer import render, network_gui

from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips


def save_confidence_heatmap(confidence_map, viewpoint_cam, dataset, iteration):
    """
    Save confidence map as heatmap for visualization.
    
    Args:
        confidence_map: Confidence map tensor [H, W]
        viewpoint_cam: Camera viewpoint object
        dataset: Dataset object
        iteration: Current iteration number
    """
    try:
        # Create output directory
        heatmap_dir = os.path.join(dataset.model_path, "confidence_heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)
        
        # Convert to numpy and save
        confidence_np = confidence_map.detach().cpu().numpy()
        
        # Save as heatmap image
        heatmap_path = os.path.join(heatmap_dir, f"iter_{iteration:06d}_{viewpoint_cam.image_name}_confidence.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(confidence_np, cmap='hot', vmin=0, vmax=1)
        plt.colorbar(label='Confidence')
        plt.title(f'Confidence Map - Iteration {iteration} - {viewpoint_cam.image_name}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not save confidence heatmap: {e}")


def load_confidence_map_for_camera(viewpoint_cam, dataset, confidence_dir=None, use_confidence=False, confidence_file_mapping=None):
    """
    Load confidence map for a given camera viewpoint.
    
    Args:
        viewpoint_cam: Camera viewpoint object
        dataset: Dataset object containing source_path
        confidence_dir: Directory containing confidence maps (REQUIRED if use_confidence=True)
                       This should be the path where generate_maps.py saved the uncertainty maps
                       (e.g., opt.output_dir/raw/uncert/)
        use_confidence: Whether to use confidence maps
        confidence_file_mapping: Optional dict mapping image_name to confidence file index or path
                                If None, tries to match by image_name
        
    Returns:
        Confidence map tensor [H, W] on CUDA, or None if not available
    """
    if not use_confidence or confidence_dir is None:
        return None
    
    try:
        confidence_map = None
        is_uncertainty = True  # generate_maps.py outputs uncertainty maps
        
        # Method 1: Use file mapping if provided
        if confidence_file_mapping is not None and viewpoint_cam.image_name in confidence_file_mapping:
            mapping = confidence_file_mapping[viewpoint_cam.image_name]
            if isinstance(mapping, int):
                # Index-based filename (from generate_maps.py: %06d_10.png)
                file_path = os.path.join(confidence_dir, f"{mapping:06d}_10.png")
            elif isinstance(mapping, str):
                # Direct file path
                file_path = os.path.join(confidence_dir, mapping) if not os.path.isabs(mapping) else mapping
            else:
                file_path = None
            
            if file_path and os.path.exists(file_path):
                confidence_map = load_confidence_map(file_path, normalize=True)
        
        # Method 2: Try to find by image name (if mapping not provided or not found)
        if confidence_map is None:
            image_name_base = os.path.splitext(viewpoint_cam.image_name)[0]
            possible_paths = [
                os.path.join(confidence_dir, f"{image_name_base}.png"),
                os.path.join(confidence_dir, f"{image_name_base}.jpg"),
                os.path.join(confidence_dir, f"{viewpoint_cam.image_name}.png"),
                os.path.join(confidence_dir, f"{viewpoint_cam.image_name}.jpg"),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    confidence_map = load_confidence_map(path, normalize=True)
                    break
        
        if confidence_map is None:
            return None
        
        # Convert uncertainty to confidence (generate_maps.py outputs uncertainty)
        confidence_map = uncertainty_to_confidence(confidence_map, method='inverse')
        
        # Move to CUDA and ensure correct shape
        confidence_map = confidence_map.cuda()
        
        # Ensure it matches image dimensions
        if confidence_map.shape != (viewpoint_cam.image_height, viewpoint_cam.image_width):
            confidence_map = F.interpolate(
                confidence_map.unsqueeze(0).unsqueeze(0),
                size=(viewpoint_cam.image_height, viewpoint_cam.image_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        
        return confidence_map
        
    except Exception as e:
        print(f"Warning: Could not load confidence map for {viewpoint_cam.image_name}: {e}")
        return None


def training(dataset, opt, pipe, args):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # Check if confidence maps should be used
    use_confidence = getattr(args, 'use_confidence', False)
    confidence_dir = getattr(args, 'confidence_dir', None)
    
    if use_confidence:
        if confidence_dir is not None:
            # Pre-computed confidence maps from files
            if not os.path.exists(confidence_dir):
                print(f"Warning: Confidence directory does not exist: {confidence_dir}")
                print("Switching to runtime confidence computation...")
                confidence_dir = None
            else:
                print(f"Using confidence-weighted loss with pre-computed maps from: {confidence_dir}")
        else:
            # Runtime confidence computation
            print(f"Using confidence-weighted loss with runtime computation (method: {getattr(args, 'confidence_method', 'variance')})")
        
        # Optional: Load file mapping if provided (for index-based filenames from generate_maps.py)
        confidence_file_mapping = None
        if confidence_dir is not None and hasattr(args, 'confidence_mapping_file') and args.confidence_mapping_file:
            try:
                import json
                with open(args.confidence_mapping_file, 'r') as f:
                    confidence_file_mapping = json.load(f)
                print(f"Loaded confidence file mapping from {args.confidence_mapping_file}")
            except Exception as e:
                print(f"Warning: Could not load confidence mapping file: {e}")
    else:
        confidence_file_mapping = None

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    viewpoint_stack, pseudo_stack = None, None
    ema_loss_for_log = 0.0
    
    # Cache for confidence maps (key: image_name, value: confidence_map tensor)
    # Only cache if using runtime computation (not pre-computed files)
    max_cache_size = getattr(args, 'confidence_cache_size', 50) if use_confidence and confidence_dir is None else 0
    confidence_cache = {} if max_cache_size > 0 else None
    
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        
        # Compute or load confidence map
        confidence_map = None
        if use_confidence:
            if confidence_dir is not None:
                # Option 1: Load from pre-computed files
                confidence_map = load_confidence_map_for_camera(
                    viewpoint_cam, dataset, confidence_dir, use_confidence, confidence_file_mapping
                )
            else:
                # Option 2: Compute at runtime (with caching)
                image_name = viewpoint_cam.image_name
                
                # Check cache first
                if confidence_cache is not None and image_name in confidence_cache:
                    confidence_map = confidence_cache[image_name].clone()
                else:
                    # Compute new confidence map
                    confidence_map = compute_confidence_map_runtime(
                        gt_image, 
                        num_augmentations=getattr(args, 'confidence_num_aug', 4),
                        method=getattr(args, 'confidence_method', 'variance')
                    )
                    
                    # Cache it (with size limit)
                    if confidence_cache is not None:
                        if len(confidence_cache) >= max_cache_size:
                            # Remove oldest entry (simple FIFO)
                            oldest_key = next(iter(confidence_cache))
                            del confidence_cache[oldest_key]
                        confidence_cache[image_name] = confidence_map.clone()
        
        # Compute loss with or without confidence weighting
        if confidence_map is not None:
            Ll1 = l1_loss_confidence(image, gt_image, confidence_map)
            Lssim = ssim_confidence(image, gt_image, confidence_map)
        else:
            Ll1 = l1_loss_mask(image, gt_image)
            Lssim = ssim(image, gt_image)
        
        loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - Lssim))
        
        # Save confidence heatmap occasionally (for visualization)
        if confidence_map is not None and hasattr(args, 'save_confidence_heatmap') and args.save_confidence_heatmap:
            if iteration % getattr(args, 'confidence_heatmap_interval', 1000) == 0:
                save_confidence_heatmap(confidence_map, viewpoint_cam, dataset, iteration)
        
        # Clear confidence map from memory after use (if not cached)
        # Note: Cached confidence maps are kept in memory for reuse
        if confidence_map is not None and (confidence_cache is None or viewpoint_cam.image_name not in confidence_cache):
            del confidence_map
            torch.cuda.empty_cache()


        rendered_depth = render_pkg["depth"][0]
        midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda()
        rendered_depth = rendered_depth.reshape(-1, 1)
        midas_depth = midas_depth.reshape(-1, 1)

        depth_loss = min(
                        (1 - pearson_corrcoef( - midas_depth, rendered_depth)),
                        (1 - pearson_corrcoef(1 / (midas_depth + 200.), rendered_depth))
        )
        loss += args.depth_weight * depth_loss

        if iteration > args.end_sample_pseudo:
            args.depth_weight = 0.001



        if iteration % args.sample_pseudo_interval == 0 and iteration > args.start_sample_pseudo and iteration < args.end_sample_pseudo:
            if not pseudo_stack:
                pseudo_stack = scene.getPseudoCameras().copy()
            pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))

            render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background)
            rendered_depth_pseudo = render_pkg_pseudo["depth"][0]
            midas_depth_pseudo = estimate_depth(render_pkg_pseudo["render"], mode='train')

            rendered_depth_pseudo = rendered_depth_pseudo.reshape(-1, 1)
            midas_depth_pseudo = midas_depth_pseudo.reshape(-1, 1)
            depth_loss_pseudo = (1 - pearson_corrcoef(rendered_depth_pseudo, -midas_depth_pseudo)).mean()

            if torch.isnan(depth_loss_pseudo).sum() == 0:
                loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
                loss += loss_scale * args.depth_pseudo_weight * depth_loss_pseudo


        loss.backward()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            testing_iterations, scene, render, (pipe, background))

            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration > first_iter and (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Densification
            if  iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration)


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            gaussians.update_learning_rate(iteration)
            if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                    iteration > args.start_sample_pseudo:
                gaussians.reset_opacity()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(image, gt_image, _mask).mean().double()
                    _ssim = ssim(image, gt_image, _mask).mean().double()
                    _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_00, 20_00, 30_00, 50_00, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[50_00, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[50_00, 10_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--train_bg", action="store_true")
    parser.add_argument("--use_confidence", action="store_true", default=False,
                        help="Use confidence-weighted loss. Can use --confidence_dir for pre-computed maps "
                             "or compute at runtime if --confidence_dir is not specified.")
    parser.add_argument("--confidence_dir", type=str, default=None,
                        help="Directory containing uncertainty/confidence maps from generate_maps.py "
                             "(e.g., opt.output_dir/raw/uncert/). If None, confidence maps are computed at runtime.")
    parser.add_argument("--confidence_mapping_file", type=str, default=None,
                        help="Optional JSON file mapping image_name to confidence file index or filename. "
                             "Useful when generate_maps.py uses index-based filenames (%%06d_10.png). "
                             "Format: {\"image_name.png\": 0, \"image_name2.png\": 1, ...}")
    parser.add_argument("--confidence_num_aug", type=int, default=4,
                        help="Number of augmentations to use for runtime confidence computation (default: 4)")
    parser.add_argument("--confidence_method", type=str, default='variance', choices=['variance', 'simple'],
                        help="Method for runtime confidence computation: 'variance' (variance over augmentations) "
                             "or 'simple' (depth gradient-based)")
    parser.add_argument("--save_confidence_heatmap", action="store_true", default=False,
                        help="Save confidence maps as heatmap images for visualization")
    parser.add_argument("--confidence_heatmap_interval", type=int, default=1000,
                        help="Interval (iterations) for saving confidence heatmaps (default: 1000)")
    parser.add_argument("--confidence_cache_size", type=int, default=50,
                        help="Maximum number of confidence maps to cache in memory (default: 50). "
                             "Set to 0 to disable caching.")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(args.test_iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")

from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import os
import time

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default='images/',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file '
             'against which matching needs to be done')
    parser.add_argument(
        '--target_img', type=str, default='target.jpg',
        help='Set the path of the image you are trying to match')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=0,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[500, 500],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=50,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.1,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.8,
        help='SuperGlue match threshold')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    timer = AverageTimer()
    # Target image
    target_frame  = cv2.imread(opt.target_img, cv2.IMREAD_GRAYSCALE)
    target_frame_tensor = frame2tensor(target_frame, device)

    last_data = matching.superpoint({'image': target_frame_tensor})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = target_frame_tensor
    last_frame = target_frame
    last_image_id = 0



    # Preprocessing stage to compute descriptors
    descriptor = {}
    for img in os.listdir(opt.input):
        frame = cv2.imread(os.path.join(opt.input, img), cv2.IMREAD_GRAYSCALE)
        frame_tensor = frame2tensor(frame, device)
        frame_data = matching.superpoint({'image': frame_tensor})
        frame_data = {k+'1': frame_data[k] for k in keys}
        frame_data['image1'] = frame_tensor
        descriptor[img] = frame_data
    timer.update("Pre-Process")
    timer.print()


    timer = AverageTimer()

    # Matching stage to compute distance metrics between source and target patches
    output_match = []
    for img, img_data in descriptor.items():
        pred = matching({**last_data, **img_data})
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        if confidence.mean() > 0.85:
            output_match.append(img)
        timer.update('forward')
        timer.print()
print("OUTPUT MATCH: ", output_match)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numpy.core._multiarray_umath import ndarray

import _init_paths
import os
import os.path as osp
import shutil
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

from collections import defaultdict
from lib.tracker.multitracker_byte import JDETracker, MCJDETracker, id2cls
from lib.tracking_utils import visualization as vis
from lib.tracking_utils.log import logger
from lib.tracking_utils.timer import Timer
from lib.tracking_utils.evaluation import Evaluator
import lib.datasets.dataset.jde_stcmot as datasets

from lib.tracking_utils.utils import mkdir_if_missing
from lib.opts import opts


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(
                    frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_dict(file_name, results_dict, data_type, num_classes=10):
    """
    :param file_name:
    :param results_dict:
    :param data_type:
    :param num_classes:
    :return:
    """
    # 需要根据实际的gt格式进行更改
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{score}, {cls_id},-1, -1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(file_name, 'w') as f:
        # 注意num_classes定义了需要记录的标签数量
        for cls_id in range(num_classes):  # process each object class
            # 评估visdron中的5个标签 0, 3, 4, 5, 8
            if cls_id == 1 or cls_id == 2 or cls_id == 6 or cls_id == 7 or cls_id == 9:
                continue

            cls_results = results_dict[cls_id]
            for frame_id, tlwhs, track_ids, scores in cls_results:
                if data_type == 'kitti':
                    frame_id -= 1
                for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                    if track_id < 0:
                        continue

                    x1, y1, w, h = tlwh
                    # x2, y2 = x1 + w, y1 + h
                    # line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                    # 训练时标签定义为0-9，评估时需要设置为1-10
                    re_cls_id = cls_id + 1
                    line = save_format.format(frame=frame_id,
                                              id=track_id,
                                              x1=x1, y1=y1, w=w, h=h,
                                              score=score,  # detection score
                                              cls_id=re_cls_id)
                    f.write(line)

    logger.info('save results to {}'.format(file_name))


def format_dets_dict2dets_list(dets_dict, w, h):
    """
    :param dets_dict:
    :param w: input image width
    :param h: input image height
    :return:
    """
    dets_list = []
    for k, v in dets_dict.items():
        for det_obj in v:
            x1, y1, x2, y2, score, cls_id = det_obj
            center_x = (x1 + x2) * 0.5 / float(w)
            center_y = (y1 + y2) * 0.5 / float(h)
            bbox_w = (x2 - x1) / float(w)
            bbox_h = (y2 - y1) / float(h)

            dets_list.append([int(cls_id), score, center_x, center_y, bbox_w, bbox_h])

    return dets_list


def eval_imgs_output_dets(opt,
                          data_loader,
                          data_type,
                          result_f_name,
                          out_dir,
                          save_dir=None,
                          show_image=True):
    """
    :param opt:
    :param data_loader:
    :param data_type:
    :param result_f_name:
    :param out_dir:
    :param save_dir:
    :param show_image:
    :return:
    """
    if save_dir:
        mkdir_if_missing(save_dir)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    else:
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)

    tracker = JDETracker(opt, frame_rate=30)

    timer = Timer()

    results_dict = defaultdict(list)
    frame_id = 0  # frame index(start from 0)
    for path, img, img_0 in data_loader:
        if frame_id % 30 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'
                        .format(frame_id, 1.0 / max(1e-5, timer.average_time)))

        blob = torch.from_numpy(img).to(opt.device).unsqueeze(0)

        # ----- run detection
        timer.tic()

        # update detection results
        dets_dict = tracker.update_detection(blob, img_0)

        timer.toc()
        # -----

        # plot detection results
        if show_image or save_dir is not None:
            online_im = vis.plot_detects(image=img_0,
                                         dets_dict=dets_dict,
                                         num_classes=opt.num_classes,
                                         frame_id=frame_id,
                                         fps=1.0 / max(1e-5, timer.average_time))

        if frame_id > 0:
            # 是否显示中间结果
            if show_image:
                cv2.imshow('online_im', online_im)
            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)

        # ----- 格式化并输出detection结果(txt)到指定目录
        # 格式化
        dets_list = format_dets_dict2dets_list(dets_dict, w=img_0.shape[1], h=img_0.shape[0])

        # 输出label(txt)到指定目录
        out_img_name = os.path.split(path)[-1]
        out_f_name = out_img_name.replace('.jpg', '.txt')
        out_f_path = out_dir + '/' + out_f_name
        with open(out_f_path, 'w', encoding='utf-8') as w_h:
            w_h.write('class prob x y w h total=' + str(len(dets_list)) + '\n')
            for det in dets_list:
                w_h.write('%d %f %f %f %f %f\n' % (det[0], det[1], det[2], det[3], det[4], det[5]))
        print('{} written'.format(out_f_path))

        # 处理完一帧, 更新frame_id
        frame_id += 1
    print('Total {:d} detection result output.\n'.format(frame_id))

    # 写入最终结果save results
    write_results_dict(result_f_name, results_dict, data_type)

    # 返回结果
    return frame_id, timer.average_time, timer.calls


def eval_seq(opt,
             data_loader,
             data_type,
             result_f_name,
             save_dir=None,
             show_image=True,
             frame_rate=30,
             mode='track'):

    if save_dir:
        mkdir_if_missing(save_dir)

    # JDE和MCJDE的区别？
    # tracker = JDETracker(opt, frame_rate)
    # 多类别时只能使用MCJDE，应为因为需要定义多类别的编号
    tracker = MCJDETracker(opt, frame_rate)

    timer = Timer()

    results_dict = defaultdict(list)
    frame_id = 0  # frame index

    for path, img, img0 in data_loader:
        if frame_id % 30 == 0 and frame_id != 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1.0 / max(1e-5, timer.average_time)))

        # img为固定尺寸，3，608，1088， img0 为实际图片尺寸?
        # --- run tracking
        blob = torch.from_numpy(img).unsqueeze(0).to(opt.device)

        if frame_id == 0:
            pre_blob = blob

        if mode == 'track':  # process tracking
            # ----- track updates of each frame
            timer.tic()
            online_targets_dict = tracker.update_tracking(pre_blob, blob, img0)
            timer.toc()
            # -----
            # collect current frame's result
            online_tlwhs_dict = defaultdict(list)
            online_ids_dict = defaultdict(list)
            online_scores_dict = defaultdict(list)

            for cls_id in range(opt.num_classes):  # process each class id
                online_targets = online_targets_dict[cls_id]
                for track in online_targets:
                    tlwh = track.tlwh
                    t_id = track.track_id
                    score = track.score
                    if tlwh[2] * tlwh[3] > opt.min_box_area:  # and not vertical:
                        online_tlwhs_dict[cls_id].append(tlwh)
                        online_ids_dict[cls_id].append(t_id)
                        online_scores_dict[cls_id].append(score)

            # collect result
            for cls_id in range(opt.num_classes):
                results_dict[cls_id].append((frame_id + 1,
                                             online_tlwhs_dict[cls_id],
                                             online_ids_dict[cls_id],
                                             online_scores_dict[cls_id]))

            # draw track/detection
            if show_image or save_dir is not None:
                if frame_id > 0:
                    online_im: ndarray = vis.plot_tracks(image=img0,
                                                         tlwhs_dict=online_tlwhs_dict,
                                                         obj_ids_dict=online_ids_dict,
                                                         num_classes=opt.num_classes,
                                                         frame_id=frame_id,
                                                         fps=1.0 / timer.average_time)

        elif mode == 'detect':  # process detections
            timer.tic()

            # update detection results of this frame(or image)
            dets_dict = tracker.update_detection(blob, img0)

            timer.toc()

            # plot detection results
            if show_image or save_dir is not None:
                online_im = vis.plot_detects(image=img0,
                                             dets_dict=dets_dict,
                                             num_classes=opt.num_classes,
                                             frame_id=frame_id,
                                             fps=1.0 / max(1e-5, timer.average_time))
        else:
            print('[Err]: un-recognized mode.')

        if frame_id > 0:
            if show_image:
                cv2.imshow('online_im', online_im)
            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)

        # update frame id
        frame_id += 1
        pre_blob = blob

    # write track/detection results
    write_results_dict(result_f_name, results_dict, data_type)

    return frame_id, timer.average_time, timer.calls


def main(opt,
         data_root='/data/MOT16/train',
         seqs=('MOT16-05',),
         exp_name='demo',
         save_images=False,
         save_videos=False,
         show_image=True):
    """
    """

    logger.setLevel(logging.INFO)
    # 文字结果保存路径
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:

        # 保存图片的路径
        output_dir = os.path.join(
            data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))

        # 加载数据
        dataloader = datasets.LoadImages(
            osp.join(data_root, seq), opt.img_size)

        # 每个视频序列的保存路径
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        frame_rate = 30

        # 评测过程，主要部分，输入包括：图片，mot，结果文件路径等
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)

        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(
                output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(
        all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(
        result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
        
    if opt.val_visdrone:
        seqs_str = '''uav0000086_00000_v
                      uav0000117_02622_v
                      uav0000137_00458_v
                      uav0000182_00000_v
                      uav0000268_05773_v
                      uav0000305_00000_v
                      uav0000339_00001_v
                      '''

        data_root = os.path.join(opt.data_dir, 'VisDrone2019/val/images')

    if opt.test_visdrone:
        seqs_str = '''uav0000009_03358_v
                      uav0000073_00600_v
                      uav0000073_04464_v
                      uav0000077_00720_v
                      uav0000088_00290_v
                      uav0000119_02301_v
                      uav0000120_04775_v
                      uav0000161_00000_v
                      uav0000188_00000_v
                      uav0000201_00000_v
                      uav0000249_00001_v
                      uav0000249_02688_v
                      uav0000297_00000_v
                      uav0000297_02761_v
                      uav0000306_00230_v
                      uav0000355_00001_v
                      uav0000370_00001_v
                      '''
        data_root = os.path.join(opt.data_dir, 'VisDrone2019/test_dev/images')
    seqs = [seq.strip() for seq in seqs_str.split()]

    # set device
    opt.device = 'cuda:0'
    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='save_file',
         show_image=False,
         save_images=True,
         save_videos=False)
import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib, os, cv2
import matplotlib.cm as cm
from PIL import Image
import torch.nn.functional as F
import torch


def _compute_conf_thresh(data):
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'scannet':
        thr = 5e-4
    elif dataset_name == 'megadepth':
        thr = 1e-4
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr


# --- VISUALIZATION --- #

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0)  # , cmap='gray')
    axes[1].imshow(img1)  # , cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=5)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=5)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=2)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color[..., :3], s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color[..., :3], s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig


def _make_evaluation_figure(data, b_id, alpha='dynamic'):
    b_mask = data['m_bids'] == b_id
    conf_thr = _compute_conf_thresh(data)
    
    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()
    
    # for megadepth, we visualize matches on the resized image
    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()[[1, 0]]

    epi_errs = data['epi_errs'][b_mask].cpu().numpy()
    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    n_gt_matches = int(data['conf_matrix_gt'][b_id].sum().cpu())
    recall = 0 if n_gt_matches == 0 else n_correct / (n_gt_matches)
    # recall might be larger than 1, since the calculation of conf_matrix_gt
    # uses groundtruth depths and camera poses, but epipolar distance is used here.

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)
    
    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
    ]
    
    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=text)
    return figure

def _make_confidence_figure(data, b_id):
    # TODO: Implement confidence figure
    raise NotImplementedError()


def make_matching_figures(data, config, mode='evaluation'):
    """ Make matching figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_LoFTR.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['evaluation', 'confidence']  # 'confidence'
    figures = {mode: []}
    for b_id in range(data['image0'].size(0)):
        if mode == 'evaluation':
            fig = _make_evaluation_figure(
                data, b_id,
                alpha=config.TRAINER.PLOT_MATCHES_ALPHA)
        elif mode == 'confidence':
            fig = _make_confidence_figure(data, b_id)
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
    figures[mode].append(fig)
    return figures


def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)


np.random.seed(1995)
color_map = np.arange(100)
np.random.shuffle(color_map)


def draw_topics(data, img0, img1, saved_folder="viz_topics", show_n_topics=8, saved_name=None):

    topic0, topic1 = data["topic_matrix"]["img0"], data["topic_matrix"]["img1"]
    hw0_c, hw1_c = data["hw0_c"], data["hw1_c"]
    hw0_i, hw1_i = data["hw0_i"], data["hw1_i"]
    # print(hw0_i, hw1_i)
    scale0, scale1 = hw0_i[0] // hw0_c[0], hw1_i[0] // hw1_c[0]
    if "scale0" in data:
        scale0 *= data["scale0"][0]
    else:
        scale0 = (scale0, scale0)
    if "scale1" in data:
        scale1 *= data["scale1"][0]
    else:
        scale1 = (scale1, scale1)

    n_topics = topic0.shape[-1]
    # mask0_nonzero = topic0[0].sum(dim=-1, keepdim=True) > 0
    # mask1_nonzero = topic1[0].sum(dim=-1, keepdim=True) > 0
    theta0 = topic0[0].sum(dim=0)
    theta0 /= theta0.sum().float()
    theta1 = topic1[0].sum(dim=0)
    theta1 /= theta1.sum().float()
    # top_topic0 = torch.argsort(theta0, descending=True)[:show_n_topics]
    # top_topic1 = torch.argsort(theta1, descending=True)[:show_n_topics]
    top_topics = torch.argsort(theta0*theta1, descending=True)[:show_n_topics]
    # print(sum_topic0, sum_topic1)

    topic0 = topic0[0].argmax(dim=-1, keepdim=True) #.float() / (n_topics - 1) #* 255 + 1 #
    # topic0[~mask0_nonzero] = -1
    topic1 = topic1[0].argmax(dim=-1, keepdim=True) #.float() / (n_topics - 1) #* 255 + 1
    # topic1[~mask1_nonzero] = -1
    label_img0, label_img1 = torch.zeros_like(topic0) - 1, torch.zeros_like(topic1) - 1
    for i, k in enumerate(top_topics):
        label_img0[topic0 == k] = color_map[k]
        label_img1[topic1 == k] = color_map[k]

#     print(hw0_c, scale0)
#     print(hw1_c, scale1)
    # map_topic0 = F.fold(label_img0.unsqueeze(0), hw0_i, kernel_size=scale0, stride=scale0)
    map_topic0 = label_img0.float().view(hw0_c).cpu().numpy() #map_topic0.squeeze(0).squeeze(0).cpu().numpy()
    map_topic0 = cv2.resize(map_topic0, (int(hw0_c[1] * scale0[0]), int(hw0_c[0] * scale0[1])))
    # map_topic1 = F.fold(label_img1.unsqueeze(0), hw1_i, kernel_size=scale1, stride=scale1)
    map_topic1 = label_img1.float().view(hw1_c).cpu().numpy() #map_topic1.squeeze(0).squeeze(0).cpu().numpy()
    map_topic1 = cv2.resize(map_topic1, (int(hw1_c[1] * scale1[0]), int(hw1_c[0] * scale1[1])))


    # show image0
    if saved_name is None:
        return map_topic0, map_topic1

    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    path_saved_img0 = os.path.join(saved_folder, "{}_0.png".format(saved_name))
    plt.imshow(img0)
    masked_map_topic0 = np.ma.masked_where(map_topic0 < 0, map_topic0)
    plt.imshow(masked_map_topic0, cmap=plt.cm.jet, vmin=0, vmax=n_topics-1, alpha=.3, interpolation='bilinear')
    # plt.show()
    plt.axis('off')
    plt.savefig(path_saved_img0, bbox_inches='tight', pad_inches=0, dpi=250)
    plt.close()

    path_saved_img1 = os.path.join(saved_folder, "{}_1.png".format(saved_name))
    plt.imshow(img1)
    masked_map_topic1 = np.ma.masked_where(map_topic1 < 0, map_topic1)
    plt.imshow(masked_map_topic1, cmap=plt.cm.jet, vmin=0, vmax=n_topics-1, alpha=.3, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(path_saved_img1, bbox_inches='tight', pad_inches=0, dpi=250)
    plt.close()


def draw_topicfm_demo(data, img0, img1, mkpts0, mkpts1, mcolor, text, show_n_topics=8,
                      topic_alpha=0.3, margin=5, path=None, opencv_display=False, opencv_title=''):
    topic_map0, topic_map1 = draw_topics(data, img0, img1, show_n_topics=show_n_topics)

    mask_tm0, mask_tm1 = np.expand_dims(topic_map0 >= 0, axis=-1), np.expand_dims(topic_map1 >= 0, axis=-1)

    topic_cm0, topic_cm1 = cm.jet(topic_map0 / 99.), cm.jet(topic_map1 / 99.)
    topic_cm0 = cv2.cvtColor(topic_cm0[..., :3].astype(np.float32), cv2.COLOR_RGB2BGR)
    topic_cm1 = cv2.cvtColor(topic_cm1[..., :3].astype(np.float32), cv2.COLOR_RGB2BGR)
    overlay0 = (mask_tm0 * topic_cm0 + (1 - mask_tm0) * img0).astype(np.float32)
    overlay1 = (mask_tm1 * topic_cm1 + (1 - mask_tm1) * img1).astype(np.float32)

    cv2.addWeighted(overlay0, topic_alpha, img0, 1 - topic_alpha, 0, overlay0)
    cv2.addWeighted(overlay1, topic_alpha, img1, 1 - topic_alpha, 0, overlay1)

    overlay0, overlay1 = (overlay0 * 255).astype(np.uint8), (overlay1 * 255).astype(np.uint8)

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    h, w = h0 * 2 + margin * 2, w0 * 2 + margin
    out_fig = 255 * np.ones((h, w, 3), dtype=np.uint8)
    out_fig[:h0, :w0] = overlay0
    if h0 >= h1:
        start = (h0 - h1) // 2
        out_fig[start:(start+h1), (w0+margin):(w0+margin+w1)] = overlay1
    else:
        start = (h1 - h0) // 2
        out_fig[:h0, (w0+margin):(w0+margin+w1)] = overlay1[start:(start+h0)]

    step_h = h0 + margin * 2
    out_fig[step_h:step_h+h0, :w0] = (img0 * 255).astype(np.uint8)
    if h0 >= h1:
        start = step_h + (h0 - h1) // 2
        out_fig[start:start+h1, (w0+margin):(w0+margin+w1)] = (img1 * 255).astype(np.uint8)
    else:
        start = (h1 - h0) // 2
        out_fig[step_h:step_h+h0, (w0+margin):(w0+margin+w1)] = (img1[start:start+h0] * 255).astype(np.uint8)

    # draw matching lines, this is inspried from https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/utils.py
    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    mcolor = (np.array(mcolor[:, [2, 1, 0]]) * 255).astype(int)

    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, mcolor):
        c = c.tolist()
        cv2.line(out_fig, (x0, y0+step_h), (x1+margin+w0, y1+step_h+(h0-h1)//2),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out_fig, (x0, y0+step_h), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out_fig, (x1+margin+w0, y1+step_h+(h0-h1)//2), 2, c, -1, lineType=cv2.LINE_AA)

        # Scale factor for consistent visualization across scales.
    sc = min(h / 960., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out_fig, t, (int(8 * sc), Ht + step_h*i), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out_fig, t, (int(8 * sc), Ht + step_h*i), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out_fig)

    if opencv_display:
        cv2.imshow(opencv_title, out_fig)
        cv2.waitKey(1)

    return out_fig







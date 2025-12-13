import torch
import torch.nn as nn
import torch.nn.functional as F
from .bevdet_dp import BEVDet_DP
from .bevdet import BEVDet
from mmdet.models import DETECTORS
from mmdet3d.models.utils import FFN
from mmdet3d.models.utils.spconv_voxelize import SPConvVoxelization

from .. import builder
from mmcv.runner import force_fp32
from mmcv import Config


@DETECTORS.register_module()
class DAL_Pillar(BEVDet_DP):
    def __init__(self, **kwargs):
        super(DAL_Pillar, self).__init__(**kwargs)

        # image view auxiliary task heads
        self.num_cls = self.pts_bbox_head.num_classes
        heads = dict(heatmap=(self.num_cls, 2))
        input_feat_dim = kwargs['pts_bbox_head']['hidden_channel']
        self.auxiliary_heads = FFN(
                input_feat_dim,
                heads,
                conv_cfg=dict(type="Conv1d"),
                norm_cfg=dict(type="BN1d"),
                bias=True)
        self.auxiliary_heads.init_weights()

        pts_voxel_cfg = kwargs.get('pts_voxel_layer', None)
        if pts_voxel_cfg:
            pts_voxel_cfg['num_point_features'] = 5
            self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_cfg)

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        img = self.prepare_inputs(img)
        x, _  = self.image_encoder(img[0])
        return [x] + img[1:]

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats)

    def forward_img_auxiliary_train(self,
                          x,
                          img_metas,
                          gt_bboxes,
                          gt_labels,
                          gt_bboxes_ignore=None,
                          proposals=None,
                          **kwargs):
        max_instance = 150
        num_pos = 0
        centers_augego = x[0].new_zeros((len(gt_bboxes), max_instance, 3))
        box_targets_all = x[0].new_zeros((len(gt_bboxes), max_instance, 10))
        valid_mask = x[0].new_zeros((len(gt_bboxes), max_instance, 1))
        label = x[0].new_zeros((len(gt_bboxes), max_instance, 1)).to(torch.long)
        for sid in range(len(gt_bboxes)):
            centers_augego_tmp = gt_bboxes[sid].gravity_center.to(x[0])
            box_targets_tmp = self.pts_bbox_head.bbox_coder.encode(gt_bboxes[sid].tensor)
            if gt_bboxes_ignore is not None:
                centers_augego_tmp = centers_augego_tmp[gt_bboxes_ignore[sid], :]
                box_targets_tmp = box_targets_tmp[gt_bboxes_ignore[sid], :]
            num_valid_samples = centers_augego_tmp.shape[0]
            num_pos += num_valid_samples
            valid_mask[sid, :num_valid_samples, :] = 1.0
            centers_augego[sid,:num_valid_samples,:] = centers_augego_tmp
            box_targets_all[sid,:num_valid_samples,:] = box_targets_tmp
            label_tmp = gt_labels[sid].unsqueeze(-1)
            if gt_bboxes_ignore is not None:
                label_tmp = label_tmp[gt_bboxes_ignore[sid], :]
            label[sid,:num_valid_samples,:] = label_tmp
        img_feats = self.pts_bbox_head.extract_img_feat_from_3dpoints(
            centers_augego, x, fuse=False)
        heatmap = self.auxiliary_heads.heatmap(img_feats)
        loss_cls_img = self.pts_bbox_head.loss_cls(
            heatmap.permute(0, 2, 1).reshape(-1, self.num_cls),
            label.flatten(),
            valid_mask.flatten(),
            avg_factor=max(num_pos, 1))
        return dict(loss_cls_img=loss_cls_img)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)# 和bevfusion过程相同
        img_feats_bev = \
            self.img_view_transformer(img_feats + img_inputs[1:7],
                                      depth_from_lidar=kwargs['gt_depth'])# 进行视觉变换

        losses = dict()
        losses_pts = \
            self.forward_pts_train([img_feats, pts_feats, img_feats_bev],
                                   gt_bboxes_3d, gt_labels_3d, img_metas,
                                   gt_bboxes_ignore)
        losses.update(losses_pts)
        losses_img_auxiliary = \
            self.forward_img_auxiliary_train(img_feats,img_metas,
                                             gt_bboxes_3d, gt_labels_3d,
                                             gt_bboxes_ignore,
                                             **kwargs)
        losses.update(losses_img_auxiliary)
        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    img_inputs=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        img_feats_bev = \
            self.img_view_transformer(img_feats + img_inputs[1:7],
                                      depth_from_lidar=kwargs['gt_depth'][0])

        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts([img_feats, pts_feats, img_feats_bev],
                                        img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list


@DETECTORS.register_module()
class DAL_Pillar_Temporal(DAL_Pillar):
    def __init__(self, 
                pre_process=None,
                bev_after_process=None,
                 align_after_view_transfromation=False,
                 num_adj=1,
                 with_prev=True,
                **kwargs):
        super(DAL_Pillar_Temporal, self).__init__(**kwargs)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.bev_after_process = builder.build_backbone(bev_after_process)
        self.align_after_view_transfromation = align_after_view_transfromation
        self.num_frame = num_adj + 1
        self.with_prev = with_prev
        self.grid = None
    
    def gen_grid(self, input, sensor2keyegos, bda, bda_adj=None):
        n, c, h, w = input.shape
        _, v, _, _ = sensor2keyegos[0].shape
        if self.grid is None:
            # generate grid
            xs = torch.linspace(
                0, w - 1, w, dtype=input.dtype,
                device=input.device).view(1, w).expand(h, w)
            ys = torch.linspace(
                0, h - 1, h, dtype=input.dtype,
                device=input.device).view(h, 1).expand(h, w)
            grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
            self.grid = grid
        else:
            grid = self.grid
        grid = grid.view(1, h, w, 3).expand(n, h, w, 3).view(n, h, w, 3, 1)

        # get transformation from current ego frame to adjacent ego frame
        # transformation from current camera frame to current ego frame
        c02l0 = sensor2keyegos[0][:, 0:1, :, :]

        # transformation from adjacent camera frame to current ego frame
        c12l0 = sensor2keyegos[1][:, 0:1, :, :]

        # add bev data augmentation
        bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        bda_[:, :, :3, :3] = bda.unsqueeze(1)
        bda_[:, :, 3, 3] = 1
        c02l0 = bda_.matmul(c02l0)
        if bda_adj is not None:
            bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
            bda_[:, :, :3, :3] = bda_adj.unsqueeze(1)
            bda_[:, :, 3, 3] = 1
        c12l0 = bda_.matmul(c12l0)

        # transformation from current ego frame to adjacent ego frame
        l02l1 = c02l0.matmul(torch.inverse(c12l0))[:, 0, :, :].view(
            n, 1, 1, 4, 4)
        '''
          c02l0 * inv(c12l0)
        = c02l0 * inv(l12l0 * c12l1)
        = c02l0 * inv(c12l1) * inv(l12l0)
        = l02l1 # c02l0==c12l1
        '''

        l02l1 = l02l1[:, :, :,
                      [True, True, False, True], :][:, :, :, :,
                                                    [True, True, False, True]]

        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.grid_interval[0]
        feat2bev[1, 1] = self.img_view_transformer.grid_interval[1]
        feat2bev[0, 2] = self.img_view_transformer.grid_lower_bound[0]
        feat2bev[1, 2] = self.img_view_transformer.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)
        tf = torch.inverse(feat2bev).matmul(l02l1).matmul(feat2bev)

        # transform and normalize
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0],
                                        dtype=input.dtype,
                                        device=input.device)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1,
                                                            2) * 2.0 - 1.0
        return grid

    @force_fp32()
    def shift_feature(self, input, sensor2keyegos, bda, bda_adj=None):
        grid = self.gen_grid(input, sensor2keyegos, bda, bda_adj=bda_adj)
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True)
        return output

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input, depth_from_lidar):
        x, _ = self.image_encoder(img)
        bev_feat, depth = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input], depth_from_lidar=depth_from_lidar)
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, x

    def prepare_inputs(self, inputs, stereo=False):
        # split the inputs into each frame
        B, N, C, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, C, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs[1:7]

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        curr2adjsensor = None
        if stereo:
            sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
            sensor2egos_curr = \
                sensor2egos_cv[:, :self.temporal_frame, ...].double()
            ego2globals_curr = \
                ego2globals_cv[:, :self.temporal_frame, ...].double()
            sensor2egos_adj = \
                sensor2egos_cv[:, 1:self.temporal_frame + 1, ...].double()
            ego2globals_adj = \
                ego2globals_cv[:, 1:self.temporal_frame + 1, ...].double()
            curr2adjsensor = \
                torch.inverse(ego2globals_adj @ sensor2egos_adj) \
                @ ego2globals_curr @ sensor2egos_curr
            curr2adjsensor = curr2adjsensor.float()
            curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
            curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            assert len(curr2adjsensor) == self.num_frame

        extra = [
            sensor2keyegos,
            ego2globals,
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
        return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
               bda, curr2adjsensor

    def extract_img_feat(self,
                         img,# len=7, img=[B,12,3,H,W]
                         img_metas,
                         depth_from_lidar,
                         pred_prev=False,
                         **kwargs):
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, _ = self.prepare_inputs(img)
        """Extract features of images."""
        bev_feat_list = []
        depth_list = []
        key_frame = True  # back propagation for key frame only
        for img, sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(
                imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin, post_rot,
                               post_tran, bda, mlp_input)
                if key_frame:
                    bev_feat, img_feats = self.prepare_bev_feat(*inputs_curr, depth_from_lidar=depth_from_lidar)
                else:
                    with torch.no_grad():
                        bev_feat, img_feats = self.prepare_bev_feat(*inputs_curr, depth_from_lidar=depth_from_lidar)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            key_frame = False
        if pred_prev:
            assert self.align_after_view_transfromation
            assert sensor2keyegos[0].shape[0] == 1
            feat_prev = torch.cat(bev_feat_list[1:], dim=0)
            ego2globals_curr = \
                ego2globals[0].repeat(self.num_frame - 1, 1, 1, 1)
            sensor2keyegos_curr = \
                sensor2keyegos[0].repeat(self.num_frame - 1, 1, 1, 1)
            ego2globals_prev = torch.cat(ego2globals[1:], dim=0)
            sensor2keyegos_prev = torch.cat(sensor2keyegos[1:], dim=0)
            bda_curr = bda.repeat(self.num_frame - 1, 1, 1)
            return feat_prev, [imgs[0],
                               sensor2keyegos_curr, ego2globals_curr,
                               intrins[0],
                               sensor2keyegos_prev, ego2globals_prev,
                               post_rots[0], post_trans[0],
                               bda_curr]
        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        bev_feat = self.bev_after_process(bev_feat)
        return bev_feat, img_feats

    def extract_feat(self, points, img, img_metas, depth_from_lidar):
        """Extract features from images and points."""
        bev_feat, img_feats = self.extract_img_feat(img, img_metas, depth_from_lidar)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats, bev_feat)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, bev_feat = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, depth_from_lidar=kwargs['gt_depth'])# 和bevfusion过程相同
        img_feats = [img_feats]+img_inputs[1:]
        
        losses = dict()
        losses_pts = \
            self.forward_pts_train([img_feats, pts_feats, bev_feat],
                                   gt_bboxes_3d, gt_labels_3d, img_metas,
                                   gt_bboxes_ignore)
        losses.update(losses_pts)
        losses_img_auxiliary = \
            self.forward_img_auxiliary_train(img_feats,img_metas,
                                             gt_bboxes_3d, gt_labels_3d,
                                             gt_bboxes_ignore,
                                             **kwargs)
        losses.update(losses_img_auxiliary)
        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    img_inputs=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, pts_feats, bev_feat = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, depth_from_lidar=kwargs['gt_depth'])
        img_feats = [img_feats]+img_inputs[1:]

        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts([img_feats, pts_feats, bev_feat],
                                        img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list


@DETECTORS.register_module()
class DAL_Pillar_Distill(DAL_Pillar):
    def __init__(self, 
                img_distill=None,
                pcd_distill=None,
                **kwargs):
        super(DAL_Pillar_Distill, self).__init__(**kwargs)
        
        if img_distill:
            self.build_img_net(img_distill)
        if pcd_distill:
            self.bulid_pcd_net(pcd_distill)

        self.distill_loss = torch.nn.L1Loss()
        
    def build_img_net(self, img_distill):
        self.img_branch_distill = Image_Distill_Net(img_distill)
        
    def bulid_pcd_net(self):
        pass

    def img_distill(self, img, depth_from_lidar):
        imgs = img[0]
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        if self.grid_mask is not None:
            imgs = self.grid_mask(imgs)
        x = self.img_branch_distill.img_backbone_dis(imgs)
        x = self.img_branch_distill.img_neck_dis(x)
        if type(x) in [list, tuple]:
            x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        img_feats_bev_dis = \
            self.img_branch_distill.img_view_transformer_dis([x] + img[1:7], depth_from_lidar=depth_from_lidar)# 进行视觉变换
        return img_feats_bev_dis
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)# 和bevfusion过程相同
        img_feats_bev = \
            self.img_view_transformer(img_feats + img_inputs[1:7],
                                      depth_from_lidar=kwargs['gt_depth'])# 进行视觉变换

        img_bev_dis = self.img_distill(img_inputs, depth_from_lidar=kwargs['gt_depth'])

        losses = dict()
        losses_pts = \
            self.forward_pts_train([img_feats, pts_feats, img_feats_bev],
                                   gt_bboxes_3d, gt_labels_3d, img_metas,
                                   gt_bboxes_ignore)
        losses.update(losses_pts)
        losses_img_auxiliary = \
            self.forward_img_auxiliary_train(img_feats,img_metas,
                                             gt_bboxes_3d, gt_labels_3d,
                                             gt_bboxes_ignore,
                                             **kwargs)
        losses.update(losses_img_auxiliary)
        losses_dis_img = self.distill_loss(img_feats_bev[0], img_bev_dis[0])
        losses.update({'losses_dis_img':losses_dis_img})
        return losses


@DETECTORS.register_module()
class Image_Distill_Net(nn.Module):
    def __init__(self, 
                img_distill):
        super(Image_Distill_Net, self).__init__()
        self.build_module(img_distill)
        self.init_weight(img_distill['pretrained_model'])

    def init_weight(self, pretrained_model):
        weights = torch.load(pretrained_model)['state_dict']
        temp_backbone_dict = self.img_backbone_dis.state_dict()
        temp_neck_dict = self.img_neck_dis.state_dict()
        temp_vt_dict = self.img_view_transformer_dis.state_dict()
        filtered_backbone_weights = {}
        filtered_neck_weights = {}
        filtered_vt_weights = {}
        for name in weights:
            if name.split('.')[0] == 'img_backbone':
                filtered_backbone_weights\
                    [name.replace('img_backbone', '')] = weights[name]
            if name.split('.')[0] == 'img_neck':
                filtered_neck_weights\
                    [name.replace('img_neck', '')] = weights[name]
            if name.split('.')[0] == 'img_view_transformer':
                filtered_vt_weights\
                    [name.replace('img_view_transformer', '')] = weights[name]
        temp_backbone_dict.update(filtered_backbone_weights)
        temp_neck_dict.update(filtered_neck_weights)
        temp_vt_dict.update(filtered_vt_weights)
        for param in self.img_backbone_dis.parameters():
            param.requires_grad = False
        for param in self.img_neck_dis.parameters():
            param.requires_grad = False
        for param in self.img_view_transformer_dis.parameters():
            param.requires_grad = False

    def build_module(self, img_distill):
        self.img_backbone_dis = builder.build_backbone(img_distill['dis_backbone'])
        self.img_neck_dis = builder.build_neck(img_distill['dis_neck'])
        self.img_view_transformer_dis = builder.build_neck(img_distill['dis_view_transformer'])
    

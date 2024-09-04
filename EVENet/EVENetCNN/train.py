# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
# IMPORTS
import pprint
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.global_var import get_class_names
from data_loader import loader
from models.criterions import ce_loss, dce_eviloss
from models.losses import get_loss_func
from models.networks import build_model
from models.optimizer import get_optimizer
from utils import logging, checkpoint as cp
from utils.lr_scheduler import get_lr_scheduler
from utils.meters import Meter
from utils.metrics import iou_score, precision_recall
from utils.misc import update_num_steps, plot_predictions

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg):
        # Set random seed from configs.
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        self.cfg = cfg

        # Create the checkpoint dir.
        self.checkpoint_dir = cp.create_checkpoint_dir(cfg.LOG_DIR, cfg.EXPR_NUM)
        logging.setup_logging(cfg.LOG_DIR + cfg.EXPR_NUM)
        logger.info("Training with config:")
        logger.info(pprint.pformat(cfg))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(cfg)
        self.loss_func = get_loss_func(cfg)

        # set up class names
        self.class_names = get_class_names(cfg.DATA.PLANE, cfg.DATA.CLASS_OPTIONS)

        # Set up logger format
        self.a = "{}\t" * (cfg.MODEL.NUM_CLASSES - 2) + "{}"
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.plot_dir = os.path.join(cfg.LOG_DIR, "pred", str(cfg.EXPR_NUM))
        os.makedirs(self.plot_dir, exist_ok=True)

        self.subepoch = False if self.cfg.TRAIN.BATCH_SIZE == 16 else True

    def train(self, train_loader, optimizer, scheduler, train_meter, epoch):
        self.model.train()
        logger.info("Training started ")
        epoch_start = time.time()
        loss_batch = np.zeros(1)

        for curr_iter, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

            images, labels, weights, scale_factors = batch['image'].to(self.device), \
                batch['label'].to(self.device), \
                batch['weight'].float().to(self.device), \
                batch['scale_factor']
            # images.shape 
            if not self.subepoch or (curr_iter) % (16 / self.cfg.TRAIN.BATCH_SIZE) == 0:
                optimizer.zero_grad()  # every second epoch to get batchsize of 16 if using 8

            pred = self.model(images, scale_factors)
            loss_total, loss_dice, loss_ce = self.loss_func(pred, labels, weights)

            train_meter.update_stats(pred, labels, loss_total)
            train_meter.log_iter(curr_iter, epoch)
            if scheduler is not None:
                train_meter.write_summary(loss_total, scheduler.get_last_lr(), loss_ce, loss_dice)
            else:
                train_meter.write_summary(loss_total, [self.cfg.OPTIMIZER.BASE_LR], loss_ce, loss_dice)

            loss_total.backward()
            if not self.subepoch or (curr_iter + 1) % (16 / self.cfg.TRAIN.BATCH_SIZE) == 0:
                optimizer.step()  # every second epoch to get batchsize of 16 if using 8
                if scheduler is not None:
                    scheduler.step(epoch + curr_iter / len(train_loader))

            loss_batch += loss_total.item()

            # Plot sample predictions
            if curr_iter == len(train_loader) - 2 and (epoch + 1) % 10 == 0:
                # if curr_iter == len(train_loader)-2:
                plt_title = 'Training Results Epoch ' + str(epoch)

                file_save_name = os.path.join(self.plot_dir,
                                              'Epoch_' + str(epoch) + '_Training_Predictions.pdf')

                _, batch_output = torch.max(pred, dim=1)
                plot_predictions(images, labels, batch_output, plt_title, file_save_name)

        train_meter.log_epoch(epoch)
        logger.info("Training epoch {} finished in {:.04f} seconds".format(epoch, time.time() - epoch_start))

    def train_trans_1(self, train_loader, optimizer, scheduler, train_meter, epoch):
        self.model.train()
        logger.info("Training started ")
        epoch_start = time.time()
        loss_batch = np.zeros(1)

        for curr_iter, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

            images, labels, weights, scale_factors = batch['image'].to(self.device), \
                batch['label'].to(self.device), \
                batch['weight'].float().to(self.device), \
                batch['scale_factor']

            if not self.subepoch or (curr_iter) % (16 / self.cfg.TRAIN.BATCH_SIZE) == 0:
                optimizer.zero_grad()  # every second epoch to get batchsize of 16 if using 8

            pred = self.model(images)
            pred = pred.squeeze(1)

            loss_total = F.mse_loss(pred, labels)
            loss_dice = loss_ce = loss_total

            train_meter.update_stats(pred, labels, loss_total)  # 修改了iou_update
            train_meter.log_iter(curr_iter, epoch)

            if scheduler is not None:
                train_meter.write_summary(loss_total, scheduler.get_last_lr(), loss_ce, loss_dice)
            else:
                train_meter.write_summary(loss_total, [self.cfg.OPTIMIZER.BASE_LR], loss_ce, loss_dice)

            loss_total.backward()

            if not self.subepoch or (curr_iter + 1) % (16 / self.cfg.TRAIN.BATCH_SIZE) == 0:
                optimizer.step()  # every second epoch to get batchsize of 16 if using 8
                if scheduler is not None:
                    scheduler.step(epoch + curr_iter / len(train_loader))

            loss_batch += loss_total.item()

            # Plot sample predictions
            if curr_iter == len(train_loader) - 2 and (epoch + 1) % 10 == 0:
                # if curr_iter == len(train_loader)-2:
                plt_title = 'Training Results Epoch ' + str(epoch)

                file_save_name = os.path.join(self.plot_dir,
                                              'Epoch_' + str(epoch) + '_Training_Predictions.pdf')

                _, batch_output = torch.max(pred, dim=1)
                plot_predictions(images, labels, pred, plt_title, file_save_name)

        train_meter.log_epoch(epoch)
        logger.info("Training epoch {} finished in {:.04f} seconds".format(epoch, time.time() - epoch_start))

    def train_trans_bdy(self, train_loader, optimizer, scheduler, train_meter, epoch):
        self.model.train()
        logger.info("Training started ")
        epoch_start = time.time()
        loss_batch = np.zeros(1)
        curr_iter = 0

        for images, labels in train_loader:

            images = images.float().permute(0, 3, 1, 2).to(self.device)
            labels = labels.float().permute(0, 3, 1, 2).to(self.device)

            curr_iter = curr_iter + 1

            if not self.subepoch or (curr_iter) % (16 / self.cfg.TRAIN.BATCH_SIZE) == 0:
                optimizer.zero_grad()  # every second epoch to get batchsize of 16 if using 8

            pred = self.model(images)
            loss_total = loss_dice = loss_ce = F.mse_loss(pred, labels)

            train_meter.update_stats(pred, labels, loss_total)
            train_meter.log_iter(curr_iter, epoch)
            if scheduler is not None:
                train_meter.write_summary(loss_total, scheduler.get_last_lr(), loss_ce, loss_dice)
            else:
                train_meter.write_summary(loss_total, [self.cfg.OPTIMIZER.BASE_LR], loss_ce, loss_dice)

            loss_total.backward()
            if not self.subepoch or (curr_iter + 1) % (16 / self.cfg.TRAIN.BATCH_SIZE) == 0:
                optimizer.step()  # every second epoch to get batchsize of 16 if using 8
                if scheduler is not None:
                    scheduler.step(epoch + curr_iter / len(train_loader))

            loss_batch += loss_total.item()

            # Plot sample predictions
            if curr_iter == len(train_loader) - 2 and (epoch + 1) % 10 == 0:
                # if curr_iter == len(train_loader)-2:
                plt_title = 'Training Results Epoch ' + str(epoch)

                file_save_name = os.path.join(self.plot_dir,
                                              'Epoch_' + str(epoch) + '_Training_Predictions.pdf')

                _, batch_output = torch.max(pred, dim=1)
                plot_predictions(images, labels, batch_output, plt_title, file_save_name)

        train_meter.log_epoch(epoch)
        logger.info("Training epoch {} finished in {:.04f} seconds".format(epoch, time.time() - epoch_start))

    def train_evidential(self, train_loader, optimizer, scheduler, train_meter, epoch):
        self.model.train()
        logger.info("Training started ")
        epoch_start = time.time()
        loss_batch = np.zeros(1)

        for curr_iter, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

            images, labels, weights, scale_factors = batch['image'].to(self.device), \
                batch['label'].to(self.device), \
                batch['weight'].float().to(self.device), \
                batch['scale_factor']
            # images.shape
            if not self.subepoch or (curr_iter) % (16 / self.cfg.TRAIN.BATCH_SIZE) == 0:
                optimizer.zero_grad()  # every second epoch to get batchsize of 16 if using 8

            pred = self.model(images, scale_factors)

            evidence = F.softplus(pred)
            alpha = evidence + 1
            numclass = self.cfg.MODEL.NUM_CLASSES
            lambda_epochs = 1

            loss = dce_eviloss(labels.to(torch.int64), alpha, numclass, epoch, lambda_epochs)
            loss = torch.mean(loss)
            loss_total, loss_dice, loss_ce = self.loss_func(pred, labels, weights)

            loss = loss * 0.1 + loss_total * 0.8

            train_meter.update_stats(pred, labels, loss)
            train_meter.log_iter(curr_iter, epoch)
            if scheduler is not None:
                train_meter.write_summary(loss_total, scheduler.get_last_lr(), loss_ce, loss_dice)
            else:
                train_meter.write_summary(loss_total, [self.cfg.OPTIMIZER.BASE_LR], loss_ce, loss_dice)

            # optimizer.zero_grad()
            loss.backward()
            # optimizer.step()

            if not self.subepoch or (curr_iter + 1) % (16 / self.cfg.TRAIN.BATCH_SIZE) == 0:
                optimizer.step()  # every second epoch to get batchsize of 16 if using 8
                if scheduler is not None:
                    scheduler.step(epoch + curr_iter / len(train_loader))

            loss_batch += loss.item()

            # Plot sample predictions
            if curr_iter == len(train_loader) - 2 and (epoch + 1) % 10 == 0:
                # if curr_iter == len(train_loader)-2:
                plt_title = 'Training Results Epoch ' + str(epoch)

                file_save_name = os.path.join(self.plot_dir,
                                              'Epoch_' + str(epoch) + '_Training_Predictions.pdf')

                _, batch_output = torch.max(pred, dim=1)
                plot_predictions(images, labels, batch_output, plt_title, file_save_name)

        train_meter.log_epoch(epoch)
        logger.info("Training epoch {} finished in {:.04f} seconds".format(epoch, time.time() - epoch_start))

    @torch.no_grad()
    def eval_evidential(self, val_loader, val_meter, epoch):
        logger.info(f"Evaluating model at epoch {epoch}")
        self.model.eval()

        val_loss_total = defaultdict(float)
        val_loss_dice = defaultdict(float)
        val_loss_ce = defaultdict(float)

        ints_ = defaultdict(lambda: np.zeros(self.num_classes - 1))
        unis_ = defaultdict(lambda: np.zeros(self.num_classes - 1))
        miou = np.zeros(self.num_classes - 1)
        per_cls_counts_gt = defaultdict(lambda: np.zeros(self.num_classes - 1))
        per_cls_counts_pred = defaultdict(lambda: np.zeros(self.num_classes - 1))
        accs = defaultdict(
            lambda: np.zeros(self.num_classes - 1))  # -1 to exclude background (still included in val loss)

        val_start = time.time()
        for curr_iter, batch in tqdm(enumerate(val_loader), total=len(val_loader)):

            images, labels, weights, scale_factors = batch['image'].to(self.device), \
                batch['label'].to(self.device), \
                batch['weight'].float().to(self.device), \
                batch['scale_factor']

            # mode = 'val'
            pred = self.model(images, scale_factors)
            evidence = F.softplus(pred)
            alpha = evidence + 1
            numclass = self.cfg.MODEL.NUM_CLASSES
            lambda_epochs = 1
            loss = dce_eviloss(labels.to(torch.int64), alpha, numclass, epoch, lambda_epochs)
            loss = torch.mean(loss)
            loss_total, loss_dice, loss_ce = self.loss_func(pred, labels, weights)

            loss = loss * 0.1 + loss_total * 0.8

            val_meter.update_stats(pred, labels, loss)
            val_meter.write_summary(loss)
            val_meter.log_iter(curr_iter, epoch)

            sf = torch.unique(scale_factors)
            if len(sf) == 1:
                sf = sf.item()
                val_loss_total[sf] += loss_total.item()
                val_loss_dice[sf] += loss_dice.item()
                val_loss_ce[sf] += loss_ce.item()

                _, batch_output = torch.max(pred, dim=1)

                # Calculate iou_scores, accuracy and dice confusion matrix + sum over previous batches
                int_, uni_ = iou_score(batch_output, labels, self.num_classes)
                ints_[sf] += int_
                unis_[sf] += uni_

                tpos, pcc_gt, pcc_pred = precision_recall(batch_output, labels, self.num_classes)
                accs[sf] += tpos
                per_cls_counts_gt[sf] += pcc_gt
                per_cls_counts_pred[sf] += pcc_pred

            # Plot sample predictions
            if curr_iter == (len(val_loader) // 2) and (epoch + 1) % 10 == 0:
                # if curr_iter == (len(val_loader) // 2):
                plt_title = 'Validation Results Epoch ' + str(epoch)

                file_save_name = os.path.join(self.plot_dir,
                                              'Epoch_' + str(epoch) + '_Validations_Predictions.pdf')

                _, batch_output = torch.max(pred, dim=1)
                plot_predictions(images, labels, batch_output, plt_title, file_save_name)

        val_meter.log_epoch(epoch)
        logger.info("Validation epoch {} finished in {:.04f} seconds".format(epoch, time.time() - val_start))

        # Get final measures and log them
        for key in accs.keys():
            ious = ints_[key] / unis_[key]
            miou += ious
            val_loss_total[key] /= (curr_iter + 1)
            val_loss_dice[key] /= (curr_iter + 1)
            val_loss_ce[key] /= (curr_iter + 1)

            # Log metrics
            logger.info("[Epoch {} stats]: SF: {}, MIoU: {:.4f}; "
                        "Mean Recall: {:.4f}; "
                        "Mean Precision: {:.4f}; "
                        "Avg loss total: {:.4f}; "
                        "Avg loss dice: {:.4f}; "
                        "Avg loss ce: {:.4f}".format(epoch, key, np.mean(ious),
                                                     np.mean(accs[key] / per_cls_counts_gt[key]),
                                                     np.mean(accs[key] / per_cls_counts_pred[key]),
                                                     val_loss_total[key], val_loss_dice[key], val_loss_ce[key]))

            logger.info(self.a.format(*self.class_names))
            logger.info(self.a.format(*ious))

        return np.mean(np.mean(miou))

    @torch.no_grad()
    def eval(self, val_loader, val_meter, epoch):
        logger.info(f"Evaluating model at epoch {epoch}")
        self.model.eval()

        val_loss_total = defaultdict(float)
        val_loss_dice = defaultdict(float)
        val_loss_ce = defaultdict(float)

        ints_ = defaultdict(lambda: np.zeros(self.num_classes - 1))
        unis_ = defaultdict(lambda: np.zeros(self.num_classes - 1))
        miou = np.zeros(self.num_classes - 1)
        per_cls_counts_gt = defaultdict(lambda: np.zeros(self.num_classes - 1))
        per_cls_counts_pred = defaultdict(lambda: np.zeros(self.num_classes - 1))
        accs = defaultdict(
            lambda: np.zeros(self.num_classes - 1))  # -1 to exclude background (still included in val loss)

        val_start = time.time()
        for curr_iter, batch in tqdm(enumerate(val_loader), total=len(val_loader)):

            images, labels, weights, scale_factors = batch['image'].to(self.device), \
                batch['label'].to(self.device), \
                batch['weight'].float().to(self.device), \
                batch['scale_factor']

            pred = self.model(images, scale_factors)
            loss_total, loss_dice, loss_ce = self.loss_func(pred, labels, weights)

            val_meter.update_stats(pred, labels, loss_total)
            val_meter.write_summary(loss_total)
            val_meter.log_iter(curr_iter, epoch)

            sf = torch.unique(scale_factors)
            if len(sf) == 1:
                sf = sf.item()
                val_loss_total[sf] += loss_total.item()
                val_loss_dice[sf] += loss_dice.item()
                val_loss_ce[sf] += loss_ce.item()

                _, batch_output = torch.max(pred, dim=1)

                # Calculate iou_scores, accuracy and dice confusion matrix + sum over previous batches
                int_, uni_ = iou_score(batch_output, labels, self.num_classes)
                ints_[sf] += int_
                unis_[sf] += uni_

                tpos, pcc_gt, pcc_pred = precision_recall(batch_output, labels, self.num_classes)
                accs[sf] += tpos
                per_cls_counts_gt[sf] += pcc_gt
                per_cls_counts_pred[sf] += pcc_pred

            # Plot sample predictions
            if curr_iter == (len(val_loader) // 2) and (epoch + 1) % 10 == 0:
                # if curr_iter == (len(val_loader) // 2):
                plt_title = 'Validation Results Epoch ' + str(epoch)

                file_save_name = os.path.join(self.plot_dir,
                                              'Epoch_' + str(epoch) + '_Validations_Predictions.pdf')

                _, batch_output = torch.max(pred, dim=1)
                plot_predictions(images, labels, batch_output, plt_title, file_save_name)

        val_meter.log_epoch(epoch)
        logger.info("Validation epoch {} finished in {:.04f} seconds".format(epoch, time.time() - val_start))

        # Get final measures and log them
        for key in accs.keys():
            ious = ints_[key] / unis_[key]
            miou += ious
            val_loss_total[key] /= (curr_iter + 1)
            val_loss_dice[key] /= (curr_iter + 1)
            val_loss_ce[key] /= (curr_iter + 1)

            # Log metrics
            logger.info("[Epoch {} stats]: SF: {}, MIoU: {:.4f}; "
                        "Mean Recall: {:.4f}; "
                        "Mean Precision: {:.4f}; "
                        "Avg loss total: {:.4f}; "
                        "Avg loss dice: {:.4f}; "
                        "Avg loss ce: {:.4f}".format(epoch, key, np.mean(ious),
                                                     np.mean(accs[key] / per_cls_counts_gt[key]),
                                                     np.mean(accs[key] / per_cls_counts_pred[key]),
                                                     val_loss_total[key], val_loss_dice[key], val_loss_ce[key]))

            logger.info(self.a.format(*self.class_names))
            logger.info(self.a.format(*ious))

        return np.mean(np.mean(miou))

    @torch.no_grad()
    def eval_trans_1(self, val_loader, val_meter, epoch):
        logger.info(f"Evaluating model at epoch {epoch}")
        self.model.eval()

        val_loss_total = defaultdict(float)
        val_loss_dice = defaultdict(float)
        val_loss_ce = defaultdict(float)

        ints_ = defaultdict(lambda: np.zeros(self.num_classes - 1))
        unis_ = defaultdict(lambda: np.zeros(self.num_classes - 1))
        miou = np.zeros(self.num_classes - 1)
        per_cls_counts_gt = defaultdict(lambda: np.zeros(self.num_classes - 1))
        per_cls_counts_pred = defaultdict(lambda: np.zeros(self.num_classes - 1))
        accs = defaultdict(
            lambda: np.zeros(self.num_classes - 1))  # -1 to exclude background (still included in val loss)

        val_start = time.time()
        for curr_iter, batch in tqdm(enumerate(val_loader), total=len(val_loader)):

            images, labels, weights, scale_factors = batch['image'].to(self.device), \
                batch['label'].to(self.device), \
                batch['weight'].float().to(self.device), \
                batch['scale_factor']

            pred = self.model(images)
            pred = pred.squeeze(1)
            # loss_total, loss_dice, loss_ce = self.loss_func(pred, labels, weights)
            loss_total = loss_dice = loss_ce = F.mse_loss(pred, labels)
            val_meter.update_stats(pred, labels, loss_total)
            val_meter.write_summary(loss_total)
            val_meter.log_iter(curr_iter, epoch)

            sf = torch.unique(scale_factors)
            if len(sf) == 1:
                sf = sf.item()
                val_loss_total[sf] += loss_total.item()
                val_loss_dice[sf] += loss_dice.item()
                val_loss_ce[sf] += loss_ce.item()

                _, batch_output = torch.max(pred, dim=1)

                # Calculate iou_scores, accuracy and dice confusion matrix + sum over previous batches
                int_, uni_ = iou_score(batch_output, labels, self.num_classes)
                ints_[sf] += int_
                unis_[sf] += uni_

                tpos, pcc_gt, pcc_pred = precision_recall(batch_output, labels, self.num_classes)
                accs[sf] += tpos
                per_cls_counts_gt[sf] += pcc_gt
                per_cls_counts_pred[sf] += pcc_pred

            # Plot sample predictions
            if curr_iter == (len(val_loader) // 2) and (epoch + 1) % 10 == 0:
                # if curr_iter == (len(val_loader) // 2):
                plt_title = 'Validation Results Epoch ' + str(epoch)

                file_save_name = os.path.join(self.plot_dir,
                                              'Epoch_' + str(epoch) + '_Validations_Predictions.pdf')

                _, batch_output = torch.max(pred, dim=1)
                # plot_predictions(images, labels, batch_output, plt_title, file_save_name)
                plot_predictions(images, labels, pred, plt_title, file_save_name)

        val_meter.log_epoch(epoch)
        logger.info("Validation epoch {} finished in {:.04f} seconds".format(epoch, time.time() - val_start))

        # Get final measures and log them
        for key in accs.keys():
            ious = ints_[key] / unis_[key]
            miou += ious
            val_loss_total[key] /= (curr_iter + 1)
            val_loss_dice[key] /= (curr_iter + 1)
            val_loss_ce[key] /= (curr_iter + 1)

            # Log metrics
            logger.info("[Epoch {} stats]: SF: {}, MIoU: {:.4f}; "
                        "Mean Recall: {:.4f}; "
                        "Mean Precision: {:.4f}; "
                        "Avg loss total: {:.4f}; "
                        "Avg loss dice: {:.4f}; "
                        "Avg loss ce: {:.4f}".format(epoch, key, np.mean(ious),
                                                     np.mean(accs[key] / per_cls_counts_gt[key]),
                                                     np.mean(accs[key] / per_cls_counts_pred[key]),
                                                     val_loss_total[key], val_loss_dice[key], val_loss_ce[key]))

            logger.info(self.a.format(*self.class_names))
            # logger.info(self.a.format(*ious))

        return np.mean(np.mean(miou))

    def eval_trans_bdy(self, val_loader, val_meter, epoch):
        logger.info(f"Evaluating model at epoch {epoch}")
        self.model.eval()
        curr_iter = 0
        for images, labels in val_loader:

            images = images.float().permute(0, 3, 1, 2).to(self.device)
            labels = labels.float().permute(0, 3, 1, 2).to(self.device)
            curr_iter = curr_iter + 1

            pred = self.model(images)

            pred = pred.squeeze(0)
            labels = labels.squeeze(0).to(self.device)

            loss_total = F.mse_loss(pred, labels)

            val_meter.update_stats(pred, labels, loss_total)
            val_meter.write_summary(loss_total)
            val_meter.log_iter(curr_iter, epoch)

            # Plot sample predictions
            if curr_iter == (len(val_loader) // 2) and (epoch + 1) % 10 == 0:
                # if curr_iter == (len(val_loader) // 2):
                plt_title = 'Validation Results Epoch ' + str(epoch)

                file_save_name = os.path.join(self.plot_dir,
                                              'Epoch_' + str(epoch) + '_Validations_Predictions.pdf')

                _, batch_output = torch.max(pred, dim=1)
                plot_predictions(images, labels, batch_output, plt_title, file_save_name)

        return loss_total

    def run(self):
        if self.cfg.NUM_GPUS > 1:
            assert self.cfg.NUM_GPUS <= torch.cuda.device_count(), \
                "Cannot use more GPU devices than available"
            print("Using ", self.cfg.NUM_GPUS, "GPUs!")
            self.model = torch.nn.DataParallel(self.model)

        val_loader, val_set = loader.get_dataloader(self.cfg, "val")
        train_loader, train_set = loader.get_dataloader(self.cfg, "train")

        # data_path = '/data01/30sub_tissue_boundary'
        # batch_size = 1
        # train_loader, val_loader = loader.create_dataloaders_bdy(data_path)

        update_num_steps(train_loader, self.cfg)

        # Transfer the model to device(s)
        self.model = self.model.to(self.device)

        optimizer = get_optimizer(self.model, self.cfg)
        scheduler = get_lr_scheduler(optimizer, self.cfg)

        checkpoint_paths = cp.get_checkpoint_path(self.cfg.LOG_DIR,
                                                  self.cfg.TRAIN.RESUME_EXPR_NUM)
        if self.cfg.TRAIN.RESUME and checkpoint_paths:
            try:
                checkpoint_path = checkpoint_paths.pop()
                checkpoint_epoch, best_metric = cp.load_from_checkpoint(
                    checkpoint_path,
                    self.model,
                    optimizer,
                    scheduler,
                    self.cfg.TRAIN.FINE_TUNE
                )
                start_epoch = checkpoint_epoch
                best_miou = best_metric
                logger.info(f"Resume training from epoch {start_epoch}")
            except Exception as e:
                print("No model to restore. Resuming training from Epoch 0. {}".format(e))
        else:
            logger.info("Training from scratch")
            start_epoch = 0
            best_miou = 0

        logger.info("{} parameters in total".format(sum(x.numel() for x in self.model.parameters())))

        # Create tensorboard summary writer

        writer = SummaryWriter(self.cfg.SUMMARY_PATH, flush_secs=15)

        train_meter = Meter(self.cfg,
                            mode='train',
                            global_step=start_epoch * len(train_loader),
                            total_iter=len(train_loader),
                            total_epoch=self.cfg.TRAIN.NUM_EPOCHS,
                            device=self.device,
                            writer=writer)

        val_meter = Meter(self.cfg,
                          mode='val',
                          global_step=start_epoch,
                          total_iter=len(val_loader),
                          total_epoch=self.cfg.TRAIN.NUM_EPOCHS,
                          device=self.device,
                          writer=writer)

        logger.info("Summary path {}".format(self.cfg.SUMMARY_PATH))
        # Perform the training loop.
        logger.info("Start epoch: {}".format(start_epoch + 1))

        # total hdf5 files
        total_train_subjects = 3  # 3x50
        total_val_subjects = 2

        for epoch in range(start_epoch, self.cfg.TRAIN.NUM_EPOCHS):

            for _ in range(total_train_subjects):
                self.train_evidential(train_loader, optimizer, scheduler, train_meter, epoch=epoch)

                train_set.on_epoch_end()  # change hdf5 file

                train_loader = DataLoader(
                    train_set,
                    batch_size=self.cfg.TRAIN.BATCH_SIZE,
                    num_workers=self.cfg.TRAIN.NUM_WORKERS,
                    shuffle=True,
                    pin_memory=True,
                )
            total_miou = 0.0

            for _ in range(total_val_subjects):
                cur_miou = self.eval_evidential(val_loader, val_meter, epoch=epoch)
                total_miou += cur_miou

                val_set.on_epoch_end()
                # val_loader = DataLoader(
                #     val_set,
                #     batch_size=self.cfg.TRAIN.BATCH_SIZE,
                #     num_workers=self.cfg.TRAIN.NUM_WORKERS,
                #     shuffle=False,
                #     pin_memory=True,
                # )

            miou = total_miou / total_val_subjects

            if (epoch + 1) % 10 == 0:
                logger.info(f"Saving checkpoint at epoch {epoch + 1}")
                cp.save_checkpoint(self.checkpoint_dir,
                                   epoch + 1,
                                   miou,
                                   self.cfg.NUM_GPUS,
                                   self.cfg,
                                   self.model,
                                   optimizer,
                                   scheduler)

            if miou > best_miou:
                best_miou = miou
                logger.info(
                    f"New best checkpoint reached at epoch {epoch + 1} with miou of {best_miou}\nSaving new best model.")
                cp.save_checkpoint(self.checkpoint_dir,
                                   epoch + 1,
                                   best_miou,
                                   self.cfg.NUM_GPUS,
                                   self.cfg,
                                   self.model,
                                   optimizer,
                                   scheduler,
                                   best=True
                                   )

def dice_loss(prediction, target):
    smooth = 1e-5

    intersection = torch.sum(prediction * target, dim=(1, 2, 3))
    union = torch.sum(prediction, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice.mean()

    return dice_loss


def ce_loss(prediction, target):
    ce_loss = F.cross_entropy(prediction, target)

    return ce_loss


def total_loss(prediction, target):
    dice = dice_loss(prediction, target)
    ce = ce_loss(prediction, target)

    total_loss = dice + ce

    return total_loss, dice, ce

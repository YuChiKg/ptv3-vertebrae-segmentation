"""
Adding notes for the setting
Arthor: Yu-Chi
Data: April 2025
"""
import argparse
from ptv3_model import PTv3Wrap
from data_utils.augmentation import NormalizeFeatures,AdjustRGBColor,RandomizeRateScheduler,ScheduleAdjustRGBColor
# for 2 classes with setting num_examples_per_specimen
# change it to ptv3_6_num for 6 classes
from data_utils.SpineDepthDataLoader_ptv3_num import TrainDataset, TestDataset
from torchvision.transforms import Compose
import os
import numpy as np
import logging
import torch, random
from pathlib import Path
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# %%
# reproducability
manual_seed = 42     
random.seed(manual_seed)    # Python's random module
np.random.seed(manual_seed)     # NumPy random module
torch.manual_seed(manual_seed)      # PyTorch CPU
torch.cuda.manual_seed_all(manual_seed)     # PyTorch GPU (for all devices)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# For DataLoader Reproducability
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# same sequence of random numbers 
g = torch.Generator()
g.manual_seed(manual_seed)


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='ptv3_small', help='model name [default: ptv3_small]')
    parser.add_argument('--num_class', type=int, default=2, help='number of classes [0 , 1-5]')
    parser.add_argument('--num_channel', type=int, default=6, help='number of features [x, y, z, r, g, b]')
    parser.add_argument('--num_examples', type=int, default=25, help='number of examples per specimen [default: 25/2 | default: 300/6]')
    parser.add_argument('--test_specimen_idx', type=int, default=3, help='Which specimen to use for test, option: 2, 3, 4, 5, 6, 7, 8, 9 [default: 3]')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [for ptv3_small default: 2]')
    parser.add_argument('--epoch', default=100, type=int, help='Epoch to run [default: 45]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--num_point', type=int, default=10000, help='Point Number [default: 10000]')
    parser.add_argument('--num_workers', type=int, default=12, help='number of workers [for ptv3_small default:12] to load Train and Test dataset')
    parser.add_argument('--sample_ratio', type=float, default=0.2, help='sample ratio for Train Dataset')
    parser.add_argument('--adjust_strength', type=float, default=0.3, help='How strongly to adjust towards the target RGB [212, 188, 102]')
    parser.add_argument('--randomize_rate', type=float, default=0.5, help='Proportion of vertebrae points to apply the adjustment to (0 = none, 1 = all)')
    parser.add_argument('--gamma', type=float, default=5.0, help='gamma value for weighted focal loss')
    

    return parser.parse_args()

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    test_specimen_str = args.test_specimen_idx
    num_class_str = args.num_class
    num_examples_str = args.num_examples
    # Base experiment directory
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)

    # Create subdirectory for the segmentation task and model
    experiment_dir = experiment_dir.joinpath(args.model)
    experiment_dir.mkdir(exist_ok=True)     # experiment_dir = (file name) 'ptv3'
    ## Add another directory for 2 classes
    exp_dir = experiment_dir.joinpath(f'class_{num_class_str}')
    exp_dir.mkdir(exist_ok=True)        # exp_dir = (file name/ptv3/) 'class_6'
    ## Add another directory for num_examples for each specimen
    expi_dir = exp_dir.joinpath(f'num_examples_{num_examples_str}')
    expi_dir.mkdir(exist_ok=True)      # expi_dir = (file name/class_2/num_examples_25))
    
    base_dir = expi_dir.joinpath(f'S_{test_specimen_str}')
    base_dir.mkdir(exist_ok=True) 

    # Create directories for checkpoints and logs
    checkpoints_dir = base_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    
    log_dir = base_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    
    print(f"Directories created: {base_dir}, checkpoints: {checkpoints_dir}, logs: {log_dir}")

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    
    root = '/home/travail/ptv3-vertebrae-segmentation/SpineDepth_labeled_symlink'

    NUM_CLASSES = args.num_class
    NUM_POINT = args.num_point
    NUM_CHANNELS = args.num_channel
    BATCH_SIZE = args.batch_size 
    NUM_WORKERS = args.num_workers
    
    ## applying color adjustment and normalizing when loading training dataset
    train_transforms = Compose([AdjustRGBColor(adjust_strength=args.adjust_strength, randomize_rate=args.randomize_rate),
                                NormalizeFeatures()])
    ## apply only normalization for testing
    test_transforms = Compose([NormalizeFeatures()])
    
    device = _device()
    print("start loading training data ...")
    # added num_examples argument to TRAIN_DATASET  
    TRAIN_DATASET = TrainDataset(root_dir=root, num_points=NUM_POINT, test_specimen_idx=args.test_specimen_idx, sample_ratio=args.sample_ratio, num_examples=args.num_examples, transforms=train_transforms)
    print("start loading test data ...")
    TEST_DATASET = TestDataset(root_dir=root, num_points=NUM_POINT, test_specimen_idx=args.test_specimen_idx, transforms=test_transforms)

    # For trainDataLoader: if train_dataset: transforms=None, set drop_last = True
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    train_weights = torch.Tensor(TRAIN_DATASET.labelweights).to(device)
    test_weights = torch.Tensor(TEST_DATASET.labelweights).to(device)
    
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    def ptv3_small(
        num_classes: int = NUM_CLASSES, 
        in_channels: int = NUM_CHANNELS, 
        num_points: int = NUM_POINT, 
        patch_size: int = 1024, 
        device: torch.device = _device()
    ):
        # Encoder settings
        enc_channels = (32, 64, 128, 256)
        enc_depths = (2, 2, 2, 2)
        enc_num_head = (2, 4, 8, 16)
        enc_patch_size = (patch_size,) * 4
        stride = (2, 2, 2)

        # Decoder settings
        dec_channels = (16, 64, 128)
        dec_depths = (2, 2, 2)
        dec_num_head = (4, 8, 16)
        dec_patch_size = (patch_size,) * 3

        # Initialize the PTv3Wrap model
        model = PTv3Wrap(
            num_classes=num_classes,
            in_channels=in_channels,
            num_points=num_points,
            enc_channels=enc_channels,
            enc_depths=enc_depths,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            dec_channels=dec_channels,
            dec_depths=dec_depths,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            stride=stride,
        )

        return model.to(device)

    
    class WeightedFocalLoss(nn.Module):
        def __init__(self, gamma=5.0, reduction='mean'):
            super(WeightedFocalLoss, self).__init__()
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, seg_pred, targets, alpha):
            """
            seg_pred: Logits from the model [B, C] where C = number of classes.
            targets: Ground truth labels [B].
            alpha: Class weights (1D tensor) for weighted focal loss.
            """
            # return correspond class weight depands on the target label
            alpha = alpha.gather(0, targets.view(-1)).unsqueeze(1) 
            
            # Calculate log-softmax for stability
            log_prob = F.log_softmax(seg_pred, dim=-1) 
            prob = torch.exp(log_prob)  # [B, C]

            # Gather the log probabilities for the correct classes
            log_prob = log_prob.gather(1, targets.unsqueeze(1))  # for pt 
            prob = prob.gather(1, targets.unsqueeze(1))  # [B, 1]

            # Compute focal loss
            focal_loss = -alpha * (1 - prob) ** self.gamma * log_prob

            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss

    "Add Checkpoint in the future"      
    model = ptv3_small()
    # model = ptv3_ori()

    num_epoch = args.epoch   # Start small for trying
    best_iou = 0
    best_dsc = 0
    loss_fn = WeightedFocalLoss(gamma=args.gamma).to(device)

    ### 0.001 -> 0.0001
    # optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001,
    #                               betas=(0.9, 0.999),
    #                               eps=1e-08,
    #                               weight_decay=5e-6)

    # [ ]  trry schedule learning rate
    param_dicts = [
        {"params": [param for name, param in model.named_parameters() if "block" in name], "lr": 0.0001},
        {"params": [param for name, param in model.named_parameters() if "block" not in name], "lr": 0.001},
    ]

    # Initialize the AdamW optimizer with parameter groups
    optimizer = AdamW(
        param_dicts,
        weight_decay=0.05,  # Global weight decay
        betas=(0.9, 0.999),
        eps=1e-08
    )

    # [] Define the OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[0.001, 0.0001],  # Max learning rates for each parameter group
        steps_per_epoch=len(trainDataLoader),  # Number of batches per epoch
        epochs=num_epoch,  # Total epochs for training
        pct_start=0.05,  # Percentage of the cycle to increase the learning rate
        anneal_strategy="cos",  # Cosine annealing
        div_factor=10.0,  # Initial learning rate is max_lr/div_factor
        final_div_factor=1000.0  # Final learning rate is max_lr/final_div_factor
    )
    " Start training..."  
    # ## apply transforms when loading
    # adjust_rgb = ScheduleAdjustRGBColor(
    #     adjust_strength=0.3,
    #     variance=10, 
    # )
    # randomize_scheduler = RandomizeRateScheduler(start_rate=1.0, end_rate=0.0, decay_epochs=10)  
    
    
    train_loss_list = []
    train_accuracy_list = []
    eval_loss_list = []
    eval_accuracy_list = []

    for epoch in tqdm(range(num_epoch)):
        log_string(f"Epoch: {epoch}\n-------")

        # # Update randomize_rate and pass it to AdjustRGBColor
        # current_rate = randomize_scheduler.step(epoch)
        # adjust_rgb.set_randomize_rate(current_rate)
        
        # # Log or print the current randomize_rate
        # log_string(f"Epoch {epoch} - Current Randomize Rate: {current_rate:.2f}")
        # print(f"[INFO] Epoch {epoch}: Randomize Rate is {current_rate:.2f}")
        
        train_loss = 0
        total_correct = 0
        total_seen = 0
        train_all_gt_labels = []
        train_all_pred_labels = []
        
        model.train()
        ## add a loop to loop through the training batch
        for batch, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
############# Data augmentation and normalization are applied by the DataLoader
            # set adjust_points -> points
            # adjust_points = adjust_rgb(points, target)
            
            ## prepare input_dict:
            grid_size=0.01
            coord = points[:, :, :3].float().to(device)
            feat =  points[:, :, 0:6].float().to(device)
            input_dict = {
            "coord": coord.view(-1, 3),  # [x, y, z]
            "feat": feat.view(-1, 6),  # [x, y, z, r, g, b]
            # "label": torch.tensor(target, dtype=torch.long).to(device),  # Labels (binary)
            "batch": torch.repeat_interleave(torch.arange(points.size(0)), points.size(1)).to(device),
            }
            # print(f"batch shape: {input_dict['batch'].shape}")
            # Calculate grid coordinates
            input_dict["grid_coord"] = torch.div(
                        input_dict["coord"] - input_dict["coord"].min(0)[0],
                        grid_size, rounding_mode='trunc'
                    ).int().to(device)
            # print(f"bf reshape: Grid coord shape: {input_dict['grid_coord'].shape}")
            input_dict["grid_coord"] = input_dict["grid_coord"].view(-1,3)
            # print(f"after reshape: Grid coord shape: {input_dict['grid_coord'].shape}")
            
            # Forward pass
            # put target to device
            # target = target.long().cuda()
            target = target.long().to(device)
            # 1. Forward pass in train data (here for ptv3 is a {dict}) using forward() method inside
            pred, output = model(input_dict)  # Logits output (B, num_classes)
            seg_pred = pred.contiguous().view(-1, NUM_CLASSES)
            target = target.view(-1, 1)[:, 0]
            
            # 2. Calculate the loss -> experiencing different loss function...
            # loss = F.nll_loss(seg_pred, target, train_weights)
            loss = loss_fn(seg_pred, target, train_weights)
            # 3. Zero the gradients of the optimizer (they accumulate by default)
            optimizer.zero_grad()
            # 4. Perform backpropagation on the loss
            loss.backward()
            # 5. Progress/step the optimizer (gradient descent)
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            # print(f"train_pred_choice: {pred_choice}")
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            # print(f"train_ground_truth: {batch_label}")
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            train_loss += loss.item()    # .item() -> if not, will create new variable (accumulate... -> explode!) in case consuming the GPU
            
            # For confusion matrix calculation, collect ground truth and predicted labels
            train_all_gt_labels.append(batch_label.flatten())  # Flatten ground truth labels
            train_all_pred_labels.append(pred_choice.flatten())   # Flatten predicted labels
    
        train_loss /= len(trainDataLoader)
        log_string(f'Train Loss: {train_loss:.4f}')
        train_accuracy = total_correct / float(total_seen)
        log_string(f'Train Accuracy: {train_accuracy:.4f}')  
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        
        # Concatenate all collected labels to compute confusion matrix
        train_all_gt_labels = np.concatenate(train_all_gt_labels)
        train_all_pred_labels = np.concatenate(train_all_pred_labels)
        
        train_CM = confusion_matrix(train_all_gt_labels, train_all_pred_labels, labels=np.arange(NUM_CLASSES))
        # Dynamic range for vertebrae classes (1 to NUM_CLASSES - 1)
        vertebrae_range = range(1, NUM_CLASSES)

        # Calculate TP, FN, FP, TN dynamically
        tp = np.sum([train_CM[i, i] for i in vertebrae_range])  # True positives for vertebrae
        fn = np.sum([train_CM[i, j] for i in vertebrae_range for j in range(NUM_CLASSES) if j != i])  # False negatives
        fp = np.sum([train_CM[j, i] for i in vertebrae_range for j in range(NUM_CLASSES) if j != i])  # False positives
        tn = train_CM[0, 0]  # True negatives (non-vertebrae correctly classified)

        # Calculate metrics
        acc = np.sum(np.diag(train_CM)) / np.sum(train_CM)  # Overall accuracy
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        IoU = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
        DSC = 2 * tp / (2 * tp + fn + fp) if (2 * tp + fn + fp) > 0 else 0
        f1 = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0

        # Log results
        log_string('\nTrainset Confusion Matrix:')
        log_string(train_CM)

        log_string('\nTrainset Accuracy (mean): {:.3f} %'.format(100 * acc))
        log_string('- Vertebrae Recall     : {:.3f}'.format(recall))
        log_string('- Vertebrae Precision  : {:.3f}'.format(precision))
        log_string('- Vertebrae F1 Score   : {:.3f}'.format(f1))
        log_string('- Vertebrae IoU        : {:.3f}'.format(IoU))
        log_string('- Vertebrae DSC        : {:.3f}'.format(DSC))
        
        # Evaluation Phase
        model.eval()
        all_gt_labels = []
        all_pred_labels = []
        
        # with torch.inference_mode():
        with torch.no_grad():
            eval_total_correct = 0
            eval_total_seen = 0
            eval_loss = 0

            #%# DEBUGGING
            log_string(f"Epoch: {epoch}\n-------")
            for batch, (eval_points, eval_target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                ## prepare input_dict:
                grid_size=0.01
                coord = eval_points[:, :, :3].float().to(device)
                feat =  eval_points[:, :, 0:6].float().to(device)
                # labels = target
                input_dict = {
                "coord": coord.view(-1, 3),  # [x, y, z]
                "feat": feat.view(-1, 6),  # [x, y, z, r, g, b]
                # "label": torch.tensor(labels, dtype=torch.long).to(device),  # Labels (binary)
                "batch": torch.repeat_interleave(torch.arange(eval_points.size(0)), eval_points.size(1)).to(device),
                }
                # print(f"batch shape: {input_dict['batch'].shape}")
                # Calculate grid coordinates
                input_dict["grid_coord"] = torch.div(
                            input_dict["coord"] - input_dict["coord"].min(0)[0],
                            grid_size, rounding_mode='trunc'
                        ).int().to(device)
                # print(f"bf reshape: Grid coord shape: {input_dict['grid_coord'].shape}")
                input_dict["grid_coord"] = input_dict["grid_coord"].view(-1,3)
                # print(f"after reshape: Grid coord shape: {input_dict['grid_coord'].shape}")
                
        ##########%%%%%%%%%%%%##############            
                # put target to device
                # eval_target = eval_target.long().cuda()
                eval_target = eval_target.long().to(device)
                pred, output = model(input_dict)
                seg_pred = pred.contiguous().view(-1, NUM_CLASSES)
                eval_target = eval_target.view(-1, 1)[:, 0]
    ##%%###############%%%%%%%%%%%######## 
                # loss = F.nll_loss(seg_pred, eval_target, test_weights)
                loss = loss_fn(seg_pred, eval_target, test_weights)           
                # pred_val = np.argmax(pred_val, 2)
                pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
                # print(f"eval_pred_choice: {pred_choice}")
                batch_label = eval_target.view(-1, 1)[:, 0].cpu().data.numpy()
                # print(f"eval_ground_truth: {batch_label}")
                
                eval_correct = np.sum((pred_choice == batch_label))
                eval_total_correct += eval_correct
                eval_total_seen += (BATCH_SIZE * NUM_POINT)
                eval_loss += loss.item()

                # For confusion matrix calculation, collect ground truth and predicted labels
                all_gt_labels.append(batch_label.flatten())  # Flatten ground truth labels
                all_pred_labels.append(pred_choice.flatten())   # Flatten predicted labels

            # DEBUGGING 
            eval_loss /= len(testDataLoader)
            log_string(f'Eval Loss: {eval_loss:.4f}')
            eval_accuracy = eval_total_correct / float(eval_total_seen)
            log_string(f'Eval Accuracy: {eval_accuracy:.4f}')    
            eval_loss_list.append(eval_loss)
            eval_accuracy_list.append(eval_accuracy)
                    
            # Concatenate all collected labels to compute confusion matrix
            all_gt_labels = np.concatenate(all_gt_labels)
            all_pred_labels = np.concatenate(all_pred_labels)
            
            CM = confusion_matrix(all_gt_labels, all_pred_labels, labels=np.arange(NUM_CLASSES))
            # Dynamic range for vertebrae classes (1 to NUM_CLASSES - 1)
            vertebrae_range = range(1, NUM_CLASSES)

            # Calculate TP, FN, FP, TN dynamically
            tp = np.sum([CM[i, i] for i in vertebrae_range])  # True positives for vertebrae
            fn = np.sum([CM[i, j] for i in vertebrae_range for j in range(NUM_CLASSES) if j != i])  # False negatives
            fp = np.sum([CM[j, i] for i in vertebrae_range for j in range(NUM_CLASSES) if j != i])  # False positives
            tn = CM[0, 0]  # True negatives (non-vertebrae correctly classified)

            # Calculate metrics
            acc = np.sum(np.diag(CM)) / np.sum(CM)  # Overall accuracy
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            IoU = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
            DSC = 2 * tp / (2 * tp + fn + fp) if (2 * tp + fn + fp) > 0 else 0
            f1 = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0

            # Print with 3 decimal precision
            log_string('\nConfusion Matrix:')
            log_string(CM)

            log_string('\nTestset Accuracy (mean): {:.3f} %'.format(100 * acc))
            log_string('- Recall     : {:.3f}'.format(recall))
            log_string('- Precision  : {:.3f}'.format(precision))
            log_string('- F1 Score   : {:.3f}'.format(f1))
            log_string('- Vertebrae IoU: {:.3f}'.format(IoU))
            log_string('- Vertebrae DSC: {:.3f}'.format(DSC))
            
        # save the training process
        save_process_path = str(checkpoints_dir) + '/model.pth'
        print('Saving Progress at %s' % save_process_path)
        state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_loss': train_loss_list,
                'train_accuracy': train_accuracy_list,
                'eval_loss': eval_loss_list,
                'eval_accuracy': eval_accuracy_list,
                'epoch': epoch,
                }
        torch.save(state, save_process_path)
        log_string('Saving model progress....')
        # for saving best model
        if IoU >= best_iou - 1e-6:
            best_iou = IoU
            best_dsc = DSC
            save_path = str(checkpoints_dir) + '/best_model.pth'
            print('Saving at %s' % save_path)
            state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_confusion_matrix': train_CM,
                'confusion_matrix': CM,
                'Testset Accuracy (mean)' : (100 * acc),
                'Recall': recall,
                'Precision': precision,
                'F1 Score' : f1, 
                'Vertebrae IoU' :IoU,
                'Vertebrae DSC' :DSC,
                'epoch': epoch,
                }
            torch.save(state, save_path)
            log_string('Saving model....')
        log_string('Best Vertebrae IoU: %f' % best_iou)   
        log_string('Best Vertebrae DSC: %f' % best_dsc) 

if __name__ == '__main__':
    args = parse_args()
    main(args)

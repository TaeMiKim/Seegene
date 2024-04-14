import argparse
import datetime as dt
import os
import pickle as pkl
import subprocess
import sys
import warnings
warnings.filterwarnings(action='ignore')
from copy import deepcopy
from os.path import abspath, dirname, join

import constants
import numpy as np
import reproducibility
import torch
import yaml
from deepmil.train import train_one_epoch, validate
from instantiators import (instantiate_models, instantiate_optimizer, instantiate_train_loss)
from loader import (MyDataParallel, PhotoDataset, _init_fn, csv_loader, default_collate)
from prologues import get_eval_dataset
from tools import (announce_msg, check_if_allow_multgpu_mode, copy_code,
                   copy_model_state_dict_from_gpu_to_cpu, count_nb_params,
                   create_folders_for_exp, get_cpu_device, get_device,
                   get_exp_name, get_rootpath_2_dataset,
                   get_train_transforms_img, get_transforms_tensor,
                   get_yaml_args, init_stats, load_pre_pretrained_model, log,
                   plot_curves)
from torch.utils.data import DataLoader

DEBUG_MODE = False  
PLOT_STATS = False

reproducibility.set_seed(None)  # use the default seed.
# Copy the seeds into the os.environ("MYSEED")

NBRGPUS = torch.cuda.device_count()

ALLOW_MULTIGPUS = check_if_allow_multgpu_mode()


if __name__ == "__main__":
    # =============================================
    # Parse the inputs and deal with the yaml file.
    # =============================================

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml",
                        type=str,
                        help="yaml file containing the configuration."
                        )
    parser.add_argument("--cudaid", type=str, default="0", help="cuda id.")

    input_args, _ = parser.parse_known_args()

    args, args_dict = get_yaml_args(input_args)

    os.environ["CUDA_VISIBLE_DEVICES"]= args.cudaid

    DEVICE = get_device(args)
    CPUDEVICE = get_cpu_device()

    CRITERION = torch.nn.CrossEntropyLoss()

    FOLDER = '.'
    sub_folder = "exps"

    tag = [
           ("dataset", args.dataset),
           ('bsz', args.batch_size),
           ('t', dt.datetime.now().strftime('%m_%d_%Y_%H_%M%f'))
           ]
    tag = [(el[0], str(el[1])) for el in tag]
    tag = '-'.join(['_'.join(el) for el in tag])

    parent_lv = "exps"
    if args.debug_subfolder != '':
        parent_lv = join(parent_lv, args.debug_subfolder)
    OUTD = join(dirname(abspath(__file__)),
                parent_lv,
                tag
                )

    if not os.path.exists(OUTD):
        os.makedirs(OUTD)

    OUTD_TR = create_folders_for_exp(OUTD, "train")
    OUTD_VL = create_folders_for_exp(OUTD, "validation")
    OUTD_TS = create_folders_for_exp(OUTD, "test")

    subdirs = ["init_params"]
    for sbdr in subdirs:
        if not os.path.exists(join(OUTD, sbdr)):
            os.makedirs(join(OUTD, sbdr))

    # save the yaml file.
    if not os.path.exists(join(OUTD, "code/")):
        os.makedirs(join(OUTD, "code/"))
    with open(join(OUTD, "code/", input_args.yaml), 'w') as fyaml:
        yaml.dump(args_dict, fyaml)

    copy_code(join(OUTD, "code/"))

    training_log = join(OUTD, "training.txt")
    results_log = join(OUTD, "results.txt")

    log(training_log, "\n\n ########### Training #########\n\n")
    log(results_log, "\n\n ########### Results #########\n\n")

    # ==========================================================
    # Data transformations: on PIL.Image.Image and torch.tensor.
    # ==========================================================

    train_transform_img = get_train_transforms_img(args)
    transform_tensor = get_transforms_tensor(args)

    # ==========================================================================
    # Datasets: create folds, load csv, preprocess files and save on disc,
    # load datasets: train, valid, test.
    # ==========================================================================

    announce_msg("SPLIT: {} \t FOLD: {}".format(args.split, args.fold))

    relative_fold_path = join(
        args.fold_folder, args.dataset,
        "split_{}".format(args.split), "fold_{}".format(args.fold)
    )
    if isinstance(args.name_classes, str):  # path
        path_classes = join(relative_fold_path, args.name_classes)
        msg = "File {} does not exist .... [NOT OK]".format(path_classes)
        assert os.path.isfile(path_classes), msg
        with open(path_classes, "r") as fin:
            args.name_classes = yaml.load(fin)

    train_csv = join(relative_fold_path,
                     "train_s_{}_f_{}.csv".format(args.split,args.fold))
    valid_csv = join(relative_fold_path,
                     "valid_s_{}_f_{}.csv".format(args.split, args.fold) )
    test_csv = join(relative_fold_path,
                    "test_s_{}_f_{}.csv".format(args.split, args.fold))

    # Check if the csv files exist. If not, raise an error.
    for fcsv in [train_csv, valid_csv, test_csv]:
        assert os.path.isfile(fcsv), "{} does not exist.".format(fcsv)

    rootpath = get_rootpath_2_dataset(args)

    train_samples = csv_loader(train_csv, rootpath)
    valid_samples = csv_loader(valid_csv, rootpath)
    test_samples = csv_loader(test_csv, rootpath)

    announce_msg("creating datasets and dataloaders")

    myseed = int(os.environ["MYSEED"])
    reproducibility.force_seed(myseed)
    reproducibility.force_seed(myseed)

    trainset = PhotoDataset(train_samples,
                            args.dataset,
                            args.name_classes,
                            transform_tensor,
                            set_for_eval=False,
                            transform_img=train_transform_img,
                            resize=args.resize,
                            # crop_size=args.crop_size,
                            padding_size=args.padding_size,
                            padding_mode=args.padding_mode,
                            # up_scale_small_dim_to=args.up_scale_small_dim_to,
                            do_not_save_samples=True
                            )

    reproducibility.force_seed(myseed)
    reproducibility.force_seed(myseed)
    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              worker_init_fn=_init_fn,
                              collate_fn=default_collate
                              )
    reproducibility.force_seed(myseed)
    validset, valid_loader = get_eval_dataset(args,
                                              myseed,
                                              valid_samples,
                                              transform_tensor
                                              )

    # #################### Instantiate models ##################################
    reproducibility.force_seed(myseed)
    model = instantiate_models(args)

    if args.model['path_pre_trained'] not in [None, 'None'] :
        warnings.warn("You have asked to load a specific pre-trained model "
                      "from {} .... [OK]".format(args.path_pre_trained))
        model = load_pre_pretrained_model(model=model,
                                          path_file=args.path_pre_trained,
                                          strict=args.strict
                                          )

    with open(join(OUTD, 'nbr_params.txt'), 'w') as fend:
        fend.write("Model: {}. \n NBR-params: {}.".format(
            args.model['model_name'], count_nb_params(model)
        ))

    # Check if there are multiple GPUS.
    if ALLOW_MULTIGPUS:
        model = MyDataParallel(model)
        if args.batch_size < NBRGPUS:
            warnings.warn("You asked for MULTIGPU mode. However, your batch "
                          "size {} is smaller than the number of "
                          "GPUs available {}. This is fine in practice. "
                          "However, some GPUs will be idol. "
                          "This is just a warning .... [OK]".format(
                args.batch_size, NBRGPUS))
    model.to(DEVICE)
    # Copy the model's params.
    best_state_dict = deepcopy(model.state_dict())  # it has to be deepcopy.

    # ############################ Instantiate optimizer #######################
    reproducibility.force_seed(myseed)
    optimizer, lr_scheduler = instantiate_optimizer(args, model)

    # ################################ Training ################################
    reproducibility.force_seed(myseed)
    tr_stats, tr_eval_stats, vl_stats = init_stats(), init_stats(), init_stats()

    best_val_acc = 0.
    best_val_loss = np.finfo(np.float32).max
    best_epoch = 0

    vl_stats = validate(model=model,
                        dataset=validset,
                        dataloader=valid_loader,
                        criterion=CRITERION,
                        device=DEVICE,
                        stats=vl_stats,
                        args=args,
                        folderout=OUTD_VL.folder,
                        epoch=-1,
                        log_file=training_log,
                        name_set="valid",
                        final_mode=False
                        )

    announce_msg("start training")
    reproducibility.force_seed(int(os.environ["MYSEED"]))
    tx0 = dt.datetime.now()

    for epoch in range(args.max_epochs):
        reproducibility.force_seed(myseed + (epoch + 1) * 10000 + 400)
        trainset.set_up_new_seeds()
        reproducibility.force_seed(myseed + (epoch + 2) * 10000 + 400)
        validset.set_up_new_seeds()

        # Start the training with fresh seeds.
        reproducibility.force_seed(myseed + (epoch + 3) * 10000 + 400)

        reproducibility.force_seed(myseed + (epoch + 4) * 10000 + 400)
        tr_stats = train_one_epoch(model,
                                   optimizer,
                                   train_loader,
                                   CRITERION,
                                   DEVICE,
                                   tr_stats,
                                   args,
                                   epoch,
                                   training_log,
                                   ALLOW_MULTIGPUS=ALLOW_MULTIGPUS,
                                   NBRGPUS=NBRGPUS
                                   )

        if lr_scheduler:  # for > 1.1 : opt.step() then l_r_s.step().
            lr_scheduler.step(epoch)
        # Eval validation set.
        reproducibility.force_seed(myseed + (epoch + 5) * 10000 + 400)
        vl_stats = validate(model=model,
                            dataset=validset,
                            dataloader=valid_loader,
                            criterion=CRITERION, 
                            device=DEVICE,
                            stats=vl_stats,
                            args=args,
                            folderout=OUTD_VL.folder,
                            epoch=epoch,
                            log_file=training_log,
                            name_set="valid",
                            final_mode=False
                            )

        reproducibility.force_seed(myseed + (epoch + 6) * 10000 + 400)

        vl_acc = vl_stats["acc"][-1]
        vl_loss = vl_stats["total_loss"][-1]

        if vl_acc >= best_val_acc:
            print("BEST VALID ABOVE.")
            best_val_loss = vl_loss
            best_val_acc = vl_acc
            best_state_dict = deepcopy(model.state_dict())

            # torch.save(best_model.state_dict(), join(OUTD, "best_model.pt"))
            best_epoch = epoch

        # CRITERION.update_t()

        if epoch < (args.max_epochs - 1):
            model.sigma = min(args.model['max_sigma'],
                              model.sigma + args.model['delta_sigma']
                              )


    # Reset the models parameters to the best found ones.
    model.load_state_dict(best_state_dict)

    announce_msg("End training. Time: {}".format(dt.datetime.now() - tx0))
    log(results_log, "Best epoch: {}".format(best_epoch))

    tx0 = dt.datetime.now()

    reproducibility.force_seed(int(os.environ["MYSEED"]))

    plot_curves(tr_stats,
                join(OUTD_TR.folder, "train.png"),
                "Train stats. Best epoch: {}.".format(best_epoch)
                )
    plot_curves(vl_stats,
                join(OUTD_VL.folder, "validation.png"),
                "Eval (validation set) stats. Best epoch: {}.".format(best_epoch)
                )

    announce_msg("start final processing stage")

    del trainset
    del train_loader

    STORE_ON_DISC = True

    reproducibility.force_seed(myseed)
    validate(model=model,
             dataset=validset,
             dataloader=valid_loader,
             criterion=CRITERION,
             device=DEVICE,
             stats=None,
             args=args,
             folderout=OUTD_VL.folder,
             epoch=best_epoch,
             log_file=results_log,
             name_set="valid",
             store_on_disc=STORE_ON_DISC,
             store_imgs=(args.dataset in [constants.GLAS]),
             final_mode=True
             )
    del validset
    del valid_loader

    testset, test_loader = get_eval_dataset(args,
                                            myseed,
                                            test_samples,
                                            transform_tensor
                                            )
    reproducibility.force_seed(myseed)
    validate(model=model,
             dataset=testset,
             dataloader=test_loader,
             criterion=CRITERION,
             device=DEVICE,
             stats=None,
             args=args,
             folderout=OUTD_TS.folder,
             epoch=best_epoch,
             log_file=results_log,
             name_set="test",
             store_on_disc=STORE_ON_DISC,  # todo: set to true.
             store_imgs=(args.dataset in [constants.GLAS]),  # todo: set to true
             final_mode=True
             )
    del testset
    del test_loader

    reproducibility.force_seed(myseed)
    trainset_eval, train_eval_loader = get_eval_dataset(args,
                                                        myseed,
                                                        train_samples,
                                                        transform_tensor
                                                        )
    reproducibility.force_seed(myseed)
    validate(model=model,
             dataset=trainset_eval,
             dataloader=train_eval_loader,
             criterion=CRITERION,
             device=DEVICE,
             stats=None,
             args=args,
             folderout=OUTD_TR.folder,
             epoch=best_epoch,
             log_file=results_log,
             name_set="train",
             store_on_disc=STORE_ON_DISC,
             store_imgs=(args.dataset in [constants.GLAS]),
             final_mode=True
             )

    del trainset_eval
    del train_eval_loader

    # Save train statistics (train, valid)
    stats_to_dump = {
        "train": tr_stats,
        "valid": vl_stats
    }
    with open(join(OUTD, "train_stats.pkl"), "wb") as fout:
        pkl.dump(stats_to_dump, fout, protocol=pkl.HIGHEST_PROTOCOL)

    # Move the state dict of the best model into CPU, then save it.
    best_state_dict_cpu = copy_model_state_dict_from_gpu_to_cpu(model)
    torch.save(best_state_dict_cpu, join(OUTD, "best_model.pt"))
    announce_msg("End final processing. Time: {}".format(dt.datetime.now() - tx0))

    announce_msg("*END-BYE*")
    

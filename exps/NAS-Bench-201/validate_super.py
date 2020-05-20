import os, sys, time, torch, random, argparse
from PIL     import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy    import deepcopy
from pathlib import Path

lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from procedures   import get_optim_scheduler
from config_utils import load_config
from datasets     import get_datasets
from log_utils    import Logger, AverageMeter, time_string, convert_secs2time
from functions import pure_evaluate, procedure

def get_op_list(n):
    ops = []
    while len(ops) < 6:
        ops.insert(0, int(n % 5))
        n /= 5
    return ops

def getvalue(item, key):
    return item[:-4].split('_')[item[:-4].split('_').index(key)+1]

def main(super_path, ckp_path, workers, datasets, xpaths, splits, use_less):
    from config_utils import dict2config
    from models import get_cell_based_tiny_net
    logger = Logger(str(ckp_path), 0, False)

    ckp = torch.load(super_path)
    from collections import OrderedDict
    state_dict = OrderedDict()
    model_name = super_path.split('/')[2][:-8]
    old_state_dict = ckp['shared_cnn'] if model_name == 'ENAS' else ckp['search_model']
    for k, v in old_state_dict.items():
        if 'module' in k:
            name = k[7:] # remove `module.`
        else:
            name = k
        state_dict[name] = v

    model_config = dict2config({'name': model_name,
                                'C': 16,
                                'N': 5,
                                'max_nodes': 4,
                                'num_classes': 10,
                                'space': ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3'],
                                'affine': False,
                                'track_running_stats': True}, None)
    supernet = get_cell_based_tiny_net(model_config)
    # supernet.load_state_dict(ckp['search_model'])
    supernet.load_state_dict(state_dict)

    ckp_names = os.listdir(ckp_path)
    from datetime import datetime
    random.seed(datetime.now())
    random.shuffle(ckp_names)
    for ckp_name in ckp_names:
        if not ckp_name.endswith('.tar'):
            continue
        if 'super' in ckp_name:
            continue
        if not os.path.exists(os.path.join(ckp_path, ckp_name)):
            continue
        arch = getvalue(ckp_name, 'arch')
        op_list = get_op_list(int(arch))
        net = supernet.extract_sub({
          '1<-0': op_list[0],
          '2<-0': op_list[1],
          '2<-1': op_list[2],
          '3<-0': op_list[3],
          '3<-1': op_list[4],
          '3<-2': op_list[5],
        })
        network = torch.nn.DataParallel(net).cuda()
        valid_losses, valid_acc1s, valid_acc5s, valid_tms = evaluate_all_datasets(network, datasets, xpaths, splits, use_less, workers, logger)
        try:
            old_ckp = torch.load(os.path.join(ckp_path, ckp_name))
        except:
            print(ckp_name)
            continue

        for key in valid_losses:
            old_ckp[key] = valid_losses[key]
        for key in valid_acc1s:
            old_ckp[key] = valid_acc1s[key]
        for key in valid_acc5s:
            old_ckp[key] = valid_acc5s[key]
        for key in valid_tms:
            old_ckp[key] = valid_tms[key]
        old_ckp['super'] = network.module.state_dict()

        cf10_super = valid_acc1s['cf10-otest-acc1']
        cf100_super = valid_acc1s['cf100-otest-acc1']
        img_super = valid_acc1s['img-otest-acc1']
        new_ckp_name = ckp_name[:-4] + f'_cf10-super_f{cf10_super}_cf100-super_f{cf100_super}_img-super_f{img_super}' + '.tar'
        torch.save(old_ckp, os.path.join(ckp_path, new_ckp_name))
        os.remove(os.path.join(ckp_path, ckp_name))



def evaluate_all_datasets(network, datasets, xpaths, splits, use_less, workers, logger):

      valid_losses, valid_acc1s, valid_acc5s, valid_tms = {}, {}, {}, {}

      for dataset, xpath, split in zip(datasets, xpaths, splits):
        # train valid data
        train_data, valid_data, xshape, class_num = get_datasets(dataset, xpath, -1)
        # load the configuration
        if dataset == 'cifar10' or dataset == 'cifar100':
          if use_less: config_path = 'configs/nas-benchmark/LESS.config'
          else       : config_path = 'configs/nas-benchmark/CIFAR.config'
          split_info  = load_config('configs/nas-benchmark/cifar-split.txt', None, None)
        elif dataset.startswith('ImageNet16'):
          if use_less: config_path = 'configs/nas-benchmark/LESS.config'
          else       : config_path = 'configs/nas-benchmark/ImageNet-16.config'
          split_info  = load_config('configs/nas-benchmark/{:}-split.txt'.format(dataset), None, None)
        else:
          raise ValueError('invalid dataset : {:}'.format(dataset))
        config = load_config(config_path, \
                                {'class_num': class_num,
                                 'xshape'   : xshape}, \
                                logger)
        # check whether use splited validation set
        if bool(split):
          assert dataset == 'cifar10'
          ValLoaders = {'ori-test': torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, shuffle=False, num_workers=workers, pin_memory=True)}
          assert len(train_data) == len(split_info.train) + len(split_info.valid), 'invalid length : {:} vs {:} + {:}'.format(len(train_data), len(split_info.train), len(split_info.valid))
          train_data_v2 = deepcopy(train_data)
          train_data_v2.transform = valid_data.transform
          valid_data = train_data_v2
          # data loader
          train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(split_info.train), num_workers=workers, pin_memory=True)
          valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(split_info.valid), num_workers=workers, pin_memory=True)
          ValLoaders['x-valid'] = valid_loader
        else:
          # data loader
          train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=workers, pin_memory=True)
          valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, shuffle=False, num_workers=workers, pin_memory=True)
          if dataset == 'cifar10':
            ValLoaders = {'ori-test': valid_loader}
          elif dataset == 'cifar100':
            cifar100_splits = load_config('configs/nas-benchmark/cifar100-test-split.txt', None, None)
            ValLoaders = {'ori-test': valid_loader,
                          'x-valid' : torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(cifar100_splits.xvalid), num_workers=workers, pin_memory=True),
                          'x-test'  : torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(cifar100_splits.xtest ), num_workers=workers, pin_memory=True)
                         }
          elif dataset == 'ImageNet16-120':
            imagenet16_splits = load_config('configs/nas-benchmark/imagenet-16-120-test-split.txt', None, None)
            ValLoaders = {'ori-test': valid_loader,
                          'x-valid' : torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(imagenet16_splits.xvalid), num_workers=workers, pin_memory=True),
                          'x-test'  : torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(imagenet16_splits.xtest ), num_workers=workers, pin_memory=True)
                         }
          else:
            raise ValueError('invalid dataset : {:}'.format(dataset))

        dataset_key = '{:}'.format(dataset)
        if bool(split): dataset_key = dataset_key + '-valid'
        logger.log('Evaluate ||||||| {:10s} ||||||| Train-Num={:}, Valid-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(dataset_key, len(train_data), len(valid_data), len(valid_loader), config.batch_size))

        optimizer, scheduler, criterion = get_optim_scheduler(network.parameters(), config)
        for epoch in range(config.epochs):  # finetune 5 epochs
            scheduler.update(epoch, 0.0)
            procedure(train_loader, network, criterion, scheduler, optimizer, 'train')

        short = {
            'ImageNet16-120': 'img',
            'cifar10': 'cf10',
            'cifar100': 'cf100',
            'ori-test': 'otest',
            'x-valid': 'xval',
            'x-test': 'xtest'
        }
        with torch.no_grad():
          for key, xloder in ValLoaders.items():
            valid_loss, valid_acc1, valid_acc5, valid_tm = pure_evaluate(xloder, network)
            valid_losses[short[dataset]+'-'+short[key]+'-'+'loss'] = valid_loss
            valid_acc1s[short[dataset]+'-'+short[key]+'-'+'acc1'] = valid_acc1
            valid_acc5s[short[dataset]+'-'+short[key]+'-'+'acc5'] = valid_acc5
            valid_tms[short[dataset]+'-'+short[key]+'-'+'tm'] = valid_tm
            logger.log('Evaluate ---->>>> {:10s} top1: {:}  top5: {:}  loss: {:}'.format(key, valid_acc1, valid_acc5, valid_loss))
      return valid_losses, valid_acc1s, valid_acc5s, valid_tms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NAS-Bench-201', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--arch_path',    type=str,                   help='Path to load supernet')
    parser.add_argument('--ckp_path',    type=str,                   help='Folder to checkpoints.')
    # use for train the model
    parser.add_argument('--workers',     type=int,   default=8,      help='number of data loading workers (default: 2)')
    parser.add_argument('--datasets',    type=str,   nargs='+',      help='The applied datasets.')
    parser.add_argument('--xpaths',      type=str,   nargs='+',      help='The root path for this dataset.')
    parser.add_argument('--splits',      type=int,   nargs='+',      help='The root path for this dataset.')
    parser.add_argument('--use_less',    type=int,   default=1,      choices=[0,1], help='Using the less-training-epoch config.')
    parser.add_argument('--seed',    type=int,   default=0,          help='Using the less-training-epoch config.')
    args = parser.parse_args()

    main(args.arch_path, args.ckp_path, args.workers, args.datasets, args.xpaths, args.splits, args.use_less > 0)

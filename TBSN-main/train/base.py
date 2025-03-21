import argparse
from torch.utils.data import DataLoader
from util.build import build
from util.io import log
from util.option import parse, recursive_log
from validate.base import validate_sidd, validate_dnd, validate_synthetic
from util.data_loader import data_load 

def main(opt):
   #train_dataset_opt = opt['train_dataset']
   #TrainDataset = getattr(__import__('dataset'), train_dataset_opt['type'])
   #train_set = build(TrainDataset, train_dataset_opt['args'])
   #train_loader = DataLoader(train_set, batch_size=train_dataset_opt['batch_size'], shuffle=True,
   #                           num_workers=4, drop_last=True)
   
   train_loader = data_load(train=True)
   #print('begin validation load.....')
   validation_loaders = data_load(train=False)
   #for validation_dataset_opt in opt['validation_datasets']:
   #     ValidationDataset = getattr(__import__('dataset'), validation_dataset_opt['type'])
   #     validation_set = build(ValidationDataset, validation_dataset_opt['args'])
   #     validation_loader = DataLoader(validation_set, batch_size=1)
   #     validation_loaders.append(validation_loader)
   #print('validation data load over .....')
   Model = getattr(__import__('model'), opt['model'])
   model = Model(opt)
   model.data_parallel()

   def train_step(data):
        model.train_step(data)
        #print('until here is normal......')
        if model.iter % opt['print_every'] == 0:
            model.log()
        pnsr_max = 0
        #if model.iter % opt['save_every'] == 0:
        #    model.save_net()

        if model.iter % opt['validate_every'] == 0:
            message = 'iter: %d, ' % model.iter
            #for validation_loader in validation_loaders:
                #if 'SIDD' in validation_loader.dataset.__class__.__name__:
            #print('validation data type is',type(validation_loaders))
            psnr, ssim,lpipss = validate_sidd(model, validation_loaders)
                #elif 'DND' in validation_loader.dataset.__class__.__name__:
                #    psnr, ssim = validate_dnd(model, validation_loader)
                #elif 'Synthetic' in validation_loader.dataset.__class__.__name__:
                #    psnr, ssim = validate_synthetic(model, validation_loader)
            message += '%s: %6.4f/%6.4f ' % (validation_loaders.dataset.__class__.__name__, psnr, ssim)
            if pnsr_max<psnr:
                pnsr_max = psnr
                model.save_net()
            log(opt['log_file'], message + '\n')

   while True:
        for i, data in enumerate(train_loader):
            train_step(data)
            print('trainning iter:{} th '.format(model.iter))
            if model.iter >= opt['num_iters']:
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the denoiser")
    parser.add_argument("--config", type=str, default='')
    argspar = parser.parse_args()

    opt = parse(argspar.config)
    recursive_log(opt['log_file'], opt)

    main(opt)

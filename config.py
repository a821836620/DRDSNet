import argparse

parser = argparse.ArgumentParser(description='DDPM Segmentaion breast tumor')
parser.add_argument('--seed', type=int, default=3407, help='random seed')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
parser.add_argument('--gpu', type=str, default='0', help='gpu')
parser.add_argument('--in_size', type=str, default='96,96,48', help='input size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--model', type=str, default='unet3d', help='model')
parser.add_argument('--loss', type=str, default='dice', help='loss')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
parser.add_argument('--scheduler', type=str, default='plateau', help='scheduler')

parser.add_argument('--save_path', type=str, default='save', help='save path')

parser.add_argument('--data', type=str, default='patch', help='data')
parser.add_argument('--root', type=str, default='', help='train path')
parser.add_argument('--T', type=int, default=5, help='MRI time')
parser.add_argument('--test_path', type=str, default='', help='test path')

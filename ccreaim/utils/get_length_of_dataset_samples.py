import soundfile as sf
from os import listdir
from os.path import isfile, join
samples_dir_path = '/scratch/other/sopi/CCREAIM/datasets/samples/'
files = [f for f in listdir(samples_dir_path) if isfile(join(samples_dir_path, f)) and not f.startswith('.') and f.endswith('.wav')]

total_len = 0

for fname in files:
    f = sf.SoundFile(join(samples_dir_path, fname))
    print('samples = {}'.format(f.frames))
    print('sample rate = {}'.format(f.samplerate))
    print('seconds = {}'.format(f.frames / f.samplerate))
    total_len += f.frames / f.samplerate

print(total_len)
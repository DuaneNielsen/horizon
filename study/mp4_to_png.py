import imageio as iio
import imageio_ffmpeg
from tqdm import tqdm

reader = iio.get_reader('/home/duane/Downloads/drone_fpv1.mp4')
frames = reader.count_frames()

i = 0
for im in tqdm(reader, total=frames):
    if i % 50 == 0:
        iio.imwrite(f'data/horizon/{i}.png', im, 'png')
    i += 1

"""
Data Loader for pytorch
"""

from util import *


class SongLoader:
    # File path for decoded song ids
    SONG_FP = os.path.join(PATH_BASE, DIR_DSET, config(f'{DIR_DSET}.my.dir_nm'), f'Song-ids.json')

    def __init__(self, pad=True):
        """
        :param pad: If true, instances returned are padded to the maximum sequence length
        """
        self.pad = pad
        with open(SongLoader.SONG_FP, 'r') as f:
            songs: list[dict[str]] = json.load(f)
            # ic(len(songs), songs[0])
            n, l_max = len(songs), max(len(s['ids']) for s in songs)
            # ic(n, l_max)
            # self.nms = [d['nm'] for d in songs]
            # self.ids = np.fromiter(
            #     (join_its(itertools.zip_longest(*[d['ids'] for d in songs], fillvalue=0))),
            #     dtype=int, count=n * l_max
            # ).reshape(n, l_max)
            self.ids = np.array(list(itertools.zip_longest(*[d['ids'] for d in songs], fillvalue=nan))).T
            # ic(self.ids, self.ids.shape)

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        ids_ = self.ids[idx]
        return ids_ if self.pad else ids_[~np.isnan(ids_)]


if __name__ == '__main__':
    from icecream import ic

    sl = SongLoader()
    ic(len(sl))
    ic(sl[0], sl[0].shape, SongLoader(pad=False)[0].shape)

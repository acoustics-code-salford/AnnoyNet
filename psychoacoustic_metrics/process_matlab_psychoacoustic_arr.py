import numpy as np
import pandas as pd
from datasets import MomentaryAnnoyance

def _w_s(s, n_5):
    if s > 1.75:
        return (s - 1.75) * 0.25 * np.log(n_5 + 10)
    else:
        return 0

def _w_fr(f, r, n_5):
    return (2.18 / n_5**0.4) * (0.4*f + 0.6*r)

def zwicker_annoyance(n_5, r, s, f):
    w_s = _w_s(s, n_5)
    w_fr = _w_fr(f, r, n_5)
    return n_5 * (1 + np.sqrt(w_s**2 + w_fr**2))


def zwicker_process(row):
    return zwicker_annoyance(row['loudness_n5'],
                             row['roughness'],
                             row['sharpness'],
                             row['fluctuation'])


psycho_df = pd.read_csv('psychoacoustic_metrics.csv', index_col=0)
psycho_df = psycho_df.drop(
    ['folder', 'date', 'bytes', 'isdir', 'datenum'], axis=1)
psycho_df['zwicker_annoyance'] = psycho_df.apply(zwicker_process, axis=1)

data = MomentaryAnnoyance('ml_data/', n_fft=128, hop_length=64, n_mels=25)
if all(psycho_df.index == data.targets.index):
    print('All keys match. Appending perceived_annoyance column.')
else:
    print('Keys do not match. Aborting process.')

psycho_df['perceived_annoyance'] = data.targets['Annoyance']

psycho_df.to_csv('psycho_metrics_processed.csv')
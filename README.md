# AnnoyNet
Development of ML models to estimate perceived annoyance of UAV & conventional 
aircraft events directly from audio clips.

**Author**: Marc C. Green (m.c.green@salford.ac.uk)

**Affiliation**: University of Salford (Acoustics Research Centre)

**Copyright statement**: This file and code is part of work undertaken within
the REFMAP project (www.refmap.eu), and is subject to license.

## Usage
`datasets.py` contains `MomentaryAnnoyance`, which is used to load preprocessed
six-second clips and mean perceived annoyance ratings stored in 
[this directory](https://testlivesalfordac-my.sharepoint.com/:f:/g/personal/m_c_green_salford_ac_uk/EqiEsZYyrSxAnTaVOIeMpWEB5GGZATPuuPiL-9rI3i4oig?e=b6yJwB).
Initialisation options include:
* `input_path` - specifies data location.
* `key_select` - allows specification of subset of data (e.g. [`KTH`, `MJL`]) will
  load data from only these subsets.
* `n_fft`, `hop_length`, `n_mels` - Inputs to torchaudio's `MelSpectrogram` transform.
* `fs` - Input audio will be resampled to this frequency.

`models.py` contains the simple `AnnoyCNN` and `AnnoyCRNN` models. These both
contain two convolutional layers and three dense layers, with the CRNN model
adding a GRU layer between these stages. These models have customisable kernels
for the convolution and MaxPool stages. The `input_shape` parameter set on
initialisation auto-scales the dense layers based on the resolution of the input
mel-spectrograms.

`train_tools.py` contains functions for automated training and testing of these 
models, with options for number of epochs to train, and early stopping.
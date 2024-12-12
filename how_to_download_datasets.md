## Instructions on how to download the datasets

The sentiment analysis dataset is available at [this address](https://ai.stanford.edu/~amaas/data/sentiment/). It can be downloaded by pressing the
"Large Movie Review Dataset v1.0" button at the top of the page. In the folder that gets downloaded, there will be two subfolders: `train/` and `test/`. These
subfolders are to be placed inside the `sentiment_analysis_dataset/` folder of this repository. Of all the contents in these two folders, only the `neg/` and
`pos/` subfolders of each of `train/` and `test/` are to be kept.

The hate speech dataset is available at [this address](https://www.kaggle.com/datasets/waalbannyantudre/hate-speech-detection-curated-dataset/data). It can be
downloaded in the same way as the previous one. In the folder thast gets downloaded, there will be two .csv files. The one the script is working with is
`HateSpeechDatasetBalanced.csv` (and not `HateSpeechDataset.csv`), but renamed as `HateSpeechDataset.csv`. This file has to be placed in the folder
`datasets/hate_speech_dataset`.

This is done in order to circumvent GitHub's 100MB file size limit imposed on the `git push` operation.

Enjoy!
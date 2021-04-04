Setup Detectron env

```
conda install python=3.7
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 opencv -c pytorch

python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.7/index.html
pip install tqdm
```

First set up datasets
```
cd datasets
./data.py
```

Then train:
```
./run.sh
```

Then run inference on test images:
```
./test.sh
```

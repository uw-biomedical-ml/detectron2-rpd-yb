Setting up Detectron2-RPD
 

Step 1) Install C++ Build Tools

Prior to starting the installation process, make sure to save any important files. In order to finish the installation, a restart is required.

Run "vs_BuildTools.exe"

![image](https://user-images.githubusercontent.com/46503967/145657344-e8cf16ae-2ae4-4baf-a9f4-2637251c42eb.png)

After a couple of seconds, the following window should pop up:

![image](https://user-images.githubusercontent.com/46503967/145657320-eb1907d6-dcff-45ee-b1e8-a8f4d1b01e3e.png)

Press Continue.

![image](https://user-images.githubusercontent.com/46503967/145657130-10c828ef-679b-4f5d-98af-00a91e26ba81.png)

Tick the box labelled "Desktop development with C++". Your installer should now look like so:

![image](https://user-images.githubusercontent.com/46503967/145657439-e145402a-dc26-4279-8705-1a2834fba5f4.png)

Press Install. 

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

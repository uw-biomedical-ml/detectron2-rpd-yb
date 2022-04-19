Setting up Detectron2-RPD for Windows

Prerequisites: Git, which can be downloaded directly [here](https://git-scm.com/download/win) (use the default options when installing).

Step 0) Download win-detectron2-rpd.zip and extract the files. The folder should contain these files:

![image](https://user-images.githubusercontent.com/46503967/160225009-ac18eb72-a13b-4f2a-994d-77e91c6fa977.png)

&nbsp;

Step 1) Install C++ Build Tools

Prior to starting the installation process, make sure to save any important files. In order to finish the installation, a restart is required.

Run "vs_BuildTools.exe"

![image](https://user-images.githubusercontent.com/46503967/145657344-e8cf16ae-2ae4-4baf-a9f4-2637251c42eb.png)

After a couple of seconds, the following window should pop up:

![image](https://user-images.githubusercontent.com/46503967/145657320-eb1907d6-dcff-45ee-b1e8-a8f4d1b01e3e.png)

Press Continue.

![image](https://user-images.githubusercontent.com/46503967/145657130-10c828ef-679b-4f5d-98af-00a91e26ba81.png)

Tick the box labelled "Desktop development with C++" (IMPORTANT). Your installer should now look like so:

![image](https://user-images.githubusercontent.com/46503967/145657439-e145402a-dc26-4279-8705-1a2834fba5f4.png)

Press Install. After the installation is finished, restart your PC to finalize the installation.

&nbsp;

Step 2) Install Miniconda with Detectron 2

To install miniconda with detectron2 and the detectron2-rpd-yb repo, run "install.bat".

![image](https://user-images.githubusercontent.com/46503967/160224875-8200344f-0515-4464-bca4-e72d65446004.png)

After the bat file finishes, the setup is complete. By default, detectron2-rpd-yb (the repo) is located in the win-detectron2-rpd folder. You can now run the model.

&nbsp;

Step 3) Running Detectron2-RPD

Before running the model, make sure to adjust the config file (options.ini) according to your needs. The example config file included with the installation:

![image](https://user-images.githubusercontent.com/46503967/164104016-a65fb186-126e-44f6-867f-43cd28b47c5c.png)

&nbsp;

VOL/DICOM EXTRACTION

run_extract (True/False): Extract images from your input files (.vol/.dicom).

input_dir: The path to the directory containing your vol/dicom files.

extracted_dir: The path to the directory where extracted images will be stored.

input_format (vol/dicom): The format of the input files, vol or dicom.

&nbsp;

INFERENCE

dataset_name: The name of your dataset.

output_dir: The path to the directory where model predictions and other data will be stored.

run_inference (True/False): Run inference on extracted images. Note: Files must already be extracted!

create_tables (True/False): Create dataset html of model outputs. Note: Inference must already be done!

&nbsp;

VISUAL OUTPUT

create_visuals (True/False): Create visualizations of model outputs. Note: Inference must already be done and bm/bmo/im flags set!

binary_mask (True/False): Output binary mask tif files. Note: create_visuals flag must be set to True!

binary_mask_overlay (True/False): Output binary mask overlay tif files. Note: create_visuals flag must be True!

instance_mask_overlay (True/False): Output instance mask overlay tif files. Note: create_visuals flag must be True!

&nbsp;

To run the model, run Anaconda Prompt (miniconda3), which can be found on your computer using the Windows Search function.

In the prompt window, go to detectron2-rpd-yb/detectron2-rpd-pkg/src/detectron2-rpd/ (using cd).

Once inside the folder, run the following command to run the model:

python run_program.py  --config %config%

%config% = path to your config file. This is required.
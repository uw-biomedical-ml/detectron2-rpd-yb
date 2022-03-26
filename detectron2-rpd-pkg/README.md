Setting up Detectron2-RPD

Step 0) Download bat files.zip and extract files. The folder should contain these files:

![image](https://user-images.githubusercontent.com/46503967/160223809-62187a5d-37ba-4bed-99f1-a190bcb13dd7.png)

&nbsp;

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

Press Install. After the installation is finished, restart your PC to finalize the installation.

&nbsp;

Step 2) Install Miniconda with Detectron 2

To install miniconda with detectron2, run "conda_install.bat".

![image](https://user-images.githubusercontent.com/46503967/145657820-33a85b39-a157-47d9-934d-22ebea3e2913.png)

After the bat file finishes, the setup is complete. You can now run the model.

&nbsp;

Step 3) Running Detectron2-RPD

Before running the model, make sure to adjust the options.ini file. The default template:

![image](https://user-images.githubusercontent.com/46503967/160223848-b1763ea4-6114-45cf-a17b-5425ca2de618.png)

&nbsp;

VOL EXTRACTION

run_extract (true/false): Extract images from your input files (.vol/.dicom).

input_dir: The path to the directory containing your vol/dicom files.

extracted_dir: The path to the directory where extracted images will be stored.

&nbsp;

INFERENCE

dataset_name: The name of your dataset.

output_dir: The path to the directory where model predictions and other data will be stored.

run_inference (true/false): Run inference on extracted images. Note: Files must already be extracted!

create_tables (true/false): Create dataset html of model outputs. Note: Inference must already be done!

&nbsp;

VISUAL OUTPUT

create_visuals (true/false): Create visualizations of model outputs. Note: Inference must already be done and bm/bmo/im flags set!

binary_mask (true/false): Output binary mask tif files. Note: create_visuals flag must be included!

binary_mask_overlay (true/false): Output binary mask overlay tif files. Note: create_visuals flag must be included!

instance_mask_overlay (true/false): Output instance mask overlay tif files. Note: create_visuals flag must be included!

&nbsp;

To run the model, run Anaconda Prompt (miniconda3), which can be found on your computer using the Windows Search function.

In the prompt window, go to detectron2-rpd-yb/detectron2-rpd-pkg/src/detectron2-rpd-test/ (using cd).

Once inside the folder, run the following command to run the model:

python run_program.py  --config %config%

%config% = path to your config file.
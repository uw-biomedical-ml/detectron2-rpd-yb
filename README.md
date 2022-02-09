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

Press Install. After the installation is finished, restart your PC to finalize the installation.

Step 2) Install Miniconda with Detectron 2

To install miniconda with detectron2, run "conda_install.bat".

![image](https://user-images.githubusercontent.com/46503967/145657820-33a85b39-a157-47d9-934d-22ebea3e2913.png)

After the bat file finishes, the setup is complete. You can now run the model.

Step 3) Running Detectron2-RPD

To run the model, run Anaconda Prompt (miniconda3), which can be found on your computer using the Windows Search function.

In the prompt window, navigate to the detectron2-rpd-yb folder (using cd).

Once inside the folder, run the following command to run the model:

python run_program.py %dataset% %input% %output% --bm --bmo --im --ptid --imgid

Required flags:

%dataset% = name of the dataset

%input% = path to the input .csv file

%output% = path to the folder where outputs will be stored (NOTE: Folder must already exist!)

Optional flags:

bm = binary mask

bmo = binary mask overlay

im = instance mask overlay

ptid = dataset html (by ptid)

imgid = dataset html (by imgid)



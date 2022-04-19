FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
RUN apt-get update && apt-get install -y build-essential
RUN pip install opencv-python-headless \
    pip install pandas \
    pip install Jinja2 \
    pip install seaborn \
    pip install sklearn \
    pip install configargparse \
    pip install progressbar \
    pip install pydicom \
    pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
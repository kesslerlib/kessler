FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

COPY . .
RUN pip install .[dev] jupyter
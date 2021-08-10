docker run -p 8888:8888 -it --rm --gpus "device=1" -v ~/Github/AnalogNoiseReduction:/tf/ANR/ -v /mnt/extra/Datasets/:/tf/Datasets/ ll/anr 

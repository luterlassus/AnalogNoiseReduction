docker run -it --rm --gpus "device=1" -v ~/Github/AnalogNoiseReduction/:/tf/ANR/ ll/anr python ./ANR/Usage/ANR.py ./ANR/input/$1 

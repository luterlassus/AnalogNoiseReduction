# Dataset Handeling
The dataset creation process got divided into two parts to make experimentation easier. The process of creating datasets gets done using Jupiter notebooks. Notebooks make changing parameters easier during creation. 

<p align="center">
<img src = "/img/datasetCreationPipeline.png">
</p>

### CreateDataset notebook
The Createdataset notebook is the first notebook used in the creation process. The original video files are loaded using cv2. After loading, both datasets are converted to the UINT8 dtype to save space and to speed up the process. The high-quality dataset gets reshaped to the same size as the low-quality dataset. The datasets are now the same dimensions and ready to be synced. After syncing the datasets, the frames get cropped to remove props and OSD features. Cropping also gives a 16:9 aspect ratio, which looks better. The datasets are then saved separately as X and Y .npy files. Each entry n of the X dataset corresponds to the n-th entry of the Y dataset.  

### Synching
The method used for syncing the datasets is the third one to be used. The first method removed every other frame of the high-quality 60fps video and assumed the datasets to be synced if the first frames of two 30fps videos were synched. Problems occurred due to the goggle DVR not being true 30fps. Frames only get saved when the DVR googles register a frame, making link quality affect framerate. This phenomenon makes syncing harder. The second method used for syncing the datasets was printing a timer on the drone OSD, but desynchronization still happens within each second. The final approach matched each low-quality frame with the high-quality frame that gave the lowest mean squared error. The first frame got checked against the entire high res dataset, but every frame after that only gets checked against a window of frames after the previous match. This approach finally gave a perfectly synchronized dataset. 
<p align="center">
<img src="/img/DatasetCreationIll.png">
</p>

### Finalizing
The Finalizedataset notebook reshapes the dataset, normalizes it, and serializes it. Reshaping gets done so that each input sample contains multiple frames. With more input frames, the autoencoder will more accurately predict what got hidden behind the noise. Serialization allows TensorFlow to use larger than memory datasets. 

### The dataset
The currently used dataset has 40206 samples, including the validation data of about 3000.  All the videos used got recorded with Fatshark Recon goggles connected to a Beta HX115 HD drone. Beta HX115 HD records the same video it transmits, making the only difference between the goggle DVR and the onboard DVR OSD features and transmission noise. Only the transmission noise is left after cropping out the OSD features, perfect for this use case. 

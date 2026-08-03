# BeamSense: Rethinking Wireless Sensing with MU-MIMO Wi-Fi Beamforming Feedback

## Python Version

###### This is the implementation of the paper [BeamSense: Rethinking Wireless Sensing with MU-MIMO Wi-Fi Beamforming Feedback](https://doi.org/10.1016/j.comnet.2024.111020). The repository shares both the datasets and the source code of **BeamSense.**

<p align="center">
<img src="Images/BeamSense-Framework.png" width="700" height="450"
     alt="Markdown Monster icon"
     style="float: center;" />
</p>

If you find the project useful and you use this code, please cite our paper:

```

@article{haque2025beamsense,
  title={BeamSense: Rethinking wireless sensing with MU-MIMO Wi-Fi beamforming feedback},
  author={Haque, Khandaker Foysal and Zhang, Milin and Meneghello, Francesca and Restuccia, Francesco},
  journal={Computer Networks},
  pages={111020},
  year={2025},
  publisher={Elsevier}
}

```
and 

```

@inproceedings{haque2024bfa,
  title={BFA-Sense: Learning Beamforming Feedback Angles for Wi-Fi Sensing},
  author={Haque, Khandaker Foysal and Meneghello, Francesca and Restuccia, Francesco},
  booktitle={2024 IEEE International Conference on Pervasive Computing and Communications Workshops and other Affiliated Events (PerCom Workshops)},
  pages={575--580},
  year={2024},
  organization={IEEE}
}

```

## Download Dataset

(I) clone the repository with ``` git clone git@github.com:kfoysalhaque/BeamSense.git ```  <br/>

(II) ```cd BeamSense``` <br/>

(III) Then download the [BeamSense Dataset](https://ieee-dataport.org/documents/dataset-human-activity-classification-mu-mimo-bfi-and-csi#files) within the repository. <br/>
  - If the IEEE Dataport doesn't work for you, please find the Google Drive link here: https://drive.google.com/file/d/1s_Tt2ifyLYC1We7WRjRtvTMGiRQcmqyB/view?usp=drive_link <br/>
  - The full dataset is also available on Hugging Face: https://huggingface.co/datasets/foysalhaque/BeamSense

**You can also contact me (haque.k@northeastern.edu) regarding the dataset.**


(IV) Unzip the downloaded file with ``` sudo unzip Data.zip ``` <br/>


## Extract BFI from Raw pcap Files

(I) At first, split the BFIs of different stations (STAs) by executing the shell script _Feedback_split_STAs.sh_ with ``` ./Feedback_split_STAs.sh ``` <br/>

(II) Now the extracted BFIs are stored within ```BeamSense/Data/BFI/Processed/<'Environment'>/<'STA'>/FeedBack_Pcap```
<br/>
Now, export the Wireshark packet Dissections as CSV (needed for time windowing). You can also use Tshark with shell. <br/>

(III) Next, move into the directory _BFI_Extraction_ with ``` cd BeamSense/BFI_Extraction/ ``` <br/>

(IV) Ensure at minimum the following scripts are in the folder: _bfi_angles.py_, _main_new.py_, _utils.py_, and _vmatrices.py_ <br/>

(V) Execute the python script _main_new.py_ with the following configuration to extract the beamforming feedback angles (BFAs): ```BeamSense/Data/BFI/Raw AC SU 3x1 80 <'Receiver Address'> 1000 ```  <br/>

(VI) Execute the script three times to ensure all data is processed, with one of three receiver addresses each time: ```b0:b9:8a:63:55:9c```, ```38:94:ED:12:3C:25```, and ```CC:40:D0:57:EA:89``` <br/>

(VII) To see the data plotted, you may run _plot.py_ for frequency over time as a line graph, or _heatmap.py_ for a heatmap

## Creating the Learning Models

(I) At this point within the processed data environment folders, each one should have three subfolders: train, val, and test. <br/>

(II) Within Learning_Models, run the python script _create_csv_CNN.py_ to create three CSV files within each station's folder <br/>

(III) Next, run _CNN_station.py_ with the following configuration, once for each environment: ```<'Environment'> skip <'Model Name'>.keras``` <br/>

(IV) If you would like to add early stopping to the learning modelo process, follow the instructions in the comments of _CNN_station.py_ <br/>

(V) To plot the normalized values compared to the original processed ones, run the python script _plottingNormalization.py_ <br/>

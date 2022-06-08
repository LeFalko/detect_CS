# A complete graphical user interface for detecting cerebellar complex spikes  

You can find our preprint article on bioRxiv: https://www.biorxiv.org/content/biorxiv/early/2019/04/05/600536.full.pdf

You can download our GUI here
[for Windows]() and
[for Mac]().

## How to use the GUI
### [STEP 1 Labeling your data](#labeling)
### [STEP 2 ] 

### <a name="labeling">Labeling your data:</a>
This is how the GUI should look when opening it for the first time:
![](./img/Screenshot1.png)

**STEP 1: Choose parameters**
The first thing to do when opening the GUI is setting your Parameters. To do that, you have to go the "Set parameters" Button in the top left corner and choose your recorded sampling rate, as well as entering the action potential and local field potential variable names of your recordings.

**STEP 2: Upload files**
After that, you click on the "Add PC for manual labeling" button and navigate to the folder with your recordings. Select a filee and press "open" to load it. In this window you can also drag your recordings folder to the left area to access it more quickly at a later time. 
The file that is added first should be plotted instantly after loading, so you can label instantly or load more files and select on which to start. All files will be added to the "loadad files" list on the left, where you can select single files to plot or remove. 
After uploading and plotting, the GUI should look similar to this:
![](./img/Screenshot2.png)

**STEP 3: Label a recording**
To label your first recording, you can choose a observed span by clicking on one of the top buttons or using the keyboard shortcuts "Q","W","E" and using "R" and "T" to zoom. Move around the file and find CS by scrolling the bottom bar. 
Dragging across one of the plots will create a selected span on which the algorithm will train, so it is important to mark the beginning and the end of the complex spike as accurately as possible- You can deselect a CS by simply left-clicking on the selected span.
If you want to find an already selected CS you can do so by buttons directly above the plot or pressing "C" to move to the previous, or pressing "V" to move to the following CS (regarding time in the recording, not time of selection in the GUI).
This is an example of a Complex Spike recording:
![](./img/Screenshot3.png)

After labeling the first recording, select another file in your list and press "plot" to plot the new recording in the GUI and proceed with labling. 

**STEP 4: Save Training data**
When you are heppy with the selected spans, simply press the "save" button in the bottom left corner and choose a filename for your training data. the filetype should stay at .mat.

The last step is to press the "Train algorithm" button and move to the training part in your browser.

### <a name="training"> Training the network:</a> 

**STEP 1: Connecting your drive**  
You will arrive at a Google colab sheet with 6 steps. Run the first cell (click on the play button) and connect to your google drive by logging in to your google account and letting the colab sheet access your Google Drive.


**STEP 2: Train your network**    
The second cell installs the necessary tools on your drive, so just run the cell and wait until all the tools are installed.

After that, you need to run the third cell and upload the file with the training data you saved earlier, it can be found in the "Data" folder by default.

The fourth cell cell doesnt have to be changed, but it needs to run for the training to work. Changes to the number of iterations the algortihm runs, the kernel size or the max pooling size can be made here.

The fifth cells trains your network and needs no interactions other that running it.

**STEP 3: Downloading weights**
After your network is done with training, you can simply download the weights and select your preferred folder to save them in. This is the last step in your browser. You can return to the GUI after completing it.


### <a name="Second Tab">Detecting complex spikes:</a>

**STEP 1: Uploading files**  
After opening the GUI again, navigate to the "Detect CS" tab at the top. 
Choose if you want to detect on one or multiple files. Press the upper two buttons to upload a recording on which the algorithm will detect and to upload the weights you just saved from the Colab sheet in your browser. If you chose to detect on multiple files, you will need to specify an output folder as well.
This is the detection tab of the GUI when re-entering:
![](./img/Screenshot4.png)


**STEP 2: Detecting CS**
By clicking the "Detect CS" button and pressing "ok" in the dialog the detection will start. The results of the algorithm will be saved in a file with the selected file name in the beginning and "output" in the end, so the name of the resulting file ist "yourfilename_output.mat.

You can move to the post-processing tab after running the algortihm to see your result and save clusters.


### <a name="Third Tab">Post-processing:</a>

**STEP 1: Uploading files**  
Again, you first need to upload some files. This time, the loaded files have to be your recording from the last tab and the corresponding output file, so you should upload your file "filename.mat" as the upper file and your file "filename_output.mat" as the lower file.
The names are shown after selection, so you can check and reupload anytime.

**STEP 2: Plotting data**
press the "Setting for plot" button in the big plotting window to change parameters (like simple spike variable name or sampling rate) to your personally used values. These can be changed and adjusted after plotting. 
Press "Plot data" to see the CS clusters the algorithm detected on your file.
This is an example of how it should look after plotting your files:
![](./img/Screenshot5.png)

**STEP 3: Selecting clusters and saving**
In the select clusters box the individual clusters can be selected to plot individually and outliers or noisy data can be deselected to remove from the results. The "update" button replots only the selected clusters.
Lastly, the "Save selected cluster data" lets you save the selected clusters as a matlab file. Additional information to any button can be found in tooltips by hovering the information button in every box.
Sure! Here is the github link for the code I wrote:
https://github.com/ebrahimfeghhi/ripple_memory

Data generation: 
• In order to generate data, I use the
filecreate_temp_df.py to first load all the sessions I don't have data
for, where I manually set the brain region + list phase (encoding or
recall) in the file. 
• I then modify create_events_mne.py to match the
brain region + list phase I created a temp file file 
• and then run
create_events.slurm to start the cluster process. 
• I think the commands
to run it are slursh, conda activate env3, followed by sbatch
create_events.slurms. I added the environment.yaml file for env3 in the
github, so you can reproduce my environment by running conda env create
-f environment.yaml.

Analyses: 
I basically save the raw data and work with that in my
pipeline. Most of the code for that is in the directory analysis_code,
which is a bit cluttered. 
•If you go to
power_analyses/averaged_plots.ipynb within that folder you can get an
idea of how I load the data and extract power. 
•The basic idea though is
I load the neural data using the load_data_np function, and then I
created some functions like load_z_scored_power to get power.

In general paths may be hardcoded to my directories, so you might need
to change them to match your directory structure. If things are
confusing/don't work let me know! 
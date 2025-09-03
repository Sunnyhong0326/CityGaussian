# Run Experiments on OVX

## Training Configuration

Write 2 training configuration files

<details>
<summary><span style="font-weight: bold;">Important arguments for coarse model yaml configuration file</span></summary>

  ## Model Parameters (model_params)
  #### source_path
  Path to the dataset
  #### resolution
  Specifies resolution of the loaded images before training. If provided 1, 2, 4 or 8, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. If provided -1 and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.
  ## Optimization Parameters (optim_params)
  #### iterations
  Specify how many iterations to train coarse model
  #### densification_interval
  Specify how many iterations interval to densify
</details>

<details>
<summary><span style="font-weight: bold;">Important arguments for fine model yaml configuration file</span></summary>

  ## Model Parameters (model_params)
  #### source_path
  Path to the dataset
  #### resolution
  Specifies resolution of the loaded images before training. If provided 1, 2, 4 or 8, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. If provided -1 and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.
  #### use_pretrain
  Whether to use pretrain model to resume training. Always set to ```True``` in parallel fine tuning stage and ```False``` in coarse model training stage.
  #### use_full_pretrain
  Whether to use full pretrain model to resume training. If ```True```, load all coarse model into parallel fine tuning stage, else only load the corresponding block into parallel finetuning stage
  #### use_mask
  Whether to use created mask from data partition stage. If ```True```, use mask to avoid redundant modeling and L1 loss will be masked.
  #### mask_threshold
  With how much percentage the specify training view will be included to the specific block
  #### block_dim
  The whole scene will be divided into [X,Y,Z] blocks
  ## Optimization Parameters (optim_params)
  #### iterations
  Specify how many iterations to train coarse model
  #### densification_interval
  Specify how many iterations interval to densify
</details>

## Run Full Pipeline

Provide 7 large scenes bash scripts for running experiments


<details>
<summary><strong>Building</strong></summary>


<pre><code>export DATASET="building-pixsfm"
export COARSE_CONFIG="building_coarse"
export CONFIG="building_c20_r4"
export TEST_PATH="data/mill19/building-pixsfm/val"
export out_name="val"
export max_block_id=19
# For OVX
export DATA_DIR="mill19/building-pixsfm"
export RESULT="building"
</code></pre>

</details>

<details>
<summary><strong>ITRI_full</strong></summary>


<pre><code>export DATASET="ITRI_full"
export COARSE_CONFIG_NAME="itri_full_coarse"
export CONFIG_NAME="itri_full_c9_r4"
export TEST_PATH="data/ITRI_full/test"
export out_name="test"
export FINAL_DIR="output/$DATASET"
export max_block_id=8
# For OVX
export DATA_DIR="ITRI_full"
export RESULT="ITRI_full"
</code></pre>

</details>

<details>
<summary><strong>MatrixCity</strong></summary>


<pre><code>export DATASET="aerial"
export COARSE_CONFIG="mc_aerial_coarse"
export CONFIG="mc_aerial_c36"
export TEST_PATH="data/matrix_city/aerial/test"
export out_name="test"
export max_block_id=35
# For OVX
export DATA_DIR="matrix_city/aerial"
export RESULT="matrixcity"
</code></pre>

</details>

<details>
<summary><strong>NTHU_ABCDE</strong></summary>


<pre><code>export DATASET="NTHU_ABCDE"
export COARSE_CONFIG_NAME="nthu_campus_coarse"
export CONFIG_NAME="nthu_campus_c10_r4"
export TEST_PATH="data/NTHU_ABCDE/test"
export out_name="test"
export FINAL_DIR="output/$DATASET"
export max_block_id=9
# For OVX
export DATA_DIR="NTHU_ABCDE"
export RESULT="NTHU_ABCDE"
</code></pre>

</details>

<details>
<summary><strong>Residence</strong></summary>


<pre><code>export DATASET="residence-pixsfm"
export COARSE_CONFIG="residence_coarse"
export CONFIG="residence_c20_r4"
export TEST_PATH="data/urban_scene_3d/residence-pixsfm/val"
export out_name="val"
export max_block_id=19
# For OVX
export DATA_DIR="urban_scene_3d/residence-pixsfm"
export RESULT="residence"
</code></pre>

</details>

<details>
<summary><strong>Rubble</strong></summary>


<pre><code>export DATASET="rubble-pixsfm"
export COARSE_CONFIG="rubble_coarse"
export CONFIG="rubble_c9_r4"
export TEST_PATH="data/mill19/rubble-pixsfm/val"
export out_name="val"
export max_block_id=8
# For OVX
export DATA_DIR="mill19/rubble-pixsfm"
export RESULT="rubble"
</code></pre>

</details>

<details>
<summary><strong>SciArt</strong></summary>


<pre><code>export DATASET="sci-art-pixsfm"
export COARSE_CONFIG="sciart_coarse"
export CONFIG="sciart_c9_r4"
export TEST_PATH="data/urban_scene_3d/sci-art-pixsfm/val"
export out_name="val"
export FINAL_DIR="output/$DATASET"
export max_block_id=8
# For OVX
export DATA_DIR="urban_scene_3d/sci-art-pixsfm"
export RESULT="sci-art"
</code></pre>

</details>



## Run Experiments on OVX

Bash script for running experiment and saving / submitting task to OVX

#### Save Job
```bash
./scripts/save_job.sh sunnyhong-mygs-exp-8
export NUCLEUS_HOSTNAME="nucleus.tpe1.local"
export RESULT_DIR="/mnt/nfs/sunnyhong/output"
export DATASET_DIR="/mnt/nfs/sunnyhong/data"
export CODE_DIR="/mnt/nfs/sunnyhong/code"
```

#### Submit CityGS Task
```bash
./scripts/submit_task.sh sunnyhong-mygs-exp-8 \
"/run.sh \
'cp -r $CODE_DIR/CityGaussian/* /app' \
'cd /app' \
'rm -rf /app/output' \
'pip install lightning' \
'pip install wandb' \
'mkdir -p /app/data/$DATA_DIR' \
'cp -r $DATASET_DIR/$DATA_DIR /app/data/$DATA_DIR/..' \
'rm -rf /app/data/$DATA_DIR/train/data_partitions' \
'tree -L 2 /app/data/$DATA_DIR' \
'ulimit -n 524288' \
'mkdir -p /app/output/$DATASET' \
'export COARSE_CONFIG=$COARSE_CONFIG' \
'export CONFIG=$CONFIG' \
'export max_block_id=$max_block_id' \
'export out_name=$out_name' \
'export TEST_PATH=$TEST_PATH' \
'export DATASET=$DATASET' \
'bash scripts/run_citygs.sh' \
'mkdir -p $RESULT_DIR/mygs/citygs/$DATASET' \
'rm -rf /app/output/$DATASET/$COARSE_CONFIG' \
'cp -r /app/output/$DATASET/* $RESULT_DIR/mygs/citygs/$DATASET'" \
"CityGS-baseline CityGS-$RESULT"
```
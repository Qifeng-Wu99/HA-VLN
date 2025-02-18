# HAVLN-CE

We have adapt [VLN-CE](https://github.com/jacobkrantz/VLN-CE/) to our task.
Please follow VLN-CE to install necessary packages.

 To train the agent of VLN-CE, you can use the script in orignal VLN-CE.
 ```bash
 python run_VLNCE.py \
  --exp-config path/to/experiment_config.yaml \
  --run-type {train | eval | inference}
 ```
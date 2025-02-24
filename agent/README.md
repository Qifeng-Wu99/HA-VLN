# HA-VLN-CE Agents

Our agents adapt settings from [VLN-CE](https://github.com/jacobkrantz/VLN-CE/). Check the [VLN-CE](VLN-CE) for more details.


 To train the agent of VLN-CE, you can use the following scripts.
 ```bash
 pip install -r requirements_VLNCE.txt
 python run_VLNCE.py \
  --exp-config path/to/experiment_config.yaml \
  --run-type {train | eval | inference}
 ```

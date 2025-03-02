import os
import json
import os
class Calculate_Metric():
    def __init__(self, split):
        current_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_path)
        parent_dir_path = os.path.abspath(os.path.join(current_dir, os.path.pardir))

        with open(os.path.join(parent_dir_path, f'Data/HA-R2R-tools/collision_num_{split}.json')) as f:
            self.num_collisions = json.load(f)
        self.beta = {'train':1.0 - 1615.0/10819, 'val_seen':1.0 - 96.0/778, 'val_unseen':1.0 - 246.0/1839, 'test':1.0 - 549.0/3408}

    def __call__(self, metric, ep_id):
        metric['TCR'] = max(0, metric['collisions']['count'] - self.num_collisions[str(ep_id)])
        metric['CR'] = min(metric['TCR'], 1)
        metric['SR'] = metric['success'] * int(metric['TCR']==0)


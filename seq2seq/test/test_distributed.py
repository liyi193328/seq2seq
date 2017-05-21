#encoding=utf-8

import os
import sys
import json
import tensorflow
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import  ClusterConfig


cluster = {
    "ps": ["localhost:2002"],
    "worker":["localhost:3002","localhost:3003"]
}
os.environ["TF_CONFIG"] = json.dumps(
    {
        "cluster": cluster,
        "task":{
            "type": "worker",
            "index": 1
        }
    }
)

print("yes")

config = ClusterConfig()

# assert config.master == 'grpc://localhost:3002', config.master
# assert config.task_id == 1
# assert config.num_ps_replicas == 1
# assert config.num_worker_replicas == 2
# assert config.cluster_spec == tf.train.ClusterSpec(cluster)
# assert config.task_type == 'worker'
# assert not config.is_chief
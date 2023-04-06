#!/usr/bin/env bash
###
### frist worker run: NNODES=2 NODE_RANK=0 rlaunch  --cpu=20 --gpu=8 --max-wait-time=24h --memory=100000  -- tools/multi_dist_train.sh projects/configs/vovnet/petr_vov_gridmask_p4_noscale_320_allcp_2node.py 8  --work-dir work_dirs/petr_vov_gridmask_p4_noscale_320_allcp_2node
### second worker run: NNODES=2 NODE_RANK=1 rlaunch  --cpu=20 --gpu=8 --max-wait-time=24h --memory=100000  -- tools/multi_dist_train.sh projects/configs/vovnet/petr_vov_gridmask_p4_noscale_320_allcp_2node.py 8  --work-dir work_dirs/petr_vov_gridmask_p4_noscale_320_allcp_2node

NCCL_IB_HCA=$(pushd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; do cat $i/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo $i ; done; popd > /dev/null)
# [ -z "$NCCL_IB_HCA"] && NCCL_IB_HCA=mlx4_1;
export NCCL_IB_HCA
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106

NNODES=${NNODES:-2} ##Node nums
NODE_RANK=${NODE_RANK:-1} ##Node rank of different machine
CONFIG=$1 
GPUS=$2 ##Num gpus of a worker

PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"10.124.227.158"}

if [[ $NODE_RANK == 0 ]];
then
  echo "Write the ip address of node 0 to the hostfile.txt"
  ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2}'|tr -d "addr:" > hostfile.txt
fi 
MASTER_ADDR=$(cat hostfile.txt)
echo "MASTER_ADDR is : $MASTER_ADDR"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}
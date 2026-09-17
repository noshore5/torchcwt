#!/bin/bash
# bench-run-spot.sh -- fire-and-forget the CWT backend benchmark on an
# interruptible spot GPU box. Single-shot, not resumable: if the spot box
# is reclaimed mid-run, rerun by hand.
#
# Reuses EEG_Benchmarks' already-existing account infra (same AWS account
# 827938107865): instance profile eeg-gpu (S3 RW), security group eeg-ssh,
# key pair eeg-box, S3 bucket noshore-eeg-benchmarks-827938107865, SNS topic
# eeg-runs. Nothing new provisioned -- torchcwt is public, so no deploy key/
# SSM secret needed to clone it. No results-CSV promotion, no checkpointing
# -- this just builds bench/Dockerfile on-box, runs cwt_benchmark.py, ships
# the log to S3, self-terminates.
#
# Usage: scripts/bench-run-spot.sh [--keep]
#
# Needs AWS creds for account 827938107865 (aws sts get-caller-identity).

set -euo pipefail

REGION=us-east-1
BUCKET=noshore-eeg-benchmarks-827938107865
REPO_URL=https://github.com/noshore5/torchcwt.git
KEY=eeg-box
PROFILE_NAME=eeg-gpu
SNS_TOPIC_ARN=${SNS_TOPIC_ARN:-arn:aws:sns:us-east-1:827938107865:eeg-runs}
NAME=torchcwt-bench-$(date -u +%Y%m%d-%H%M%S)

KEEP=0
[ "${1:-}" = "--keep" ] && KEEP=1

# Same capacity-fallback candidates as EEG_Benchmarks' eeg-run-spot.sh.
CANDIDATES=(
  "g5.xlarge:us-east-1a"    "g5.xlarge:us-east-1b"     "g5.xlarge:us-east-1c"
  "g5.xlarge:us-east-1d"    "g5.xlarge:us-east-1f"
  "g6.xlarge:us-east-1a"    "g6.xlarge:us-east-1b"     "g6.xlarge:us-east-1c"
  "g4dn.xlarge:us-east-1c"  "g4dn.xlarge:us-east-1d"  "g4dn.xlarge:us-east-1a"
  "g4dn.xlarge:us-east-1b"  "g4dn.xlarge:us-east-1f"
)
subnet_for() {
  case "$1" in
    us-east-1a) echo subnet-057fcd8e8ed1ec050;;
    us-east-1b) echo subnet-07cdbc1058cf4f752;;
    us-east-1c) echo subnet-021f5ceeb4af26220;;
    us-east-1d) echo subnet-00252702f59bac48f;;
    us-east-1e) echo subnet-0d55ae5c23cf61927;;
    us-east-1f) echo subnet-0090cef9098097cb0;;
    *) echo "unknown AZ $1" >&2; return 1;;
  esac
}
maxprice_for() {
  case "$1" in
    g4dn.xlarge) echo 0.526;;
    g5.xlarge)   echo 1.006;;
    g6.xlarge)   echo 0.8048;;
    *)           echo 1.00;;
  esac
}

AMI=$(aws ec2 describe-images --owners amazon --region "$REGION" \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*" \
            "Name=state,Values=available" \
  --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text)
SG=$(aws ec2 describe-security-groups --region "$REGION" \
  --filters Name=group-name,Values=eeg-ssh --query 'SecurityGroups[0].GroupId' --output text)

PFX="s3://$BUCKET/exports/torchcwt-bench/$NAME"
SHUTDOWN_BEHAVIOR=terminate; [ "$KEEP" = 1 ] && SHUTDOWN_BEHAVIOR=stop

UD=$(cat <<EOF
#!/bin/bash
set -x
exec > /var/log/torchcwt-bench.log 2>&1
export HOME=/root DEBIAN_FRONTEND=noninteractive
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/sbin:/usr/bin:/bin:/snap/bin
( sleep 7200; shutdown -h now ) &   # 2h hard watchdog -- this job is small
PFX="$PFX"

finish() {
  RC=\${RC:-1}
  aws s3 cp /var/log/torchcwt-bench.log "\$PFX/boot.log" || true
  [ -f /root/bench.log ] && aws s3 cp /root/bench.log "\$PFX/bench.log" || true
  if [ "\$RC" != 0 ]; then
    aws sns publish --region $REGION --topic-arn "$SNS_TOPIC_ARN" \
      --subject "torchcwt-bench FAILED (rc=\$RC)" \
      --message "logs: \$PFX/" || true
  fi
  [ "$KEEP" = 1 ] || shutdown -h now || systemctl poweroff || halt -p
}
trap finish EXIT

apt-get update -y && apt-get install -y git >> /root/bench.log 2>&1
nvidia-smi >> /root/bench.log 2>&1 || echo "NO GPU" >> /root/bench.log

git clone --depth 1 "$REPO_URL" /root/repo >> /root/bench.log 2>&1
cd /root/repo
echo "code checkout: \$(git rev-parse --short HEAD)" >> /root/bench.log

docker build -f bench/Dockerfile -t torchcwt-bench . >> /root/bench.log 2>&1 \
  && echo "docker build ok" >> /root/bench.log \
  || { echo "docker build FAILED" >> /root/bench.log; exit 1; }

( while true; do aws s3 cp /root/bench.log "\$PFX/bench.log" 2>/dev/null; sleep 15; done ) &
TAILER=\$!

set +e
docker run --name torchcwt-bench --gpus all --rm -e PYTHONUNBUFFERED=1 torchcwt-bench \
  python -u cwt_benchmark.py --devices cuda >> /root/bench.log 2>&1
RC=\$?
set -e
kill \$TAILER 2>/dev/null || true
exit \$RC
EOF
)

for c in "${CANDIDATES[@]}"; do
  ITYPE=${c%%:*}; AZ=${c#*:}
  SUBNET=$(subnet_for "$AZ")
  PRICE=$(maxprice_for "$ITYPE")
  echo "Trying $ITYPE in $AZ (subnet $SUBNET, max \$$PRICE/hr)..."
  OUT=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" --instance-type "$ITYPE" --key-name "$KEY" \
    --subnet-id "$SUBNET" --security-group-ids "$SG" \
    --iam-instance-profile Name="$PROFILE_NAME" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":60,\"VolumeType\":\"gp3\"}}]" \
    --instance-market-options "MarketType=spot,SpotOptions={MaxPrice=$PRICE,SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}" \
    --instance-initiated-shutdown-behavior "$SHUTDOWN_BEHAVIOR" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=Project,Value=eeg}]" \
    --user-data "$UD" 2>&1) && { echo "$OUT"; echo "Launched on $ITYPE/$AZ."; echo "Logs: $PFX/"; exit 0; }
  echo "  failed: $(echo "$OUT" | tail -1)"
done
echo "No capacity across all candidates." >&2
exit 1

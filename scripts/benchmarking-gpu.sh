#/usr/bin/env bash

set -e

SLCDIR="/u/aurora-r0/staniewi/dev/hawaii/frame97/stack-no-lut"
NSLC="15 27"
for blocksize_gb in 1 4; do
    for s in 2 3; do # strides
        for tpw in 4 32; do
            for n in $NSLC; do
                slcfiles=$(find $SLCDIR -name "*185684_iw2*.h5" | head -n $n | paste -sd " ")
                strides="$((2 * s)) $s"

                outfile="dolphin_config_gpu_block${blocksize_gb}GB_strides${s}_tpw${tpw}_nslc${n}.yaml"
                dolphin config --amp-dispersion-threshold 0.25 --single-update \
                    --slc-files $slcfiles \
                    --block-size-gb $blocksize_gb \
                    --strides $strides \
                    --threads-per-worker $tpw \
                    --outfile $outfile
            done
        done
    done
done

for f in *.yaml; do
    echo "Running $f"
    echo "Removing the scratch/linked_phase directory"
    rm -rf scratch/linked_phase scratch/slc_stack.vrt
    echo "Removing the SLCs from the cache"
    vmtouch -e $SLCDIR

    dolphin run $f &>${f/.yaml/.log}
done

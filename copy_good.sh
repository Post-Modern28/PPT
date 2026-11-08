cd ~/benchmarking/PPT



for ckpt in checkpoints/trained_on_mist/*_net_G.pth; do
    epoch=$(basename "$ckpt" | cut -d'_' -f1)
    echo "=== Testing epoch $epoch ==="
    IMG_PATH=~/benchmarking/PPT/results/trained_on_mist/test_${epoch}/images
    TGT_PATH=~/benchmarking/PPT/results/trained_on_mist/test_${epoch}

    ~/benchmarking/copy_images.sh ~/benchmarking/good_clean.txt ${IMG_PATH} ${TGT_PATH}/good
    ~/benchmarking/copy_images.sh ~/benchmarking/norm_clean.txt ${IMG_PATH} ${TGT_PATH}/norm
    ~/benchmarking/copy_images.sh ~/benchmarking/combined_clean.txt ${IMG_PATH} ${TGT_PATH}/comb

    # python eval_by_vlad.py  --results_dir $IMG_PATH --output_csv ~/benchmarking/PPT/metrics/MIST/epoch_${epoch}.csv --batch_size 1 --num_workers 2
done


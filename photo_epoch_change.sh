#!/bin/bash
#SBATCH --job-name=compare_ph
#SBATCH --partition=mem
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/move_photo.out
#SBATCH --error=logs/move_photo.err

echo "Start"
cd ~/benchmarking/PPT

SRC_DIR=~/benchmarking/PPT/results/test
DST_DIR=~/benchmarking/PPT/compare

# создаём целевую папку, если не существует
mkdir -p "$DST_DIR"


# проходим по всем подпапкам test_*
ls
echo '_________________'
ls $SRC_DIR
for folder in "$SRC_DIR"/test_*/images; do
    # проверяем, что это действительно папка
    [ -d "$folder" ] || continue
    
    # извлекаем имя папки (например, test_20)
    epoch_name=$(basename "$(dirname "$folder")")
    
    # исходный файл   
    src_file="$folder/12M2102884_10_7_fake_B.png"
    # проверяем, существует ли файл
    if [ -f "$src_file" ]; then
        # целевое имя файла с суффиксом
        dst_file="$DST_DIR/12M2102884_10_7_Fake_B_${epoch_name}.png"
        
        # копируем
        cp "$src_file" "$dst_file"
        echo "✅ Скопирован: $src_file → $dst_file"
    else
        echo "⚠️ Файл не найден: $src_file"
    fi
done
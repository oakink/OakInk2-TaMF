#!/bin/bash

echo "split:" $1
echo "model:" $2
echo "model_name:" $3

confirm() {
    while true; do
        read -r -p "Are you sure you want to proceed? (y/n) " response
        response=$(echo "$response" | tr '[:upper:]' '[:lower:]')
        case "$response" in
            y|yes)
                echo "Confirmed"
                return 0
                ;;
            n|no)
                echo "Cancelled"
                return 1
                ;;
            *)
                echo "Please answer 'yes' or 'no'"
        esac
    done
}

if confirm; then
    echo "Starting ..."
else
    echo "Aborted."
    exit -1
fi

python -m oakink2_tamf.launch.sample_refine \
    --data.process_range "?(file:./asset/split/$1.txt)" \
    --data.cache_dict_filepath common/save_cache_dict/main/cache/$1.pkl \
    --debug.model_weight_filepath $2 \
    --debug.sample_save_offset $1/$3 \
    --commit

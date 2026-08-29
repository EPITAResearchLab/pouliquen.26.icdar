#!/bin/bash

# Usage: ./generate_configs.sh data.txt template.yaml

DATA_FILE="$1"
TEMPLATE_FILE="$2"

# Skip header line if present, read each line
tail -n +2 "$DATA_FILE" | while read -r fold_name projector_model_path; do
    # Skip empty lines
    [[ -z "$fold_name" ]] && continue
    
    # Generate output filename
    output_file="config_${fold_name}.yaml"
    
    # Replace placeholders and write to new file
    sed -e "s|\$(fold_name)|${fold_name}|g" \
        -e "s|\$(projector_model_path)|${projector_model_path}|g" \
        "$TEMPLATE_FILE" > "$output_file"
    
    echo "Generated: $output_file"
done

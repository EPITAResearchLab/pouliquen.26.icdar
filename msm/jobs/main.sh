for i in {0..4}; do 
    python script/prepare_data.py --config configs/onlyorigins/config_k$i.yaml
    python script/train.py --config configs/onlyorigins/config_k$i.yaml
    python script/test3.py --config configs/onlyorigins/config_k$i.yaml
done
python scripts/convert_replay_to_zarr.py \
    --demo_path data/train_data \
    --out_path  data/train_zarr \
    --tasks light_bulb_in put_money_in_safe place_wine_at_rack_location put_groceries_in_cupboard place_shape_in_shape_sorter push_buttons insert_onto_square_peg stack_cups place_cups \
    --num_demos 100 \
    --split train

python scripts/convert_replay_to_zarr.py \
    --demo_path data/test_data \
    --out_path  data/test_zarr \
    --tasks light_bulb_in put_money_in_safe place_wine_at_rack_location put_groceries_in_cupboard place_shape_in_shape_sorter push_buttons insert_onto_square_peg stack_cups place_cups \
    --num_demos 25 \
    --split test
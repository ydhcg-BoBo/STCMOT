cd src
python3 train.py --task mot \
                 --exp_id 'hm_6_4' \
                 --batch_size 16 \
                 --load_model '../models/ctdet_coco_dla_2x.pth'\
                 --data_cfg '../src/lib/cfg/visdrone.json'\
                 --gpus '0,1,2,3'\
                 --lr_step '20'\
                 --lr 7e-5 \
                 --num_epochs 30
cd ..
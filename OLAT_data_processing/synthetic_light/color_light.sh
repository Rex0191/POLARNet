# python studio_hdr_generator.py --out /mnt/bn/pico-idl-avatar2/cz/OLAT/data/hdrs_color --writeexamples
# python studio_hdr_generator.py --out /mnt/bn/pico-idl-avatar2/cz/OLAT/data/hdrs_color --scene-json scenes_400.json
# python studio_hdr_generator_full.py --out /mnt/bn/pico-idl-avatar2/cz/OLAT/data/hdrs_color --scene-json fans_diag_crisp_400.json
# 生成 400 张多风格 Studio HDR 环境图，含预览
python make_studio_hdrs.py --out_dir /mnt/bn/pico-idl-avatar2/cz/OLAT/data/hdrs_color --count 400

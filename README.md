# TẠO MÔI TRƯỜNG ẢO VÀ ACTIVE #
python3 -m venv myvenv
source myvenv/bin/activate

# CÀI ĐẶT MỘT SỐ THƯ VIỆN CẦN THIẾT #
pip install -r requirement.txt

# TRAIN #
python3 login_hugging.py
python3 main.py

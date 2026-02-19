#!/bin/bash
# rsync -av --include='*/' --include='*.png' --exclude='*' \
#     simpsone@xlogin2:~/blend2/router_output/ ./images/temp/
# # rsync -av --include='*/' --include='*.png' --exclude='*' \
# #  simpsone@xlogin2:~/blend2/router_hybrid_output/ ./images/temp/
# scp -r simpsone@xlogin2:~/blend2/router_hybrid_output/* ./images/temp/
# scp -r simpsone@xlogin2:~/blend2/threshold_plots/* ./images/temp/
scp -r simpsone@xlogin2:~/blend2/metrics_output/* ./images/temp/
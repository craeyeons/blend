#!/bin/bash
rsync -av --include='*/' --include='*.png' --exclude='*' \
    simpsone@xlogin2:~/blend2/router_output_multi/ ./images/temp_multi/
scp -r simpsone@xlogin2:~/blend2/threshold_plots_multi/* ./images/temp_multi/
scp -r simpsone@xlogin2:~/blend2/metrics_output_multi/* ./images/temp_multi/

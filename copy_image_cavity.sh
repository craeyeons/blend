#!/bin/bash
rsync -av --include='*/' --include='*.png' --exclude='*' \
    simpsone@xlogin2:~/blend2/router_output_cavity/ ./images/temp_cavity/
scp -r simpsone@xlogin2:~/blend2/threshold_plots_cavity/* ./images/temp_cavity/
scp -r simpsone@xlogin2:~/blend2/metrics_output_cavity/* ./images/temp_cavity/
scp -r simpsone@xlogin2:~/blend2/timing_output/* ./images/temp_cavity/
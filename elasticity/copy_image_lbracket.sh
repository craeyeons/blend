#!/bin/bash
rsync -av --include='*/' --include='*.png' --exclude='*' \
    simpsone@xlogin2:~/blend2/elasticity/router_output/l_bracket/ ./images/temp_lbracket/
scp -r simpsone@xlogin2:~/blend2/elasticity/threshold_plots_lbracket/* ./images/temp_lbracket/
scp -r simpsone@xlogin2:~/blend2/elasticity/metrics_output_lbracket/* ./images/temp_lbracket/
scp simpsone@xlogin2:~/blend2/elasticity/models/pinn_l_bracket_loss.png ./images/temp_lbracket/
scp simpsone@xlogin2:~/blend2/elasticity/results/fdm_l_bracket.png ./images/temp_lbracket/

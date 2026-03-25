#!/bin/bash
rsync -av --include='*/' --include='*.png' --exclude='*' \
    simpsone@xlogin2:~/blend2/elasticity/router_output/plate_with_hole/ ./images/temp_plate/
scp -r simpsone@xlogin2:~/blend2/elasticity/threshold_plots_plate/* ./images/temp_plate/
scp -r simpsone@xlogin2:~/blend2/elasticity/metrics_output_plate/* ./images/temp_plate/
scp simpsone@xlogin2:~/blend2/elasticity/models/pinn_plate_with_hole_loss.png ./images/temp_plate/
scp simpsone@xlogin2:~/blend2/elasticity/results/fdm_plate_with_hole.png ./images/temp_plate/

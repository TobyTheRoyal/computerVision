
fnames=()
fnames[0]=01_init_kernel_k.png
fnames[1]=02_init_kernel_kL.png
fnames[2]=03_init_kernel_kR.png
fnames[3]=04_init_kernel_kGauss_2d.png
fnames[4]=05_init_kernel_kGauss_1d.png
fnames[5]=06_swf_iteration_it_0.png
fnames[6]=06_swf_iteration_it_1.png
fnames[7]=06_swf_iteration_it_2.png
fnames[8]=06_swf_iteration_it_3.png
fnames[9]=07_diff_iteration_it_0.png
fnames[10]=07_diff_iteration_it_1.png
fnames[11]=07_diff_iteration_it_2.png
fnames[12]=07_diff_iteration_it_3.png
fnames[13]=08_end_swf.png
fnames[14]=09_gauss_2d_iteration_it_0.png
fnames[15]=09_gauss_2d_iteration_it_1.png
fnames[16]=09_gauss_2d_iteration_it_2.png
fnames[17]=09_gauss_2d_iteration_it_3.png
fnames[18]=10_end_gaus_2d.png
fnames[19]=11_gauss_1d_iteration_it_0.png
fnames[20]=11_gauss_1d_iteration_it_1.png
fnames[21]=11_gauss_1d_iteration_it_2.png
fnames[22]=11_gauss_1d_iteration_it_3.png
fnames[23]=12_end_gaus_1d.png
fnames[24]=13_bgr2yuv.png
fnames[25]=14_channel_u.png
fnames[26]=14_channel_v.png
fnames[27]=14_channel_y.png
fnames[28]=15_yuv2bgr.png
fnames[29]=16_bonus_bil_iteration_it_0.png
fnames[30]=16_bonus_bil_iteration_it_1.png
fnames[31]=16_bonus_bil_iteration_it_2.png
fnames[32]=16_bonus_bil_iteration_it_3.png
fnames[33]=17_end_bonus_bil.png


for tc in "baer" "dragonball" "haus" "jet" "kamera" "starwars1" "starwars2"
do
    if [ -d data/ref_x64/${tc} ]
    then
      mkdir -p dif/${tc}
      for fname in ${fnames[*]}
      do
        if [ -f output/${tc}/${fname} ]
        then
          convert data/ref_x64/${tc}/${fname} output/${tc}/${fname} -compose difference -composite -negate -contrast-stretch 0 dif/${tc}/${fname}
        else
          convert -size 300x300 xc:red dif/${tc}/${fname}
        fi
      done
    fi
done



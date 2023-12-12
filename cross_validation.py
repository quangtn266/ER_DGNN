import subprocess

ls = [3]#,4,8,9]

for i in range(1,15):
    for j in range(1,11):
    #for j in ls:
        subprocess.call('python ./train.py '
                           +'--train_dir=./191121_ck_ms_ed_glu/Test%d/test%d '%(i,j)
                       +'--fold=%d'%j,
                            shell=True)

        print ('Training Done for %d in %d fold' %(i,j))        

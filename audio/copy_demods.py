import os
import numpy as np
import sys
'''
Description:
scp that shit over here

Date: 

Author: Hunter Akins
'''



def copy_over_chunks(freq, num_chunks):
    os.chdir('npy_files/' + str(freq))
    dirs = os.listdir()
    print(dirs)
    for i in range(num_chunks):
        if 'chunk'+str(i) not in dirs:
            os.mkdir('chunk'+str(i))
        os.chdir('chunk'+str(i))
        os.system('scp fakins@tscc-login.sdsc.edu:/oasis/tscc/scratch/fakins/' + str(freq) + '/\*chunk_'+str(i)+'.npy .')
        os.system('scp fakins@tscc-login.sdsc.edu:/oasis/tscc/scratch/fakins/' + str(freq) + '/chunk_'+str(i)+'_f_grid.npy .')
        x = np.load('chunk_' + str(i) + '_f_grid.npy')
        print('Mean freq for chunk ',i, np.mean(x))
        os.chdir('..')
    os.chdir('..')

def copy_over_f_demod(freq):
    os.chdir('npy_files/' + str(freq))
    dirs = os.listdir()
    print(dirs)
    for i in range(1):
        if 'chunk'+str(i) not in dirs:
            os.mkdir('chunk'+str(i))
        os.chdir('chunk'+str(i))
        os.system('scp fakins@tscc-login.sdsc.edu:/oasis/tscc/scratch/fakins/' + str(freq) + '/\*_nb.npy .')
        os.system('scp fakins@tscc-login.sdsc.edu:/oasis/tscc/scratch/fakins/' + str(freq) + '/nb_f_grid.npy .')
        x = np.load('nb_f_grid.npy')
        print('Mean freq for chunk ',i, np.mean(x))
        os.chdir('..')
    os.chdir('../..')

def copy_over_fests(freq, N=1500, delta_n=750):
    root = 'fakins@tscc-login.sdsc.edu:/oasis/tscc/scratch/fakins/fests/'+str(freq) + '_' + str(N) + '_' + str(delta_n) +  '_'
    os.chdir('npy_files/fests')
    os.system('scp ' + root + 'fhat.npy .')
    os.system('scp ' + root + 'fhat_err.npy .')
    os.system('scp ' + root + 'fhat_amp.npy .')


freq = int(sys.argv[1])
#copy_over_chunks(freq, 1)
copy_over_f_demod(freq)
#copy_over_fests(freq)


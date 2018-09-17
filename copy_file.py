import shutil

ori='../DeliFGAN/samples/sliced/'
dest='sliced/'
number=3000

for i in range(1000):
    num=number-i
    for j in range(1,9):
        for k in range(1,9):
            file_name='fake_samples_'+str(num)+'.png_0'+str(j)+'_0'+str(k)+'.png'
            shutil.copy(ori+file_name,dest)
			
